import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from acd.checkpoints import load_model, model_config, read_checkpoint, validate_resume
from acd.data import evaluation_dataset
from tools.download_models import checksum, download, select_models
import train


def tiny_model(*args, **kwargs):
    model = torch.nn.Linear(3, 6)
    return model, list(model.parameters())


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads(Path('configs/res56_res20.json').read_text())
        self.config.update(method='ce', workers=0, epochs=3, batch_size=2, milestones=[1, 2])
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        self.dataset = TensorDataset(torch.arange(18, dtype=torch.float32).reshape(6, 3)/18,
                                     torch.tensor([0, 1, 2, 3, 4, 5]))

    def run_main(self, *arguments):
        command = ['train.py', '--config', str(self.config_path), '--data-root', str(self.root),
                   '--device', 'cpu', *map(str, arguments)]
        with patch('sys.argv', command), contextlib.redirect_stdout(io.StringIO()), patch('train.build', tiny_model):
            train.main()

    def test_metrics_and_empty_dataset(self):
        scores = torch.tensor([[8., 7., 6., 5., 4., 3.], [6., 5., 4., 3., 2., 1.]])
        labels = torch.tensor([0, 5])
        metrics = train.evaluate(torch.nn.Identity(), DataLoader(TensorDataset(scores, labels), batch_size=1), 'cpu')
        self.assertEqual((metrics['top1'], metrics['top5'], metrics['samples']), (50., 50., 2))
        self.assertAlmostEqual(metrics['ce'], torch.nn.functional.cross_entropy(scores, labels).item(), places=6)
        with self.assertRaisesRegex(ValueError, 'empty'):
            train.evaluate(torch.nn.Identity(), [], 'cpu')

    def test_evaluation_builds_only_test_split(self):
        with patch('acd.data.datasets.CIFAR100') as cifar:
            evaluation_dataset({'dataset': 'cifar100'}, self.root)
            self.assertEqual(cifar.call_count, 1)
            self.assertIs(cifar.call_args.kwargs['train'], False)
        with patch('acd.data.datasets.ImageFolder') as imagenet:
            imagenet.return_value.classes = list(range(1000))
            evaluation_dataset({'dataset': 'imagenet'}, self.root)
            self.assertEqual(imagenet.call_count, 1)
            self.assertEqual(imagenet.call_args.args[0], str(self.root)+'/val')

    def test_eval_cli_needs_no_teacher_or_training_split(self):
        model, _ = tiny_model()
        checkpoint = self.root / 'student.pth'
        torch.save({'state_dict': model.state_dict(), 'model_config': model_config(self.config)}, checkpoint)
        metrics_path = self.root / 'metrics.json'
        with patch('train.datasets_for', side_effect=AssertionError('Training split requested')), \
             patch('train.evaluation_dataset', return_value=self.dataset):
            self.run_main('--evaluate', checkpoint, '--metrics-json', metrics_path)
        metrics = json.loads(metrics_path.read_text())
        self.assertEqual(metrics['samples'], 6)
        self.assertEqual(metrics['checkpoint_sha256'], checksum(checkpoint))
        self.assertEqual(metrics['role'], 'student')

    def test_teacher_eval_uses_affine_teacher_identity(self):
        model, _ = tiny_model()
        checkpoint = self.root / 'teacher.pth'
        torch.save({'state_dict': model.state_dict(), 'model_config': model_config(self.config, 'teacher')}, checkpoint)
        with patch('train.datasets_for', side_effect=AssertionError('Training split requested')), \
             patch('train.evaluation_dataset', return_value=self.dataset):
            self.run_main('--evaluate', checkpoint, '--evaluate-teacher')

    def test_weights_only_loading_and_metadata_check(self):
        model, _ = tiny_model()
        checkpoint = self.root / 'model.pth'
        torch.save({'state_dict': model.state_dict(), 'model_config': model_config(self.config)}, checkpoint)
        load_model(checkpoint, model, config=self.config)
        with self.assertRaisesRegex(ValueError, 'differs'):
            load_model(checkpoint, model, config={**self.config, 'student': 'resnet32'})
        torch.save({'state_dict': model.state_dict(), 'legacy_rng': np.arange(3)}, checkpoint)
        with self.assertRaisesRegex(ValueError, 'legacy'):
            read_checkpoint(checkpoint)
        load_model(checkpoint, model, allow_legacy=True)

    def test_smoke_and_export_cannot_resume(self):
        for checkpoint in [{'status': 'incomplete_smoke'}, {'state_dict': {}}]:
            with self.assertRaisesRegex(ValueError, 'epoch-boundary'):
                validate_resume(checkpoint, self.config, None)

    def test_resume_matches_uninterrupted_training(self):
        full = self.root / 'full'
        resumed = self.root / 'resumed'
        with patch('train.datasets_for', return_value=(self.dataset, self.dataset)):
            self.run_main('--output', full)
            original_evaluate = train.evaluate
            calls = 0

            def interrupted_evaluation(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interruption')
                return original_evaluate(*args)

            with patch('train.evaluate', side_effect=interrupted_evaluation):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    self.run_main('--output', resumed)
            self.run_main('--output', resumed, '--resume', resumed / 'latest.pth')
        expected = read_checkpoint(full / 'latest.pth')
        actual = read_checkpoint(resumed / 'latest.pth')
        for key, value in expected['student'].items():
            self.assertTrue(torch.equal(value, actual['student'][key]), key)
        self.assertEqual(expected['scheduler'], actual['scheduler'])
        self.assertEqual(expected['validation'], actual['validation'])
        self.assertEqual((full / 'history.jsonl').read_text(), (resumed / 'history.jsonl').read_text())

    def test_completed_resume_into_new_directory_keeps_checkpoint(self):
        original = self.root / 'original'
        copied = self.root / 'copied'
        with patch('train.datasets_for', return_value=(self.dataset, self.dataset)):
            self.run_main('--output', original)
            self.run_main('--resume', original / 'latest.pth', '--output', copied)
        self.assertEqual(read_checkpoint(copied / 'latest.pth')['epoch'], 3)
        final = json.loads((copied / 'final.json').read_text())
        self.assertEqual(final['checkpoint_sha256'], checksum(copied / 'latest.pth'))

    def test_fresh_run_rejects_existing_output(self):
        output = self.root / 'run'
        with patch('train.datasets_for', return_value=(self.dataset, self.dataset)):
            self.run_main('--output', output)
            previous = (output / 'history.jsonl').read_text()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_main('--output', output)
        self.assertEqual((output / 'history.jsonl').read_text(), previous)

    def test_control_selection_keeps_variants_and_reuses_teacher(self):
        teacher = {'filename': 'teacher.pth', 'role': 'teacher',
                   'model_config': {'architecture': 'vgg13'}}
        students = [{'filename': name, 'role': 'student', 'pair': 'vgg13_vgg8', 'teacher': 'vgg13'}
                    for name in ('linear.pth', 'cosine.pth')]
        manifest = {'models': students, 'teacher_dependencies': [teacher]}
        self.assertEqual(select_models(manifest, pair='vgg13_vgg8'), students + [teacher])
        self.assertEqual(select_models(manifest, filename='cosine.pth'), [students[1], teacher])
        self.assertEqual(select_models(manifest, pair='vgg13_vgg8', student_only=True), students)
        self.assertEqual(select_models(manifest, filename='teacher.pth'), [teacher])
        with self.assertRaisesRegex(ValueError, 'No matching'):
            select_models(manifest, pair='missing')
        with self.assertRaisesRegex(ValueError, 'missing a teacher'):
            select_models({'models': students}, pair='vgg13_vgg8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_models({'models': students + students})

    def test_downloader_uses_teacher_release_and_downloads_matching_config(self):
        contents = b'model or configuration'
        source = self.root / 'source'
        source.write_bytes(contents)
        student = {'filename': 'student.pth', 'role': 'student', 'teacher': 'vgg13',
                   'sha256': checksum(source), 'config_file': 'student.json'}
        teacher = {'filename': 'teacher.pth', 'role': 'teacher', 'model_config': {'architecture': 'vgg13'},
                   'sha256': checksum(source), 'download_base_url': 'https://teacher.invalid/old'}
        cfg = {'filename': 'student.json', 'sha256': checksum(source)}
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'models': [student], 'teacher_dependencies': [teacher],
                                        'configs': [cfg], 'download_base_url': 'https://student.invalid/new'}))
        command = ['download_models', '--manifest', str(manifest), '--model', 'student.pth',
                   '--output', str(self.root / 'models')]
        with patch('sys.argv', command), contextlib.redirect_stdout(io.StringIO()), \
             patch('tools.download_models.urlopen', side_effect=lambda *a, **k: io.BytesIO(contents)) as request:
            from tools.download_models import main
            main()
        self.assertEqual([call.args[0] for call in request.call_args_list], [
            'https://student.invalid/new/student.pth', 'https://teacher.invalid/old/teacher.pth',
            'https://student.invalid/new/student.json'])
        self.assertEqual(sorted(p.name for p in (self.root / 'models').iterdir()),
                         ['student.json', 'student.pth', 'teacher.pth'])

    def test_downloader_verifies_and_reuses_without_overwriting(self):
        source = self.root / 'original'
        source.write_bytes(b'checked model weights')
        entry = {'filename': 'model.pth', 'sha256': checksum(source)}
        output = self.root / 'download'
        with patch('tools.download_models.urlopen', return_value=io.BytesIO(source.read_bytes())) as request:
            destination = download(entry, output, 'https://example.invalid')
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            download(entry, output, 'https://example.invalid')
            self.assertEqual(request.call_count, 1)
        destination.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Existing file'):
            download(entry, output, 'https://example.invalid')
        with patch('tools.download_models.urlopen', return_value=io.BytesIO(b'corrupt')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                download(entry, self.root / 'bad', 'https://example.invalid')
        self.assertEqual(list((self.root / 'bad').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
