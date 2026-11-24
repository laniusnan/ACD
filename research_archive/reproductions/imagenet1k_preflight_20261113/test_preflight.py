import json
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import TensorDataset

import data
import methods
import run


torch.set_num_threads(2)


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = json.loads((run.HERE / "protocol.json").read_text())

    def test_eight_distinct_arms(self):
        self.assertEqual(len(set(methods.ARMS.values())), 8)
        self.assertEqual(methods.ARMS["I7"], ("cosine", "sicd"))

    def test_cosine_supports_1000_classes_512_features_without_rng_change(self):
        initial = torch.randn(1000, 512)
        rng = torch.get_rng_state().clone()
        head = methods.CosineHead(initial, 30)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        features = torch.randn(3, 512)
        expected = 30 * functional.linear(functional.normalize(features, dim=1),
                                           functional.normalize(initial, dim=1))
        torch.testing.assert_close(head(features), expected)
        self.assertFalse(head.scale.requires_grad)

    def test_pairing_preserves_backbone_head_and_global_rng(self):
        rng = torch.get_rng_state().clone()
        linear, linear_backbone, first = methods.build_student("resnet34_resnet18", "linear", 30, 3101)
        cosine, cosine_backbone, second = methods.build_student("resnet34_resnet18", "cosine", 30, 3101)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn(id(cosine.fc.weight), {id(parameter) for parameter in cosine_backbone})
        self.assertEqual(len(linear_backbone), len(cosine_backbone))

    def test_full_kl_decomposition(self):
        student = torch.randn(5, 1000, dtype=torch.float64)
        teacher = torch.randn_like(student)
        labels = torch.tensor([2, 4, 8, 16, 32])
        temperature = 4
        teacher_probability = (teacher / temperature).softmax(1)
        student_probability = (student / temperature).softmax(1)
        teacher_target = teacher_probability.gather(1, labels[:, None])
        student_target = student_probability.gather(1, labels[:, None])
        teacher_binary = torch.cat((teacher_target, 1 - teacher_target), dim=1)
        student_binary = torch.cat((student_target, 1 - student_target), dim=1)
        binary = functional.kl_div(student_binary.log(), teacher_binary, reduction="batchmean") * temperature ** 2
        conditional = methods.conditional_kl(student, teacher, labels, temperature)
        torch.testing.assert_close(methods.full_kl(student, teacher, temperature), binary + conditional,
                                   atol=1e-12, rtol=1e-10)

    def test_standardization_affine_invariance_and_sample_std(self):
        logits = torch.randn(7, 1000, dtype=torch.float64)
        scale = torch.rand(7, 1, dtype=torch.float64) + 1
        shift = torch.randn(7, 1, dtype=torch.float64)
        standardized = methods.standardize(logits)
        torch.testing.assert_close(standardized, methods.standardize(scale * logits + shift))
        torch.testing.assert_close(standardized.std(1, correction=1), torch.ones(7, dtype=torch.float64))

    def test_extreme_conditional_logits_have_finite_gradients(self):
        student = (1000 * torch.randn(4, 1000)).requires_grad_()
        teacher = (1000 * torch.randn(4, 1000)).requires_grad_()
        loss = methods.conditional_kl(student, teacher, torch.arange(4), 4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_constant_logits_finite_after_standardization(self):
        logits = torch.ones(3, 1000, requires_grad=True)
        value = methods.standardize(logits).square().sum()
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_detached_matching_uses_backbone_gradient_norm(self):
        features = torch.randn(4, 8, requires_grad=True)
        classifier = torch.randn(1000, 8, requires_grad=True)
        logits = functional.linear(features, classifier)
        teacher = torch.randn(4, 1000, requires_grad=True)
        matched, diagnostics = methods.matched_relation(logits, teacher, torch.arange(4), [features],
                                                         4, 1e-6, 1e-12)
        self.assertFalse(diagnostics["match_ratio"].requires_grad)
        gradient = torch.autograd.grad(matched, features, retain_graph=True)[0]
        torch.testing.assert_close(gradient.norm(), diagnostics["reference_gradient_norm"], rtol=2e-5, atol=1e-8)
        matched.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(classifier.grad)

    def test_all_objectives_finite_and_teacher_frozen(self):
        for arm in methods.ARMS:
            with self.subTest(arm=arm):
                features = torch.randn(3, 8, requires_grad=True)
                logits = features @ torch.randn(8, 1000)
                teacher = torch.randn(3, 1000, requires_grad=True)
                total, diagnostics = methods.objective(logits, teacher, torch.arange(3), [features], arm, self.config)
                total.backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertIsNone(teacher.grad)
                self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))


class StateTests(unittest.TestCase):
    def test_checkpoint_replays_model_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = nn.Sequential(nn.Linear(6, 9), nn.Dropout(.4), nn.Linear(9, 3))
            optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.1)
            path = Path(temporary) / "latest.pth"
            identity = {"source": "known"}

            def update():
                inputs = torch.randn(4, 6)
                optimizer.zero_grad()
                model(inputs).square().sum().backward()
                optimizer.step()
                scheduler.step()
                return random.random(), float(np.random.rand())

            update()
            run.save_checkpoint(path, model, optimizer, scheduler, 1, identity)
            expected_random = update()
            expected = methods.tensor_digest(model.state_dict())
            expected_lr = scheduler.get_last_lr()
            self.assertEqual(run.restore_checkpoint(path, model, optimizer, scheduler, identity), 1)
            self.assertEqual(update(), expected_random)
            self.assertEqual(methods.tensor_digest(model.state_dict()), expected)
            self.assertEqual(scheduler.get_last_lr(), expected_lr)
            with self.assertRaisesRegex(ValueError, "Resume refuses"):
                run.restore_checkpoint(path, model, optimizer, scheduler, {"source": "changed"})

    def test_gate_refuses_missing_data_and_unverified_sicd(self):
        config = json.loads((run.HERE / "protocol.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            blockers = run.formal_gate(config, "I7", None, Path(temporary) / "missing.json")
        self.assertGreaterEqual(len(blockers), 4)
        self.assertTrue(any("source missing" in reason for reason in blockers))
        self.assertTrue(any("candidate" in reason for reason in blockers))

    def test_freezing_recipe_does_not_bypass_missing_source(self):
        config = json.loads((run.HERE / "protocol.json").read_text())
        config["status"] = "frozen_after_real_data_preflight"
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.json"
            audit.write_text("{}")
            blockers = run.formal_gate(config, "I7", temporary, audit)
        self.assertTrue(any("source missing" in reason for reason in blockers))


class DataTests(unittest.TestCase):
    def test_subset_is_not_accepted_as_imagenet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "train/n00000000"
            folder.mkdir(parents=True)
            Image.new("RGB", (256, 256)).save(folder / "one.jpg")
            with self.assertRaisesRegex(ValueError, "Not full ImageNet"):
                data.open_split(root, "train", ["n00000000"])

    def test_mapping_order_and_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            index = {str(number): [f"n{number:08d}", "test"] for number in range(1000)}
            data.save_json(path, index)
            self.assertEqual(len(data.class_names(path)), 1000)
            index["0"], index["1"] = index["1"], index["0"]
            data.save_json(path, index)
            with self.assertRaisesRegex(ValueError, "Class order"):
                data.class_names(path)

    def test_missing_data_does_not_write_passing_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            with self.assertRaises(FileNotFoundError):
                data.audit(Path(temporary) / "absent", Path(temporary) / "index.json", output)
            self.assertFalse(output.exists())

    def test_atomic_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            data.save_json(path, {"status": "valid"})
            with self.assertRaises(ValueError):
                data.save_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "valid"})


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads((run.HERE / "protocol.json").read_text())
        self.config.update(status="frozen_after_real_data_preflight", epochs=2, batch_size=2, workers=0)
        self.audit = self.root / "audit.json"
        self.audit.write_text("{}")
        generator = torch.Generator().manual_seed(123)
        dataset = TensorDataset(torch.randn(4, 6, generator=generator), torch.arange(4))
        self.teacher = nn.Linear(6, 1000).requires_grad_(False).eval()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(run, "formal_gate", return_value=[]))
        self.stack.enter_context(patch.object(data, "audited_datasets", return_value={"train": dataset, "val": dataset}))
        self.stack.enter_context(patch.object(run, "load_teacher", return_value=(self.teacher, {"identity": "unit_test"})))
        self.stack.enter_context(patch.object(run, "environment", return_value={"kind": "cpu_unit_test"}))

        def build(*arguments):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1234)
                model = nn.Sequential(nn.Linear(6, 8), nn.Dropout(.2), nn.Linear(8, 1000))
            return model, list(model[0].parameters()), {"kind": "unit_test"}

        self.stack.enter_context(patch.object(methods, "build_student", side_effect=build))
        self.evaluation = self.stack.enter_context(patch.object(run, "evaluate", return_value={
            "count": 50000, "top1": 76.13, "top5": 90.0, "kind": "mock_unit_test_not_accuracy"}))

    def arguments(self, name, resume=False):
        return SimpleNamespace(arm="I2", data_root=str(self.root), audit=self.audit, device="cpu",
                               output=self.root / name, resume=resume)

    def test_completion_reload_and_no_overwrite(self):
        args = self.arguments("complete")
        self.assertEqual(run.train(args, self.config), 0)
        result = json.loads((args.output / "final.json").read_text())
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["reloaded_checkpoint_sha256"], data.sha256_file(args.output / "latest.pth"))
        self.assertEqual(self.evaluation.call_count, 2)
        with self.assertRaises(FileExistsError):
            run.train(args, self.config)

    def test_interrupted_epoch_resume_matches_uninterrupted_run(self):
        reference = self.arguments("reference")
        run.train(reference, self.config)
        interrupted = self.arguments("interrupted")
        original_step = run.step
        calls = 0

        def interrupt_after_epoch(*arguments):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("unit_test_interruption")
            return original_step(*arguments)

        with patch.object(run, "step", side_effect=interrupt_after_epoch):
            with self.assertRaisesRegex(RuntimeError, "unit_test_interruption"):
                run.train(interrupted, self.config)
        interrupted.resume = True
        run.train(interrupted, self.config)
        expected = torch.load(reference.output / "latest.pth", weights_only=True)
        resumed = torch.load(interrupted.output / "latest.pth", weights_only=True)
        self.assertEqual(methods.tensor_digest(expected["student"]), methods.tensor_digest(resumed["student"]))
        self.assertEqual(expected["scheduler"], resumed["scheduler"])

    def test_resume_after_startup_failure_without_checkpoint(self):
        args = self.arguments("startup_failure")
        self.evaluation.side_effect = RuntimeError("unit_test_validation_failure")
        with self.assertRaisesRegex(RuntimeError, "unit_test_validation_failure"):
            run.train(args, self.config)
        self.assertFalse((args.output / "latest.pth").exists())
        self.evaluation.side_effect = None
        args.resume = True
        self.assertEqual(run.train(args, self.config), 0)


if __name__ == "__main__":
    unittest.main()
