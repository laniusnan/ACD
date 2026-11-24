import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acd.checkpoints import model_config
from acd.data import datasets_for, stratified_split
from acd.heads import CosineHead, RotatingETFHead, optimizer_groups
from acd.losses import distillation_loss
from acd.models import build, logits
import test_workflows


class SevenArmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_rotating_head_matches_etf_initialization_and_preserves_gram(self):
        weight = torch.randn(5, 7, dtype=torch.float64)
        rng = torch.get_rng_state()
        free = CosineHead(weight, etf=True)
        torch.set_rng_state(rng)
        rotating = RotatingETFHead(weight)
        x = torch.randn(8, 7, dtype=torch.float64)
        self.assertTrue(torch.allclose(free(x), rotating(x), atol=1e-12))
        before = rotating.frame.detach().clone()
        optimizer = torch.optim.SGD(optimizer_groups(rotating), lr=.04, momentum=.9, weight_decay=.01)
        for _ in range(12):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(rotating(x), torch.arange(8) % 5).backward()
            rotating.project_gradient()
            optimizer.step()
            rotating.retract(optimizer)
            self.assertLess(rotating.geometry_error(), 1e-12)
            q = rotating.frame
            m = optimizer.state[q]['momentum_buffer']
            self.assertLess(float((q.T @ m + m.T @ q).detach().abs().max()), 1e-10)
        self.assertFalse(torch.equal(before, rotating.frame))

    def test_lift_backbone_and_initialization_paired_across_seven_arms(self):
        arms = [('kd', False, False), ('angular_kd', False, False),
                ('angular_kd', True, False), ('angular_kd', False, True),
                ('acd', False, False), ('acd', True, False), ('acd', False, True)]
        reference = None
        for method, etf, rotating in arms:
            torch.manual_seed(91)
            model, backbone = build('resnet20', angular=method != 'kd', etf=etf,
                                    rotating_etf=rotating, feature_lift=128)
            state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('fc.classifier.')}
            if reference is None:
                reference = state
            else:
                for key in state:
                    self.assertTrue(torch.equal(state[key], reference[key]), key)
            ids = {id(p) for p in backbone}
            self.assertTrue(all(id(p) in ids for p in model.fc.projection.parameters()))
            self.assertTrue(all(id(p) not in ids for p in model.fc.classifier.parameters()))
            scores = logits(model, torch.randn(2, 3, 32, 32))
            loss, _ = distillation_loss(scores, torch.randn_like(scores), torch.tensor([1, 2]),
                                        backbone, {'method': method})
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_stratified_holdout_has_no_image_or_view_overlap(self):
        targets = np.repeat(np.arange(100), 500).tolist()
        training, validation = stratified_split(targets, 50, 20261118)
        self.assertEqual((len(training), len(validation)), (45000, 5000))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(50000)))
        self.assertEqual((training, validation), stratified_split(targets, 50, 20261118))
        self.assertTrue(np.all(np.bincount(np.asarray(targets)[validation]) == 50))
        with self.assertRaises(ValueError):
            stratified_split(targets, 500, 7)

    def test_tuning_never_loads_official_test_set(self):
        class FakeCifar:
            def __init__(self, root, train, **kwargs):
                if not train:
                    raise AssertionError('Official test images requested during recipe selection')
                self.data = np.zeros((12, 32, 32, 3), dtype=np.uint8)
                self.targets = [0]*6 + [1]*6
            def __len__(self):
                return len(self.targets)
        with patch('acd.data.datasets.CIFAR100', FakeCifar):
            train, validation = datasets_for({'dataset':'cifar100','validation_per_class':2,
                                              'augmentation':'double_view'}, '.')
        self.assertEqual(len(train), 16)
        self.assertEqual(len(validation), 4)
        self.assertEqual(train.base.targets, [0]*4 + [1]*4)

    def test_new_heads_have_distinct_checkpoint_identity(self):
        cfg = {'method':'acd','student':'resnet20','teacher':'resnet56','dataset':'cifar100'}
        original = model_config(cfg)
        self.assertNotEqual(original, model_config({**cfg, 'feature_lift':128}))
        self.assertNotEqual(original, model_config({**cfg, 'rotating_etf':True}))
        self.assertEqual(model_config(cfg, 'teacher'),
                         model_config({**cfg, 'feature_lift':128, 'rotating_etf':True}, 'teacher'))


class StagedResumeTests(unittest.TestCase):
    setUp = test_workflows.WorkflowTests.setUp
    run_main = test_workflows.WorkflowTests.run_main

    def test_staged_stop_is_resumable_and_never_reported_complete(self):
        full = self.root / 'full'
        staged = self.root / 'staged'
        with patch('train.datasets_for', return_value=(self.dataset, self.dataset)):
            self.run_main('--output', full)
            self.run_main('--output', staged, '--stop-after-epoch', 1)
            self.assertFalse((staged/'final.json').exists())
            self.assertEqual(json.loads((staged/'stage.json').read_text())['epoch'], 1)
            self.run_main('--output', staged, '--resume', staged/'latest.pth')
        self.assertEqual((full/'history.jsonl').read_text(), (staged/'history.jsonl').read_text())
        expected = torch.load(full/'latest.pth', weights_only=True)
        actual = torch.load(staged/'latest.pth', weights_only=True)
        for key in expected['student']:
            self.assertTrue(torch.equal(expected['student'][key], actual['student'][key]))


if __name__ == '__main__':
    unittest.main()
