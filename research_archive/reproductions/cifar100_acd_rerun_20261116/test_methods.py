import unittest
import torch
from torch.nn import functional as F
import methods


torch.set_num_threads(2)


class ScientificTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_finite_temperature_conditional_decomposition(self):
        z = torch.randn(7, 100, dtype=torch.float64, requires_grad=True)
        t = torch.randn_like(z)
        y = torch.arange(7)
        p, q = (t / 4).softmax(1), (z / 4).softmax(1)
        pt, qt = p.gather(1, y[:, None]), q.gather(1, y[:, None])
        binary = F.kl_div(torch.cat((qt, 1-qt), 1).log(), torch.cat((pt, 1-pt), 1), reduction="batchmean") * 16
        full = methods.kl(z, t, 4)
        decomposed = binary + methods.relation(z, t, y, 4)
        torch.testing.assert_close(full, decomposed, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(torch.autograd.grad(full, z, retain_graph=True)[0],
                                   torch.autograd.grad(decomposed, z)[0], atol=1e-12, rtol=1e-10)

    def test_affine_invariance_and_constant_logits(self):
        z = torch.randn(4, 100, dtype=torch.float64)
        scale = torch.rand(4, 1, dtype=torch.float64) + .1
        shift = torch.randn(4, 1, dtype=torch.float64)
        torch.testing.assert_close(methods.standardize(z, 1e-6), methods.standardize(scale*z+shift, 1e-6))
        constant = torch.ones(3, 100, requires_grad=True)
        methods.standardize(constant, 1e-6).square().sum().backward()
        self.assertTrue(torch.isfinite(constant.grad).all())

    def test_matching_uses_backbone_and_keeps_teacher_frozen(self):
        h = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        w = torch.randn(100, 8, dtype=torch.float64, requires_grad=True)
        t = torch.randn(4, 100, dtype=torch.float64, requires_grad=True)
        z = F.linear(h, w)
        policy = {"temperature": 4, "standardization": {"epsilon": 1e-6}, "gradient_matching": {"epsilon": 1e-12}}
        loss, diagnostics = methods.objective(z, t, torch.arange(4), [h], "acd", policy, {"ce_weight": 0., "kd_weight": 1.})
        self.assertFalse(diagnostics["ratio"].requires_grad)
        gradient = torch.autograd.grad(loss, h, retain_graph=True)[0]
        torch.testing.assert_close(gradient.norm(), diagnostics["reference_gradient_norm"])
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(w.grad)

    def test_angular_head_is_invariant_to_positive_feature_and_weight_scales(self):
        weight = torch.randn(100, 64)
        h = torch.randn(5, 64)
        head = methods.AngularHead(weight, 30.)
        torch.testing.assert_close(head(h), head(3*h), atol=2e-6, rtol=1e-5)
        initial = head(h).detach()
        with torch.no_grad(): head.weight.mul_(torch.rand(100, 1) + 1)
        torch.testing.assert_close(initial, head(h), atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
