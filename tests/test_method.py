import unittest
import torch
from acd.heads import CosineHead
from acd.losses import standardize, full_kl, conditional_kl, distillation_loss
from acd.models import build, logits, REGISTRY

class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(12)
    def test_angular_invariance_and_etf(self):
        head=CosineHead(torch.randn(5,8),etf=True)
        gram=torch.nn.functional.normalize(head.weight,dim=1)@torch.nn.functional.normalize(head.weight,dim=1).T
        self.assertTrue(torch.allclose(gram,torch.eye(5)*1.25-.25,atol=1e-5))
        x=torch.randn(3,8);before=head(x)
        with torch.no_grad():head.weight.mul_(torch.arange(1,6)[:,None])
        self.assertTrue(torch.allclose(before,head(x*3),atol=1e-5))
    def test_exact_kl_decomposition(self):
        z=torch.randn(4,6,dtype=torch.double);t=torch.randn_like(z);y=torch.tensor([0,1,2,3]);tau=4.
        p=(t/tau).softmax(1).gather(1,y[:,None]);q=(z/tau).softmax(1).gather(1,y[:,None])
        binary=tau**2*torch.nn.functional.kl_div(torch.cat([q,1-q],1).log(),torch.cat([p,1-p],1),reduction='batchmean')
        self.assertTrue(torch.allclose(full_kl(z,t,tau),binary+conditional_kl(z,t,y,tau),atol=1e-12))
    def test_gradient_matching_and_teacher_detach(self):
        layer=torch.nn.Linear(7,9);head=CosineHead(torch.randn(6,9));x=torch.randn(4,7);z=head(layer(x));t=torch.randn(4,6,requires_grad=True);y=torch.tensor([0,1,2,3])
        loss,d=distillation_loss(z,t,y,list(layer.parameters()),{'method':'acd'})
        self.assertFalse(d['r'].requires_grad);loss.backward()
        self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(layer.weight.grad).all())
    def test_all_cifar_models(self):
        for name in REGISTRY['models']:
            with self.subTest(name=name):
                net,_=build(name,angular=True);net.eval()
                with torch.no_grad(): self.assertEqual(tuple(logits(net,torch.randn(2,3,32,32)).shape),(2,100))

    def test_etf_preserves_dtype_at_minimum_dimension(self):
        head = CosineHead(torch.randn(5, 4, dtype=torch.float64), etf=True)
        expected = torch.eye(5, dtype=torch.float64) * 1.25 - .25
        self.assertEqual(head.weight.dtype, torch.float64)
        self.assertEqual(head.scale.dtype, torch.float64)
        self.assertTrue(torch.allclose(head.weight @ head.weight.T, expected, atol=1e-12))
        head(torch.randn(3, 4, dtype=torch.float64)).square().mean().backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_etf_rejects_impossible_or_affine_configuration(self):
        with self.assertRaisesRegex(ValueError, 'feature dimension'):
            CosineHead(torch.randn(100, 64), etf=True)
        with self.assertRaisesRegex(ValueError, 'angular'):
            build('resnet20', angular=False, etf=True)

    def test_nonfinite_cosine_scale_is_rejected(self):
        for scale in (0., -1., float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                CosineHead(torch.randn(4, 6), scale=scale)

    def test_standardized_relation_invariant_to_positive_affine_scores(self):
        student = torch.randn(4, 6, dtype=torch.float64)
        teacher = torch.randn_like(student)
        labels = torch.tensor([0, 1, 2, 3])
        scales = torch.tensor([.2, 2., 4., 7.], dtype=torch.float64)[:, None]
        shifts = torch.tensor([4., -3., 1., 9.], dtype=torch.float64)[:, None]
        original = conditional_kl(standardize(student), standardize(teacher), labels, 4.)
        transformed = conditional_kl(standardize(student * scales + shifts),
                                     standardize(teacher / scales - shifts), labels, 4.)
        self.assertTrue(torch.allclose(original, transformed, atol=1e-12))

    def test_matching_uses_standardized_backbone_reference(self):
        layer = torch.nn.Linear(7, 9)
        head = CosineHead(torch.randn(6, 9))
        student = head(layer(torch.randn(4, 7)))
        teacher = torch.randn(4, 6, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        loss, diagnostics = distillation_loss(student, teacher, labels, layer.parameters(),
                                               {'method': 'acd', 'ce_weight': 0.})
        reference = full_kl(standardize(student), standardize(teacher), 4.)
        actual_gradients = torch.autograd.grad(loss, tuple(layer.parameters()), retain_graph=True)
        reference_gradients = torch.autograd.grad(reference, tuple(layer.parameters()), retain_graph=True)
        actual_norm = torch.cat([gradient.flatten() for gradient in actual_gradients]).norm()
        reference_norm = torch.cat([gradient.flatten() for gradient in reference_gradients]).norm()
        self.assertTrue(torch.allclose(actual_norm, reference_norm, rtol=2e-5, atol=1e-7))
        self.assertFalse(diagnostics['r'].requires_grad)
        loss.backward()
        self.assertIsNone(teacher.grad)

    def test_standardization_retains_target_dependence(self):
        student = torch.tensor([[2., 1., -.5, -1.]], requires_grad=True)
        teacher = torch.tensor([[-1., .2, 3., -2.]])
        labels = torch.tensor([0])
        relation = conditional_kl(standardize(student), standardize(teacher), labels, 4.)
        gradient = torch.autograd.grad(relation, student)[0]
        self.assertGreater(abs(gradient[0, 0].item()), 1e-5)

if __name__=='__main__':unittest.main()
