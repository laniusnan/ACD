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

if __name__=='__main__':unittest.main()
