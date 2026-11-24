"""New CIFAR ACD protocol: explicit angular, standardized and conditional losses."""
import hashlib
import importlib.util
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent


def module(path):
    spec = importlib.util.spec_from_file_location("cifar_rerun_" + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def model(name, config):
    entry = config["models"][name]
    result = getattr(module(HERE / "frozen/models" / (entry["module"] + ".py")), entry["constructor"])(num_classes=100)
    return result[0] if isinstance(result, tuple) else result


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class AngularHead(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(weight.detach().clone(), dim=1))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=1))


def student(pair, seed, arm, policy, config):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        result = model(config["pairs"][pair]["student"], config)
    head_name = config["models"][config["pairs"][pair]["student"]]["head"]
    head = getattr(result, head_name)
    initial = {"backbone": tensor_hash({n: v for n, v in result.state_dict().items() if not n.startswith(head_name + ".")}),
               "linear_head": tensor_hash(head.state_dict()), "feature_dimension": head.in_features}
    if arm != "linear_kd":
        setattr(result, head_name, AngularHead(head.weight, policy["angular_scale"]))
    backbone = [p for n, p in result.named_parameters() if p.requires_grad and not n.startswith(head_name + ".")]
    return result, backbone, initial


def standardize(z, epsilon):
    return (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True, correction=1).clamp_min(epsilon)


def kl(z, t, temperature):
    return F.kl_div((z / temperature).log_softmax(1), (t.detach() / temperature).softmax(1),
                    reduction="batchmean") * temperature**2


def relation(z, t, labels, temperature):
    t = t.detach()
    mask = torch.ones_like(z, dtype=torch.bool).scatter_(1, labels[:, None], False)
    divergence = F.kl_div((z[mask].reshape(len(labels), -1) / temperature).log_softmax(1),
                          (t[mask].reshape(len(labels), -1) / temperature).softmax(1), reduction="none").sum(1)
    mass = (t / temperature).softmax(1).masked_fill(~mask, 0).sum(1)
    return (mass * divergence).mean() * temperature**2


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    return torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt()


def objective(z, t, labels, backbone, arm, policy, recipe):
    t = t.detach()
    ce = F.cross_entropy(z, labels)
    diagnostics = {}
    if arm in ("linear_kd", "angular_kd"):
        kd = kl(z, t, policy["temperature"])
    else:
        zs, ts = standardize(z, policy["standardization"]["epsilon"]), standardize(t, policy["standardization"]["epsilon"])
        full = kl(zs, ts, policy["temperature"])
        if arm == "angular_full":
            kd = full
        elif arm == "acd":
            conditional = relation(zs, ts, labels, policy["temperature"])
            reference_norm, conditional_norm = gradient_norm(full, backbone), gradient_norm(conditional, backbone)
            ratio = (reference_norm / conditional_norm.clamp_min(policy["gradient_matching"]["epsilon"])).detach()
            kd = ratio * conditional
            diagnostics = {"ratio": ratio, "reference_gradient_norm": reference_norm,
                           "relation_gradient_norm": conditional_norm,
                           "denominator_clamped": (conditional_norm < policy["gradient_matching"]["epsilon"]).float()}
        else:
            raise ValueError(arm)
        diagnostics.update(student_std_clamped_fraction=(z.detach().std(1, correction=1) < policy["standardization"]["epsilon"]).float().mean(),
                           teacher_std_clamped_fraction=(t.std(1, correction=1) < policy["standardization"]["epsilon"]).float().mean())
    diagnostics.update(ce=ce.detach(), kd=kd.detach())
    return recipe["ce_weight"] * ce + recipe["kd_weight"] * kd, diagnostics


def clean():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))])


class DoubleViews(Dataset):
    def __init__(self, root):
        self.base = datasets.CIFAR100(root, train=True, download=False)
        augmentation = module(HERE / "frozen/augmentation.py")
        self.weak = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), clean()])
        self.strong = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          augmentation.RandAugment(2, 10), clean()])

    def __len__(self):
        return 2 * len(self.base)

    def __getitem__(self, index):
        identity = index % len(self.base)
        transform = self.strong if index >= len(self.base) else self.weak
        return transform(Image.fromarray(self.base.data[identity])), self.base.targets[identity], index


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def train_loader(dataset, policy, seed, epoch):
    generator = torch.Generator().manual_seed(seed * 1000 + epoch)
    return DataLoader(dataset, batch_size=policy["batch_size"], shuffle=True, num_workers=policy["workers"],
                      generator=generator, worker_init_fn=seed_worker, pin_memory=True)


def test_loader():
    dataset = datasets.CIFAR100(HERE / "data", train=False, transform=clean(), download=False)
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
