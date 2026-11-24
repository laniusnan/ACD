"""ImageNet heads and equation-level SICD reconstruction, not sealed-source parity."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision import models


ARMS = {
    "I0": ("linear", "ce"),
    "I1": ("cosine", "ce"),
    "I2": ("linear", "kd"),
    "I3": ("cosine", "kd"),
    "I4": ("linear", "standardized_full"),
    "I5": ("cosine", "standardized_full"),
    "I6": ("linear", "sicd"),
    "I7": ("cosine", "sicd"),
}
PAIRS = {
    "resnet50_mobilenet_v2": (models.resnet50, models.mobilenet_v2,
                              models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet34_resnet18": (models.resnet34, models.resnet18,
                         models.ResNet34_Weights.IMAGENET1K_V1),
}
SICD_REFERENCE_VERIFIED = False


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class CosineHead(nn.Module):
    def __init__(self, initial_weight, scale):
        super().__init__()
        self.weight = nn.Parameter(functional.normalize(initial_weight.detach().clone(), dim=1))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, features):
        return self.scale * functional.linear(functional.normalize(features, dim=-1),
                                               functional.normalize(self.weight, dim=1))


def build_student(pair, head, scale, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = PAIRS[pair][1](weights=None)
    head_name = "classifier.1" if pair == "resnet50_mobilenet_v2" else "fc"
    classifier = student.get_submodule(head_name)
    initial = {
        "backbone_sha256": tensor_digest({name: value for name, value in student.state_dict().items()
                                          if not name.startswith(head_name + ".")}),
        "linear_head_sha256": tensor_digest(classifier.state_dict()),
        "feature_dimension": classifier.in_features,
    }
    if head == "cosine":
        replacement = CosineHead(classifier.weight, scale)
        if head_name == "fc":
            student.fc = replacement
        else:
            student.classifier[1] = replacement
    elif head != "linear":
        raise ValueError(head)
    backbone = [parameter for name, parameter in student.named_parameters()
                if not name.startswith(head_name + ".") and parameter.requires_grad]
    return student, backbone, initial


def standardize(logits, epsilon=1e-6):
    values = logits.double() if logits.dtype == torch.float64 else logits.float()
    return (values - values.mean(1, keepdim=True)) / values.std(
        1, keepdim=True, correction=1).clamp_min(epsilon)


def full_kl(student, teacher, temperature):
    return functional.kl_div((student / temperature).log_softmax(1),
                             (teacher.detach() / temperature).softmax(1),
                             reduction="batchmean") * temperature ** 2


def conditional_kl(student, teacher, labels, temperature, weighted=True):
    teacher = teacher.detach()
    mask = torch.ones_like(student, dtype=torch.bool).scatter_(1, labels[:, None], False)
    student_conditional = student[mask].reshape(len(labels), -1) / temperature
    teacher_conditional = teacher[mask].reshape(len(labels), -1) / temperature
    divergence = functional.kl_div(student_conditional.log_softmax(1),
                                   teacher_conditional.softmax(1), reduction="none").sum(1)
    weight = (teacher / temperature).softmax(1).masked_fill(~mask, 0).sum(1)
    return (divergence * weight if weighted else divergence).mean() * temperature ** 2


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=False,
                                    allow_unused=False)
    return torch.stack([gradient.detach().float().square().sum()
                        for gradient in gradients]).sum().sqrt()


def matched_relation(student, teacher, labels, parameters, temperature, epsilon, gradient_epsilon):
    student_standardized = standardize(student, epsilon)
    teacher_standardized = standardize(teacher.detach(), epsilon)
    reference = full_kl(student_standardized, teacher_standardized, temperature)
    relation = conditional_kl(student_standardized, teacher_standardized, labels, temperature)
    reference_norm = gradient_norm(reference, parameters)
    relation_norm = gradient_norm(relation, parameters)
    ratio = (reference_norm / relation_norm.clamp_min(gradient_epsilon)).detach()
    if not torch.isfinite(ratio):
        raise FloatingPointError("Non-finite SICD gradient ratio")
    return ratio * relation, {
        "match_ratio": ratio,
        "reference_gradient_norm": reference_norm,
        "relation_gradient_norm": relation_norm,
        "gradient_denominator_clamped": (relation_norm < gradient_epsilon).float(),
    }


def objective(logits, teacher_logits, labels, backbone, arm, config):
    logits = logits.float()
    teacher_logits = teacher_logits.detach().float()
    cross_entropy = functional.cross_entropy(logits, labels)
    distillation = logits.new_zeros(())
    diagnostics = {}
    loss_name = ARMS[arm][1]
    if loss_name == "kd":
        distillation = full_kl(logits, teacher_logits, config["temperature"])
    elif loss_name == "standardized_full":
        distillation = full_kl(standardize(logits, config["standardization_epsilon"]),
                              standardize(teacher_logits, config["standardization_epsilon"]),
                              config["temperature"])
    elif loss_name == "sicd":
        distillation, diagnostics = matched_relation(
            logits, teacher_logits, labels, backbone, config["temperature"],
            config["standardization_epsilon"], config["gradient_epsilon"])
    if loss_name in ("standardized_full", "sicd"):
        diagnostics["student_std_clamped_fraction"] = (
            logits.detach().std(1, correction=1) < config["standardization_epsilon"]).float().mean()
        diagnostics["teacher_std_clamped_fraction"] = (
            teacher_logits.std(1, correction=1) < config["standardization_epsilon"]).float().mean()
    diagnostics.update(ce=cross_entropy.detach(), kd=distillation.detach())
    total = config["ce_weight"] * cross_entropy + config["kd_weight"] * distillation
    return total, diagnostics
