#!/usr/bin/env python3
"""Read-only multi-pair counterfactual probe for ACD mechanism figures.

For each completed affine-head linear-KD checkpoint, hold the input, teacher
target, and normalized student feature/classifier directions fixed.  Compare
the angular score, the score after scale/norm injection, and the raw affine
score.  This creates no training job, checkpoint, or queue mutation.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision import datasets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "reproductions/linear_kd_baseline_20261028"
OUT = HERE / "mechanism_figures"
OUT.mkdir(exist_ok=True)
PAIRS = ("vgg13_mv1", "res50_mv1", "res32x4_sv1", "wrn402_sv1", "res32x4_sv2")
LABELS = {
    "vgg13_mv1": "VGG13→MV1",
    "res50_mv1": "R50→MV1",
    "res32x4_sv1": "R32×4→SV1",
    "wrn402_sv1": "WRN40-2→SV1",
    "res32x4_sv2": "R32×4→SV2",
}
TEMPERATURE = 4.0
PROBE_SIZE = 128


def digest(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def conditional(logits, labels):
    probs = (logits / TEMPERATURE).softmax(1)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask.scatter_(1, labels[:, None], False)
    return probs[mask].reshape(len(labels), -1) / (1 - probs.gather(1, labels[:, None])).clamp_min(1e-12)


def kd_gradient(logits, target, feature):
    loss = TEMPERATURE ** 2 * F.kl_div(
        (logits / TEMPERATURE).log_softmax(1),
        (target.detach() / TEMPERATURE).softmax(1), reduction="batchmean")
    return torch.autograd.grad(loss, feature)[0]


def scores(feature, weight, bias):
    angular = F.normalize(feature, dim=1) @ F.normalize(weight, dim=1).T
    scale_norm = feature.norm(dim=1, keepdim=True) * weight.norm(dim=1)[None, :] * angular
    return angular, scale_norm, scale_norm + bias[None, :]


def capture_forward(model, head, inputs, need_feature):
    holder = {}
    hook = head.register_forward_pre_hook(lambda _m, args: holder.setdefault("feature", args[0]))
    try:
        output = model(inputs)
    finally:
        hook.remove()
    logits = output[0] if isinstance(output, tuple) else output
    if not need_feature:
        return logits.detach()
    return logits, holder["feature"]


def main():
    sys.path.insert(0, str(BASE))
    pilot = module("acd_multimodel_baseline_pilot", BASE / "pilot.py")
    methods = pilot.base
    config = pilot.configuration()
    data = datasets.CIFAR100(ROOT / "data", train=False, download=False, transform=methods.clean_transform())
    inputs = torch.stack([data[index][0] for index in range(PROBE_SIZE)])
    labels = torch.tensor([data[index][1] for index in range(PROBE_SIZE)])
    results, sources = {}, {}
    for pair in PAIRS:
        spec = config["pairs"][pair]
        student_name, teacher_name = spec["student"], spec["teacher"]
        student = methods.model_for(student_name, config).eval()
        teacher = methods.model_for(teacher_name, config).eval()
        head_name = config["models"][student_name]["head"]
        teacher_head_name = config["models"][teacher_name]["head"]
        head, teacher_head = getattr(student, head_name), getattr(teacher, teacher_head_name)
        if not isinstance(head, torch.nn.Linear) or not isinstance(teacher_head, torch.nn.Linear):
            raise RuntimeError(f"{pair} does not expose affine linear heads")
        checkpoint = BASE / "artifacts" / f"{pair}__s2061__skd_recipe_linear" / "last.pth"
        teacher_path = ROOT / "reproductions/head_transfer_20261021/frozen/teachers" / f"best_model_weights_{teacher_name}.pth"
        student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["student"], strict=True)
        teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            teacher_logits = capture_forward(teacher, teacher_head, inputs, need_feature=False)
        logits, feature0 = capture_forward(student, head, inputs, need_feature=True)
        torch.testing.assert_close(logits.detach(), F.linear(feature0.detach(), head.weight.detach(), head.bias.detach()))
        feature = feature0.detach().clone().requires_grad_(True)
        angular, scale_norm, raw = scores(feature, head.weight.detach(), head.bias.detach())
        g_angular = kd_gradient(angular, teacher_logits, feature).detach()
        feature_scale = feature0.detach().clone().requires_grad_(True)
        _, scale_norm, _ = scores(feature_scale, head.weight.detach(), head.bias.detach())
        g_scale = kd_gradient(scale_norm, teacher_logits, feature_scale).detach()
        feature_raw = feature0.detach().clone().requires_grad_(True)
        _, _, raw = scores(feature_raw, head.weight.detach(), head.bias.detach())
        g_raw = kd_gradient(raw, teacher_logits, feature_raw).detach()
        with torch.no_grad():
            angular0, scale0, raw0 = scores(feature0.detach(), head.weight.detach(), head.bias.detach())
            tv_scale = .5 * (conditional(angular0, labels) - conditional(scale0, labels)).abs().sum(1)
            tv_raw = .5 * (conditional(angular0, labels) - conditional(raw0, labels)).abs().sum(1)
            delta_direction = 1 - F.cosine_similarity(g_angular, g_raw, dim=1)
            # A pair-local basis: angular force plus raw's orthogonal residual.
            force_a, force_s, force_r = (-g_angular.mean(0), -g_scale.mean(0), -g_raw.mean(0))
            e1 = force_a / force_a.norm().clamp_min(1e-12)
            residual = force_r - (force_r @ e1) * e1
            e2 = residual / residual.norm().clamp_min(1e-12)
            base = force_a.norm().clamp_min(1e-12)
            trajectory = {
                "angular": [float((force_a @ e1) / base), float((force_a @ e2) / base)],
                "scale_norm": [float((force_s @ e1) / base), float((force_s @ e2) / base)],
                "raw": [float((force_r @ e1) / base), float((force_r @ e2) / base)],
            }
        results[pair] = {
            "label": LABELS[pair],
            "distribution_tv_angular_to_scale_norm": tv_scale.tolist(),
            "distribution_tv_angular_to_raw": tv_raw.tolist(),
            "gradient_direction_deviation_angular_to_raw": delta_direction.tolist(),
            "summary": {
                "mean_tv_angular_to_scale_norm": float(tv_scale.mean()),
                "mean_tv_angular_to_raw": float(tv_raw.mean()),
                "mean_gradient_direction_deviation": float(delta_direction.mean()),
            },
            "trajectory": trajectory,
        }
        sources[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
        sources[str(teacher_path.relative_to(ROOT))] = digest(teacher_path)
    payload = {
        "status": "completed",
        "scope": "Read-only multi-pair counterfactual probe; no training, endpoint evaluation, or queue mutation.",
        "setup": {"pairs": list(PAIRS), "labels": LABELS, "seed": 2061, "epoch": 240,
                  "probe": f"first {PROBE_SIZE} official CIFAR-100 test images", "temperature": TEMPERATURE,
                  "counterfactual": "Per sample, hold image, teacher raw target, normalized student feature direction, and normalized classifier directions fixed; change only student score parameterization."},
        "results": results,
        "sources_sha256": sources,
        "limitations": [
            "The five source-recipe pilot checkpoints share one exploratory seed and are not an independent multi-seed performance comparison.",
            "A pair-local force basis makes trajectories visually comparable but does not put different architectures in one literal feature space.",
            "This measures affine-head parameterization sensitivity, not end-to-end ACI/SICD accuracy or a complete optimization trajectory.",
        ],
    }
    path = OUT / "multimodel_parameterization_probe.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
