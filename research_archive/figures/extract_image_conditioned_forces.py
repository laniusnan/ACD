#!/usr/bin/env python3
"""CPU-only, read-only image-conditioned KD-force probe.

Three examples are selected by fixed quartiles of the raw-versus-angular
feature-force deviation on the first 128 CIFAR-100 test images.  This is a
visualization probe, not a new training or endpoint-evaluation procedure.
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
OUT = HERE / "mechanism_figures"
STUDY = ROOT / "reproductions/acd_cnn_interaction_20261109"
JOB = "vgg13_mv1__s2102__linear_kd"
CHECKPOINT = STUDY / "artifacts" / JOB / "epoch_240.pth"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kd_gradient(logits, target, feature):
    loss = 16 * F.kl_div(
        (logits / 4).log_softmax(1), (target.detach() / 4).softmax(1), reduction="batchmean"
    )
    return torch.autograd.grad(loss, feature)[0]


def score_stages(feature, weight, bias):
    directional = F.normalize(feature, dim=1) @ F.normalize(weight, dim=1).T
    scale = feature.norm(dim=1, keepdim=True) * weight.norm(dim=1)[None, :] * directional
    return {"angular": directional, "scale_norm": scale, "raw": scale + bias[None, :]}


def local_force_coordinates(forces):
    """Return unit force directions in a per-image angular/raw plane."""
    angular, raw = forces["angular"], forces["raw"]
    e1 = angular / angular.norm().clamp_min(1e-12)
    residual = raw - (raw @ e1) * e1
    e2 = residual / residual.norm().clamp_min(1e-12)
    result = {}
    for name, gradient in forces.items():
        # Negative gradient is the instantaneous SGD force.
        force = -gradient
        x, y = float(force @ e1), float(force @ e2)
        norm = (x * x + y * y) ** 0.5
        result[name] = {"x": x / norm, "y": y / norm}
    return result


def main():
    study = module("image_force_probe_study", STUDY / "study.py")
    job = study.jobs()[JOB]
    study.core.seed_all(2102)
    student, teacher, _, _ = study.build(job)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    teacher.eval()

    transformed = datasets.CIFAR100(ROOT / "data", train=False, download=False, transform=study.base.clean_transform())
    raw_dataset = datasets.CIFAR100(ROOT / "data", train=False, download=False)
    x = torch.stack([transformed[i][0] for i in range(128)])
    with torch.no_grad():
        teacher_logits, _ = teacher(x)
        _, features = student(x)
        pooled = features["pooled_feat"]

    weight, bias = student.fc.weight.detach(), student.fc.bias.detach()
    scores = score_stages(pooled.detach().clone().requires_grad_(True), weight, bias)
    gradients = {}
    for name in ("angular", "scale_norm", "raw"):
        feature = pooled.detach().clone().requires_grad_(True)
        logits = score_stages(feature, weight, bias)[name]
        gradients[name] = kd_gradient(logits, teacher_logits, feature).detach()

    angular, raw = gradients["angular"], gradients["raw"]
    deviations = 1 - F.cosine_similarity(angular, raw, dim=1)
    selected = []
    used = set()
    for quantile in (0.25, 0.50, 0.75):
        target = torch.quantile(deviations, quantile)
        order = (deviations - target).abs().argsort().tolist()
        idx = next(candidate for candidate in order if candidate not in used)
        used.add(idx)
        forces = {name: gradients[name][idx] for name in gradients}
        selected.append({
            "quantile": quantile,
            "index": idx,
            "label": int(raw_dataset.targets[idx]),
            "class_name": raw_dataset.classes[raw_dataset.targets[idx]],
            "raw_vs_angular_deviation": float(deviations[idx]),
            "unit_force_coordinates": local_force_coordinates(forces),
            "rgb": raw_dataset.data[idx].tolist(),
        })

    record = {
        "status": "completed",
        "scope": "Read-only visualization probe; no training, endpoint evaluation, or queue mutation.",
        "setup": {
            "pair": "VGG13→MobileNetV1", "student_arm": "linear KD", "seed": 2102,
            "checkpoint_epoch": 240, "probe": "first 128 official CIFAR-100 test images",
            "selection_rule": "nearest distinct 25th, 50th, and 75th percentiles of per-image 1-cos(angular-force, raw-force)",
            "force": "negative KD feature gradient; each force is projected to its own angular/raw plane and unit-normalized",
        },
        "examples": selected,
        "sources_sha256": {
            str(CHECKPOINT.relative_to(ROOT)): digest(CHECKPOINT),
            str((STUDY / "study.py").relative_to(ROOT)): digest(STUDY / "study.py"),
        },
        "limitations": [
            "The three examples are fixed quantiles from one pair/seed, not independent seeds or accuracy-selected examples.",
            "Unit normalization intentionally removes force magnitude; the panel visualizes directions only.",
            "Every example has its own local force plane, so arrows are comparable within a card but do not embed images in a common feature space.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "image_conditioned_force_probe.json").write_text(json.dumps(record, indent=2) + "\n")
    print(OUT / "image_conditioned_force_probe.json")


if __name__ == "__main__":
    main()
