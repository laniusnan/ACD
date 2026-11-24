#!/usr/bin/env python3
"""Read-only input-force maps for one pre-defined median CIFAR-100 example."""
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
CHECKPOINT = STUDY / "artifacts" / "vgg13_mv1__s2102__linear_kd" / "epoch_240.pth"


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


def scores(feature, weight, bias):
    angular = F.normalize(feature, dim=1) @ F.normalize(weight, dim=1).T
    scaled = feature.norm(dim=1, keepdim=True) * weight.norm(dim=1)[None, :] * angular
    return {"angular": angular, "scale_norm": scaled, "raw": scaled + bias[None, :]}


def main():
    study = module("same_input_force_probe_study", STUDY / "study.py")
    job = study.jobs()["vgg13_mv1__s2102__linear_kd"]
    study.core.seed_all(2102)
    student, teacher, _, _ = study.build(job)
    student.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["student"], strict=True)
    student.eval(); teacher.eval()

    # Indices 66 and 31 are the fixed q25 and q75 samples in
    # image_conditioned_force_probe.json. They give a pre-defined low/high
    # disagreement comparison without selecting examples by appearance.
    indices = (66, 31)
    transformed = datasets.CIFAR100(ROOT / "data", train=False, download=False, transform=study.base.clean_transform())
    raw_set = datasets.CIFAR100(ROOT / "data", train=False, download=False)
    examples = []
    for index in indices:
        x0 = transformed[index][0].unsqueeze(0)
        with torch.no_grad():
            target, _ = teacher(x0)
        force_maps = {}
        for name in ("angular", "scale_norm", "raw"):
            x = x0.detach().clone().requires_grad_(True)
            _, features = student(x)
            logits = scores(features["pooled_feat"], student.fc.weight, student.fc.bias)[name]
            loss = 16 * F.kl_div((logits / 4).log_softmax(1), (target / 4).softmax(1), reduction="batchmean")
            # Negative input gradient is the instantaneous first-order SGD force.
            force_maps[name] = (-torch.autograd.grad(loss, x)[0][0]).detach().tolist()
        examples.append({"index": index, "class_name": raw_set.classes[raw_set.targets[index]], "rgb": raw_set.data[index].tolist(), "force_maps": force_maps})

    result = {
        "status": "completed",
        "scope": "Read-only local pixel-force visualization probe; no training, endpoint evaluation, or queue mutation.",
        "setup": {
            "pair": "VGG13→MobileNetV1", "student_arm": "linear KD", "seed": 2102, "epoch": 240,
            "selection_rule": "the pre-defined q25 and q75 examples from image_conditioned_force_probe.json; they are nearest low/high quartiles of raw-versus-angular feature-force deviation among first 128 official CIFAR-100 test images",
            "force": "negative KD loss gradient with respect to the normalized input pixels",
        },
        "examples": examples,
        "sources_sha256": {str(CHECKPOINT.relative_to(ROOT)): digest(CHECKPOINT), str((STUDY / "study.py").relative_to(ROOT)): digest(STUDY / "study.py")},
        "limitations": [
            "These are local pixel-input gradients for one image, not a full training trajectory or feature-force plot.",
            "Unit normalization removes each method's absolute gradient magnitude; color depth therefore represents relative spatial concentration or signed projection, depending on the draft.",
            "Color depth alone cannot encode a high-dimensional vector direction; drafts that use it show a clearly stated scalar projection or local intensity.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "same_input_pixel_force_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(OUT / "same_input_pixel_force_probe.json")


if __name__ == "__main__":
    main()
