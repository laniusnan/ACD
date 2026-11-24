#!/usr/bin/env python3
"""Read-only, fixed-probe extraction for the ACD teaser figure.

This script never trains or changes a checkpoint.  It evaluates the first 256
examples of CIFAR-100's official test ordering and records classifier-output
diagnostics for already completed ViT-T -> ResNet8x4 checkpoints.
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = ROOT / "reproductions/cvpr_angular_interface_interaction_20261109"
PARENT = ROOT / "reproductions/cvpr_vit_rkd_three_seed_extension_recovery_20261109"
OUT = Path(__file__).resolve().parent / "artifacts"
N_PROBE = 256
TEMPERATURE = 4.0


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


study = module("acd_teaser_study", STUDY_DIR / "study.py")


def checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["student"]


def build_student(arm, path):
    study.core.seed_all(2102)
    student, teacher, head_name, _ = study.build({"arm": arm, "seed": 2102})
    student.load_state_dict(checkpoint(path), strict=True)
    student.eval()
    return student, teacher.eval(), head_name


def linear_head(model):
    candidates = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 100:
            candidates.append((name, layer))
    if not candidates:
        raise RuntimeError("No 100-way linear classifier found")
    return candidates[-1]


def angular_and_raw(head, feature):
    weight = head.weight.detach()
    angular = F.normalize(feature, dim=1) @ F.normalize(weight, dim=1).T
    radial = feature @ weight.T
    bias = torch.zeros_like(radial) if head.bias is None else head.bias.detach()[None, :]
    return angular, radial, radial + bias


def non_target_topk(scores, labels, k=5):
    value = scores.clone()
    value[torch.arange(len(labels)), labels] = -torch.inf
    return value.topk(k, dim=1).indices


def ordering_metrics(first, second, labels):
    a1 = non_target_topk(first, labels, 1)[:, 0]
    b1 = non_target_topk(second, labels, 1)[:, 0]
    a5 = non_target_topk(first, labels, 5)
    b5 = non_target_topk(second, labels, 5)
    overlap = torch.stack([
        torch.isin(a5[i], b5[i]).float().mean() for i in range(len(labels))
    ]).mean()
    return {
        "top1_changed_pct": float(100 * a1.ne(b1).float().mean()),
        "mean_top5_overlap_pct": float(100 * overlap),
    }


def normalize_logits(z):
    return (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True, correction=1).clamp_min(1e-6)


def kl_decomposition(student, teacher, labels):
    p = (normalize_logits(teacher) / TEMPERATURE).softmax(1)
    q = (normalize_logits(student) / TEMPERATURE).softmax(1)
    idx = torch.arange(len(labels))
    py, qy = p[idx, labels], q[idx, labels]
    binary = py * (py.log() - qy.log()) + (1 - py) * ((1 - py).log() - (1 - qy).log())
    mask = F.one_hot(labels, num_classes=p.shape[1]).bool()
    pminus = p.masked_select(~mask).reshape(len(labels), -1) / (1 - py[:, None])
    qminus = q.masked_select(~mask).reshape(len(labels), -1) / (1 - qy[:, None])
    relation = (1 - py) * (pminus * (pminus.log() - qminus.log())).sum(1)
    full = binary + relation
    return binary, relation, full


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    paths = {
        "linear_kd": PARENT / "artifacts/vit_res8x4__s2102__linear_kd/epoch_240.pth",
        "cosine_kd": PARENT / "artifacts/vit_res8x4__s2102__cosine_kd/epoch_240.pth",
        "cosine_standardized_full": STUDY_DIR / "artifacts/vit_res8x4__s2102__cosine_standardized_full/epoch_240.pth",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    linear, teacher, linear_name = build_student("linear_kd", paths["linear_kd"])
    cosine, _, cosine_name = build_student("cosine_kd", paths["cosine_kd"])
    standardized, _, standardized_name = build_student("cosine_standardized_full", paths["cosine_standardized_full"])
    teacher_name, teacher_head = linear_head(teacher)
    linear_head_layer = getattr(linear, linear_name)

    split = study.load(STUDY_DIR / "artifacts/split.json")
    # The registered evaluation loader uses worker processes.  This read-only
    # audit intentionally reads its fixed indices from the same dataset without
    # workers, so it is portable to sandboxed analysis environments.
    dataset = study.recipe.evaluation_loader("confirmation", split).dataset
    records = {"teacher": [], "linear": [], "cosine": [], "standardized": [], "labels": []}
    count = 0
    with torch.no_grad():
        for start in range(0, N_PROBE, 32):
            items = [dataset[index] for index in range(start, min(start + 32, N_PROBE))]
            image = torch.stack([item[0] for item in items])
            label = torch.tensor([item[1] for item in items], dtype=torch.long)
            take = len(label)
            tlogit, tfeat = teacher(image)
            llogit, lfeat = linear(image)
            clogit, cfeat = cosine(image)
            slogit, sfeat = standardized(image)
            records["teacher"].append((tlogit, tfeat["pooled_feat"]))
            records["linear"].append((llogit, lfeat["pooled_feat"]))
            records["cosine"].append((clogit, cfeat["pooled_feat"]))
            records["standardized"].append((slogit, sfeat["pooled_feat"]))
            records["labels"].append(label)
            count += take

    labels = torch.cat(records.pop("labels"))
    values = {key: (torch.cat([x[0] for x in value]), torch.cat([x[1] for x in value])) for key, value in records.items()}
    tangle, tradial, traw = angular_and_raw(teacher_head, values["teacher"][1])
    langle, lradial, lraw = angular_and_raw(linear_head_layer, values["linear"][1])

    ordering = {
        "teacher_magnitude": ordering_metrics(tangle, tradial, labels),
        "teacher_bias": ordering_metrics(tradial, traw, labels),
        "linear_student_magnitude": ordering_metrics(langle, lradial, labels),
        "linear_student_bias": ordering_metrics(lradial, lraw, labels),
    }
    binary, relation, full = kl_decomposition(values["standardized"][0], values["teacher"][0], labels)
    decomposition = {
        "mean_full": float(full.mean()),
        "mean_binary": float(binary.mean()),
        "mean_relation": float(relation.mean()),
        "binary_share_pct": float(100 * binary.mean() / full.mean()),
        "relation_share_pct": float(100 * relation.mean() / full.mean()),
    }
    norm_data = {
        "linear_weight_norms": linear_head_layer.weight.detach().norm(dim=1).tolist(),
        "linear_biases": linear_head_layer.bias.detach().tolist(),
        "cosine_parameter_norms": getattr(cosine, cosine_name).weight.detach().norm(dim=1).tolist(),
        "cosine_effective_norm": 1.0,
        "cosine_bias": 0.0,
    }
    report = {
        "status": "completed",
        "probe": {"dataset": "CIFAR-100 official test", "indices": [0, N_PROBE - 1], "n": int(len(labels)), "selection": "first fixed test indices; no sample selection"},
        "models": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "teacher_classifier": teacher_name,
        "ordering": ordering,
        "standardized_kl_decomposition": decomposition,
        "classifier_parameters": norm_data,
    }
    (OUT / "teaser_probe.json").write_text(json.dumps(report, indent=2) + "\n")

    # The plotting asset is rendered by a companion SVG template so the full
    # evidence artifact remains dependency-free.  This script writes only the
    # fixed-probe measurements it needs.


if __name__ == "__main__":
    main()
