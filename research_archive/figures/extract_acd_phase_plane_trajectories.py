#!/usr/bin/env python3
"""Extract read-only checkpoint phase-plane probes for three ACD comparisons.

Each point is a stored epoch-1/30/120/240 checkpoint, evaluated on the same
fixed 128-image CIFAR-100 official-test probe.  The two coordinates are raw
logit KL and standardized non-target conditional-TV discrepancy.  This script
loads checkpoints on CPU only and never trains, evaluates official Top-1, or
changes queue state.
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
EPOCHS = (1, 30, 120, 240)
TEMPERATURE = 4.0
PROBE_SIZE = 128


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize(logits):
    return (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True).clamp_min(1e-6)


def conditional(logits, labels):
    probs = (logits / TEMPERATURE).softmax(1)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask.scatter_(1, labels[:, None], False)
    return probs[mask].reshape(len(labels), -1) / (1 - probs.gather(1, labels[:, None])).clamp_min(1e-12)


def metrics(student_logits, teacher_logits, labels):
    raw_kl = TEMPERATURE ** 2 * F.kl_div(
        (student_logits / TEMPERATURE).log_softmax(1),
        (teacher_logits / TEMPERATURE).softmax(1), reduction="batchmean")
    relation_tv = .5 * (conditional(normalize(student_logits), labels) - conditional(normalize(teacher_logits), labels)).abs().sum(1).mean()
    return {"raw_kl": float(raw_kl), "conditional_tv_standardized": float(relation_tv)}


def logits(model, images):
    output = model(images)
    return output[0] if isinstance(output, tuple) else output


def source(name, relative):
    study = load_module(name, ROOT / relative / "study.py")
    return {"study": study, "root": ROOT / relative, "artifacts": ROOT / relative / "artifacts"}


def probe(study):
    transform = study.base.clean_transform()
    data = datasets.CIFAR100(ROOT / "data", train=False, download=False, transform=transform)
    return torch.stack([data[i][0] for i in range(PROBE_SIZE)]), torch.tensor([data[i][1] for i in range(PROBE_SIZE)])


def trajectory(source_info, pair, seed, arm, label, images, labels):
    study, artifacts = source_info["study"], source_info["artifacts"]
    job = study.jobs()[f"{pair}__s{seed}__{arm}"]
    study.core.seed_all(seed)
    built = study.build(job)
    student, teacher = built[0].eval(), built[1].eval()
    with torch.no_grad():
        teacher_logits = logits(teacher, images)
    rows, sources = [], {}
    initial = None
    for epoch in EPOCHS:
        checkpoint = artifacts / job["name"] / f"epoch_{epoch:03d}.pth"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_key = "model" if "model" in payload else "student"
        student.load_state_dict(payload[state_key], strict=True)
        student.eval()
        with torch.no_grad():
            student_logits = logits(student, images)
        rows.append({"epoch": epoch, **metrics(student_logits, teacher_logits, labels)})
        sources[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
        current = payload.get("initial")
        if initial is None:
            initial = current
        elif current != initial:
            raise RuntimeError(f"initialization changed within {job['name']}")
    return {"label": label, "pair": pair, "seed": seed, "arm": arm, "rows": rows, "initial": initial}, sources


def extract_component():
    src = source("phase_component", "reproductions/acd_cnn_interaction_20261109")
    images, labels = probe(src["study"])
    # One strictly paired heterogeneous run keeps all four traces interpretable.
    spec = (("linear_kd", "Linear KD"), ("cosine_kd", "ACI + KL"),
            ("cosine_standardized_full", "ACI + standardized full"),
            ("cosine_relation_matched", "ACI + SICD"))
    rows, hashes = [], {}
    for arm, label in spec:
        value, used = trajectory(src, "vgg13_mv1", 2102, arm, label, images, labels)
        rows.append(value); hashes.update(used)
    initials = {json.dumps(row["initial"], sort_keys=True) for row in rows}
    if len(initials) != 1:
        raise RuntimeError("component arms are not initialization-paired")
    return {"status": "completed", "question": "component ablation", "setup": {"pair": "VGG13→MobileNetV1", "seed": 2102, "epochs": list(EPOCHS), "probe": f"first {PROBE_SIZE} official CIFAR-100 test images", "temperature": TEMPERATURE, "pairing": "same interaction study and recorded initial backbone hash"}, "trajectories": rows, "sources_sha256": hashes, "limitations": ["One heterogeneous pair and one seed; checkpoints are temporal observations, not independent repetitions.", "The two axes are diagnostic discrepancies in different coordinate systems; neither is an official-test accuracy metric."]}


def extract_external():
    interaction = source("phase_external_interaction", "reproductions/acd_cnn_interaction_20261109")
    reviewer = source("phase_external_reviewer", "reproductions/cvpr_reviewer_minimal_ready_20261106")
    images, labels = probe(interaction["study"])
    spec = ((interaction, "vgg13_vgg8", 2102, "cosine_kd", "Cosine KL"),
            (reviewer, "vgg13_vgg8", 2102, "cosine_dkd", "DKD"),
            (reviewer, "vgg13_vgg8", 2101, "cosine_nckd_matched", "NCKD"),
            (reviewer, "vgg13_vgg8", 2101, "cosine_lskd_source_recipe", "LSKD"),
            (interaction, "vgg13_vgg8", 2102, "cosine_relation_matched", "ACD"))
    rows, hashes = [], {}
    for info, pair, seed, arm, label in spec:
        value, used = trajectory(info, pair, seed, arm, label, images, labels)
        rows.append(value); hashes.update(used)
    return {"status": "completed", "question": "external objective comparison", "setup": {"pair": "VGG13→VGG8", "epochs": list(EPOCHS), "probe": f"first {PROBE_SIZE} official CIFAR-100 test images", "temperature": TEMPERATURE}, "trajectories": rows, "sources_sha256": hashes, "limitations": ["The five methods are drawn from different registrations, seeds, and in LSKD's case a source recipe. This is descriptive only, not a fair comparison or ranking.", "The two axes are diagnostic discrepancies, not official-test accuracy."]}


def extract_plugin():
    at = source("phase_plugin_at", "reproductions/acd_at_res32x4_sv1_priority_20261111")
    review = source("phase_plugin_review", "reproductions/acd_reviewkd_wrn_priority_20261110")
    rows, hashes = [], {}
    for info, pair, seed, arm, label in ((at, "res32x4_sv1", 2101, "at_cosine_kl", "AT + cosine KL"),
                                         (at, "res32x4_sv1", 2101, "at_cosine_sicd", "AT + SICD"),
                                         (review, "wrn402_wrn401", 2101, "review_cosine_kl", "ReviewKD + cosine KL"),
                                         (review, "wrn402_wrn401", 2101, "review_cosine_sicd", "ReviewKD + SICD")):
        images, labels = probe(info["study"])
        value, used = trajectory(info, pair, seed, arm, label, images, labels)
        rows.append(value); hashes.update(used)
    return {"status": "completed", "question": "output-loss replacement", "setup": {"pipelines": ["AT: ResNet32×4→ShuffleNetV1", "ReviewKD: WRN40-2→WRN40-1"], "seed": 2101, "epochs": list(EPOCHS), "probe": f"first {PROBE_SIZE} official CIFAR-100 test images per pipeline", "temperature": TEMPERATURE}, "trajectories": rows, "sources_sha256": hashes, "limitations": ["One seed per pipeline; the two feature pipelines and teacher–student pairs have different scales, so their absolute locations should not be ranked against each other.", "The two axes are diagnostic discrepancies, not official-test accuracy."]}


def main():
    OUT.mkdir(exist_ok=True)
    for filename, fn in (("acd_component_ablation_phase_plane.json", extract_component), ("acd_external_kd_phase_plane.json", extract_external), ("acd_plugin_replacement_phase_plane.json", extract_plugin)):
        path = OUT / filename
        path.write_text(json.dumps(fn(), indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
