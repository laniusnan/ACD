"""Fixed-sample CPU diagnostics for ACD analysis figures."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as functional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "reproductions/imagenet1k_acd_20261113"
sys.path.insert(0, str(PACKAGE))
import data
import methods
import run as experiment

ARTIFACTS = HERE / "artifacts"
TEMPERATURE = 4.0
SEED = 20261116


def emit(**values):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), **values)), flush=True)


def save(name, value):
    data.save_json(ARTIFACTS / name, value)


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def normalize(logits):
    return methods.standardize(logits, 1e-6)


def probabilities(logits):
    return (logits / TEMPERATURE).softmax(1)


def conditional(logits, labels):
    mask = torch.ones_like(logits, dtype=torch.bool).scatter_(1, labels[:, None], False)
    return (logits[mask].reshape(len(labels), -1) / TEMPERATURE).softmax(1)


def full_rows(logits, target):
    target_log = (target / TEMPERATURE).log_softmax(1)
    return TEMPERATURE ** 2 * (target_log.exp() * (target_log - (logits / TEMPERATURE).log_softmax(1))).sum(1)


def oracle(logits, target):
    scores = logits.double() / TEMPERATURE
    scores = scores - scores.mean(1, keepdim=True)
    target_probability = probabilities(target.double())
    target_mean = (target_probability * scores).sum(1)

    def derivative(scale):
        return ((scale[:, None] * scores).softmax(1) * scores).sum(1) - target_mean

    lower = torch.zeros(len(scores), dtype=torch.float64)
    upper = torch.ones_like(lower)
    boundary = derivative(lower) >= 0
    for iteration in range(40):
        unbracketed = (derivative(upper) < 0) & ~boundary
        if not unbracketed.any():
            break
        upper[unbracketed] *= 2
    if ((derivative(upper) < -1e-12) & ~boundary).any():
        raise ValueError("Oracle scalar minimum was not bracketed")
    for iteration in range(60):
        middle = (lower + upper) / 2
        positive = derivative(middle) >= 0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    scale = torch.where(boundary, torch.zeros_like(lower), (lower + upper) / 2)
    before = full_rows(logits.double(), target.double())
    after = full_rows(logits.double() * scale[:, None], target.double())
    if (after > before + 1e-8).any():
        raise ValueError("Oracle optimization increased KL")
    return {"before": before.tolist(), "after": after.tolist(), "alpha": scale.tolist(),
            "boundary_zero_count": int(boundary.sum()),
            "interior_derivative_max_abs": float(derivative(scale)[~boundary].abs().max()) if (~boundary).any() else None}


def sample_manifest():
    root = Path("/fastdata/users/iaii2026/datasets/imagenet1k")
    audit_path = PACKAGE / "artifacts/dataset_audit.json"
    audit = json.loads(audit_path.read_text())
    index_path = PACKAGE / "artifacts/imagenet_class_index.json"
    if not audit["decoded_all"] or not audit["content_hashed_all"] or audit["root"] != str(root):
        raise ValueError("Independent complete dataset audit required")
    if data.sha256_file(index_path) != data.CLASS_INDEX_SHA256:
        raise ValueError("Class mapping changed")
    generator = np.random.default_rng(SEED)
    names = data.class_names(index_path)
    categories = generator.permutation(1000)[:128].tolist()
    rows = []
    diagnostic_indices, embedding_indices = [], []
    for position, label in enumerate(categories):
        paths = sorted((root / "train" / names[label]).glob("*.JPEG"))
        needed = 20 if position < 12 else 2
        selected = generator.choice(len(paths), needed, replace=False)
        for local, file_index in enumerate(selected):
            path = paths[int(file_index)]
            entry_index = len(rows)
            rows.append({"path": str(path.relative_to(root)), "label": label,
                         "size": path.stat().st_size, "sha256": data.sha256_file(path)})
            if local < 2:
                diagnostic_indices.append(entry_index)
            if position < 12:
                embedding_indices.append(entry_index)
    gradient_indices = generator.permutation(diagnostic_indices)[:64].tolist()
    result = {"root": str(root), "split": "train", "seed": SEED,
              "sampling": "128 random classes x 2 images; first 12 chosen classes x 20 images for t-SNE",
              "selection_independent_of_model_outputs": True, "rows": rows,
              "diagnostic_indices": diagnostic_indices, "embedding_indices": embedding_indices,
              "gradient_indices": gradient_indices, "gradient_batches": 8, "gradient_batch_size": 8,
              "audit_sha256": data.sha256_file(audit_path), "class_index_sha256": data.sha256_file(index_path),
              "transform": str(data.transform(False)), "teacher_may_have_seen_training_images": True}
    save("samples.json", result)
    return result


def load_images(manifest, indices):
    transform = data.transform(False)
    images = []
    for index in indices:
        path = Path(manifest["root"]) / manifest["rows"][index]["path"]
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images)


def checkpoint_model(path, head):
    digest = data.sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if data.sha256_file(path) != digest:
        raise ValueError("Checkpoint changed while reading")
    model, backbone, initial = methods.build_student("resnet50_mobilenet_v2", head, 30, 3101)
    model.load_state_dict(payload["student"], strict=True)
    model.eval()
    return model, backbone, {"path": str(path.relative_to(ROOT)), "sha256": digest,
                             "stored_epoch_field": payload["epoch"], "identity": payload["identity"]}


def features_and_logits(model, images, head_name):
    captured = []
    hook = model.get_submodule(head_name).register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            outputs = model(images)
    finally:
        hook.remove()
    return captured[0], outputs


def extract(manifest):
    paths = {
        "ce_final": (PACKAGE / "runs/I0_seed3101_baseline_20261114/latest.pth", "linear"),
        "kd_pilot": (PACKAGE / "artifacts/real_preflight_20261114_gpu0/I2_preflight_resume.pth", "linear"),
        "acd_pilot": (PACKAGE / "artifacts/real_preflight_20261114_gpu1/I7_preflight_resume.pth", "cosine"),
    }
    config = json.loads((PACKAGE / "protocol.json").read_text())
    pilot_reports = [json.loads((PACKAGE / f"artifacts/real_preflight_20261114_gpu{device}/preflight.json").read_text()) for device in (0, 1)]
    if any(report["status"] != "passed_real_data_preflight_not_formal_training" for report in pilot_reports):
        raise ValueError("Pilot provenance is incomplete")
    if pilot_reports[0]["arms"]["I2"]["initial"] != pilot_reports[1]["arms"]["I7"]["initial"]:
        raise ValueError("Pilot initialization was not paired")
    arrays = {"labels": np.array([row["label"] for row in manifest["rows"]])}
    sources = {}
    for state, (path, head) in paths.items():
        emit(phase="feature_extraction", state=state, images=len(manifest["rows"]))
        model, backbone, identity = checkpoint_model(path, head)
        if state == "ce_final":
            final = json.loads((path.parent / "final.json").read_text())
            if final["reloaded_checkpoint_sha256"] != identity["sha256"] or final["epoch"] != 100:
                raise ValueError("CE endpoint does not match the completed run")
        elif identity["identity"].get("kind") != "real_data_preflight_only":
            raise ValueError("Expected a short real-data pilot checkpoint")
        sources[state] = identity
        feature_rows, output_rows = [], []
        for offset in range(0, len(manifest["rows"]), 16):
            indices = list(range(offset, min(offset + 16, len(manifest["rows"]))))
            features, outputs = features_and_logits(model, load_images(manifest, indices), "classifier.1")
            feature_rows.append(features)
            output_rows.append(outputs)
        arrays[state + "_features"] = torch.cat(feature_rows).numpy()
        arrays[state + "_logits"] = torch.cat(output_rows).numpy()
        arrays[state + "_weight"] = model.classifier[1].weight.detach().numpy()
        if head == "linear":
            arrays[state + "_bias"] = model.classifier[1].bias.detach().numpy()
        np.savez_compressed(ARTIFACTS / "features.npz", **arrays)
        del model, backbone
    teacher, teacher_identity = experiment.load_teacher(config["pair"], "cpu")
    outputs = []
    emit(phase="teacher_inference", images=len(manifest["rows"]))
    for offset in range(0, len(manifest["rows"]), 16):
        indices = list(range(offset, min(offset + 16, len(manifest["rows"]))))
        with torch.no_grad():
            outputs.append(teacher(load_images(manifest, indices)))
    arrays["teacher_logits"] = torch.cat(outputs).numpy()
    np.savez_compressed(ARTIFACTS / "features.npz", **arrays)
    sources["teacher"] = teacher_identity
    sources["pilot_updates"] = 210
    sources["pilot_epoch_field_is_checkpoint_replay_marker"] = True
    sources["pilot_config"] = config
    sources["pilot_reports"] = [{"path": str((PACKAGE / f"artifacts/real_preflight_20261114_gpu{device}/preflight.json").relative_to(ROOT)),
                                  "sha256": data.sha256_file(PACKAGE / f"artifacts/real_preflight_20261114_gpu{device}/preflight.json")} for device in (0, 1)]
    save("sources.json", sources)
    return arrays, sources


def gradient_comparison(vector, reference):
    norms = vector.norm(dim=1)
    reference_norms = reference.norm(dim=1)
    valid = (norms > 1e-12) & (reference_norms > 1e-12)
    cosine = functional.cosine_similarity(vector[valid], reference[valid], dim=1, eps=1e-20)
    return {"deviation": (1 - cosine).tolist(), "norm_ratio": (norms[valid] / reference_norms[valid]).tolist(),
            "excluded_zero_gradients": int((~valid).sum())}


def head_probes(arrays, manifest):
    indices = manifest["diagnostic_indices"]
    features = torch.from_numpy(arrays["ce_final_features"][indices]).double()
    weight = torch.from_numpy(arrays["ce_final_weight"]).double()
    bias = torch.from_numpy(arrays["ce_final_bias"]).double()
    target = torch.from_numpy(arrays["teacher_logits"][indices]).double()
    labels = torch.from_numpy(arrays["labels"][indices])
    raw = functional.linear(features, weight, bias)
    saved_logits = torch.from_numpy(arrays["ce_final_logits"][indices]).double()
    if not torch.allclose(raw, saved_logits, atol=1e-4, rtol=1e-4):
        raise ValueError("Feature/head reconstruction differs from model output")
    unit_weight = functional.normalize(weight, dim=1)
    angular = 30 * functional.linear(functional.normalize(features, dim=1), unit_weight)
    calibrated = {"raw": oracle(raw, target), "angular": oracle(angular, target)}
    save("oracle_same_state.json", {"state": "CE epoch100; angular is a post-hoc head substitution",
                                    "temperature": TEMPERATURE, "angular_scale": 30, "objective": "T^2 KL(teacher||student)",
                                    "domain": "alpha>=0; alpha=0 is the positive-domain boundary limit", "cases": calibrated})
    base_feature = features.clone().requires_grad_(True)
    base_logits = functional.linear(base_feature, weight, bias)
    base_gradient = torch.autograd.grad(full_rows(base_logits, target).sum(), base_feature)[0]
    base_conditional = conditional(raw, labels)
    base_rank = raw.clone().scatter_(1, labels[:, None], -torch.inf).argmax(1)
    norm = weight.norm(dim=1)
    geometric_mean = norm.log().mean().exp()
    factors = []
    for factor, strengths in (("common_scale", [0.25, 0.5, 0.75, 1, 1.5, 2, 4]),
                               ("class_norm", [0, 0.25, 0.5, 0.75, 1, 1.5, 2]),
                               ("class_bias", [0, 0.25, 0.5, 0.75, 1, 1.5, 2])):
        for strength in strengths:
            feature = features.clone().requires_grad_(True)
            if factor == "common_scale":
                logits = strength * functional.linear(feature, weight, bias)
            elif factor == "class_norm":
                scaled_norm = geometric_mean * (strength * (norm / geometric_mean).log()).exp()
                logits = functional.linear(feature, scaled_norm[:, None] * unit_weight, bias)
            else:
                logits = functional.linear(feature, weight, bias.mean() + strength * (bias - bias.mean()))
            gradient = torch.autograd.grad(full_rows(logits, target).sum(), feature)[0]
            rank = logits.detach().clone().scatter_(1, labels[:, None], -torch.inf).argmax(1)
            entry = {"factor": factor, "strength": strength,
                     "conditional_tv": (0.5 * (conditional(logits.detach(), labels) - base_conditional).abs().sum(1)).tolist(),
                     "top1_alternative_changed": (rank != base_rank).double().tolist(),
                     **gradient_comparison(gradient, base_gradient)}
            if factor == "common_scale" and any(entry["top1_alternative_changed"]):
                raise ValueError("Positive common scaling changed the ranking")
            if strength == 1 and max(entry["conditional_tv"]) > 1e-8:
                raise ValueError("Factor identity control failed")
            factors.append(entry)
    save("parameterization.json", {"reference": "unaltered CE epoch100 affine head", "rows": factors})
    errors = {"raw": TEMPERATURE * (probabilities(raw) - probabilities(target)),
              "angular": TEMPERATURE * (probabilities(angular) - probabilities(target))}
    decomposition = []
    for jacobian in ("raw", "angular"):
        for error_name, error in errors.items():
            feature = features.clone().requires_grad_(True)
            logits = (functional.linear(feature, weight, bias) if jacobian == "raw" else
                      30 * functional.linear(functional.normalize(feature, dim=1), unit_weight))
            gradient = torch.autograd.grad(logits, feature, grad_outputs=error)[0]
            decomposition.append({"jacobian": jacobian, "probability_error": error_name,
                                  **gradient_comparison(gradient, base_gradient)})
    save("gradient_sources.json", {"rows": decomposition,
                                   "reference": "raw Jacobian + raw probability error; gradients w.r.t. pre-head features",
                                   "cross_combinations_are_diagnostic_vectors": True})
    target_dependency = []
    raw_conditional = conditional(raw, labels)
    normalized_conditional = conditional(normalize(raw), labels)
    teacher_normalized = normalize(target)
    for changed_side in ("student", "teacher"):
        for shift in (-8, -4, -2, 0, 2, 4, 8):
            student_changed, teacher_changed = raw.clone(), target.clone()
            changed = student_changed if changed_side == "student" else teacher_changed
            changed.scatter_add_(1, labels[:, None], (shift * changed.std(1, correction=1))[:, None])
            student_standardized, teacher_standardized = normalize(student_changed), normalize(teacher_changed)
            distribution = conditional(student_standardized, labels)
            teacher_probability = probabilities(teacher_standardized)
            mass = 1 - teacher_probability.gather(1, labels[:, None]).squeeze(1)
            teacher_conditional = conditional(teacher_standardized, labels)
            relation = TEMPERATURE ** 2 * (teacher_conditional * (teacher_conditional.log() - distribution.log())).sum(1)
            original = raw if changed_side == "student" else target
            changed_conditional = conditional(normalize(changed), labels)
            target_dependency.append({"side": changed_side, "shift_std_units": shift,
                "raw_conditional_tv": (0.5 * (conditional(changed, labels) - conditional(original, labels)).abs().sum(1)).tolist(),
                "standardized_conditional_tv": (0.5 * (changed_conditional - conditional(normalize(original), labels)).abs().sum(1)).tolist(),
                "teacher_non_target_mass": mass.tolist(), "weighted_relation": (mass * relation).tolist()})
    probabilities_teacher = probabilities(teacher_normalized)
    probabilities_student = probabilities(normalize(raw))
    teacher_target = probabilities_teacher.gather(1, labels[:, None])
    student_target = probabilities_student.gather(1, labels[:, None])
    binary_teacher = torch.cat((teacher_target, 1-teacher_target), 1)
    binary_student = torch.cat((student_target, 1-student_target), 1)
    binary = TEMPERATURE ** 2 * (binary_teacher * (binary_teacher.log()-binary_student.log())).sum(1)
    rel_teacher = conditional(teacher_normalized, labels)
    relation = TEMPERATURE ** 2 * (1-teacher_target.squeeze(1)) * (rel_teacher * (rel_teacher.log()-normalized_conditional.log())).sum(1)
    full = full_rows(normalize(raw), teacher_normalized)
    checks = {"kl_decomposition_max_abs_error": float((full-binary-relation).abs().max()),
              "standardization_affine_max_abs_error": float((normalize(raw*3.5+11)-normalize(raw)).abs().max()),
              "angular_standardization_cancellation_max_abs_error": float((normalize(angular)-normalize(functional.linear(features, unit_weight))).abs().max()),
              "raw_conditional_target_intervention_max_tv": max(max(row["raw_conditional_tv"]) for row in target_dependency)}
    if max(checks.values()) > 1e-8:
        raise ValueError(f"Mathematical identity checks failed: {checks}")
    save("conditional_dependency.json", {"rows": target_dependency, "numerical_checks": checks,
        "conditional_loss": "teacher-mass-weighted standardized conditional KL, before backbone norm matching"})
    emit(phase="head_probes_finished", numerical_checks=checks)


def gradient_stats(gradient, ce_gradient):
    norm = gradient.double().norm()
    ce_norm = ce_gradient.double().norm()
    if norm < 1e-12 or ce_norm < 1e-12:
        return {"cosine": None, "norm_ratio": None, "excluded_zero_gradient": True}
    return {"cosine": float(functional.cosine_similarity(gradient.double(), ce_gradient.double(), dim=0, eps=1e-20)),
            "norm_ratio": float(norm/ce_norm), "excluded_zero_gradient": False}


def backbone_probes(manifest, arrays, sources):
    cases = (("ce_final", "linear"), ("ce_final_angular", "cosine"),
             ("kd_pilot", "linear"), ("acd_pilot", "cosine"))
    rows = []
    state_checks = {}
    for case, head in cases:
        source = "ce_final" if case == "ce_final_angular" else case
        model, backbone, identity = checkpoint_model(ROOT / sources[source]["path"], "linear" if source == "ce_final" else head)
        if case == "ce_final_angular":
            model.classifier[1] = methods.CosineHead(model.classifier[1].weight, 30)
            model.eval()
        if identity["sha256"] != sources[source]["sha256"]:
            raise ValueError("Source checkpoint changed")
        before = methods.tensor_digest(model.state_dict())
        for batch in range(manifest["gradient_batches"]):
            indices = manifest["gradient_indices"][batch*8:(batch+1)*8]
            images = load_images(manifest, indices)
            labels = torch.tensor([manifest["rows"][index]["label"] for index in indices])
            teacher_logits = torch.from_numpy(arrays["teacher_logits"][indices])
            logits = model(images)
            losses = {"ce": functional.cross_entropy(logits, labels),
                      "raw_kl": methods.full_kl(logits, teacher_logits, TEMPERATURE),
                      "standardized_full": methods.full_kl(normalize(logits), normalize(teacher_logits), TEMPERATURE),
                      "conditional": methods.conditional_kl(normalize(logits), normalize(teacher_logits), labels, TEMPERATURE)}
            gradients = {}
            for name, loss in losses.items():
                values = torch.autograd.grad(loss, backbone, retain_graph=name != "conditional")
                gradient = torch.cat([value.detach().reshape(-1) for value in values])
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Non-finite backbone gradient")
                gradients[name] = gradient
            ratio = float(gradients["standardized_full"].norm()/gradients["conditional"].norm().clamp_min(1e-12))
            matched = ratio * gradients["conditional"]
            gradients["matched_conditional"] = matched
            conditional_stats = gradient_stats(gradients["conditional"], gradients["ce"])
            matched_stats = gradient_stats(matched, gradients["ce"])
            if conditional_stats["cosine"] is not None and abs(conditional_stats["cosine"]-matched_stats["cosine"]) > 1e-6:
                raise ValueError("Detached positive norm matching changed cosine")
            for name in ("raw_kl", "standardized_full", "conditional", "matched_conditional"):
                rows.append({"state": case, "batch": batch, "objective": name,
                             "gradient_match_ratio": ratio,
                             **gradient_stats(gradients[name], gradients["ce"]),
                             "combined_gradient_ce_cosine": gradient_stats(gradients["ce"]+gradients[name], gradients["ce"])["cosine"]})
            emit(phase="backbone_gradient_probe", state=case, batch=batch+1)
            save("backbone_gradients.json", {"rows": rows, "mode": "eval; frozen BN and disabled dropout",
                 "parameter_set": "all trainable student parameters except classifier.1", "batch_size": 8,
                 "batches_per_state": 8, "same_images_across_states": True,
                 "same_state_cross_objectives": True, "training_batch_size_was": 256,
                 "state_unchanged": state_checks})
            del gradients, losses, logits, images
        if before != methods.tensor_digest(model.state_dict()):
            raise ValueError("Probe changed model parameters or buffers")
        state_checks[case] = True
        del model, backbone
    save("backbone_gradients.json", {"rows": rows, "mode": "eval; frozen BN and disabled dropout",
         "parameter_set": "all trainable student parameters except classifier.1", "batch_size": 8,
         "batches_per_state": 8, "same_images_across_states": True, "same_state_cross_objectives": True,
         "training_batch_size_was": 256, "state_unchanged": state_checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-cache", action="store_true")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    started = time.monotonic()
    save("status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                          "device": "cpu", "threads": 4, "source_sha256": data.sha256_file(__file__)})
    if args.resume_cache:
        manifest = json.loads((ARTIFACTS / "samples.json").read_text())
        arrays = dict(np.load(ARTIFACTS / "features.npz"))
        sources = json.loads((ARTIFACTS / "sources.json").read_text())
        for name in ("ce_final", "kd_pilot", "acd_pilot"):
            if data.sha256_file(ROOT / sources[name]["path"]) != sources[name]["sha256"]:
                raise ValueError("Cached source identity changed")
    else:
        if (ARTIFACTS / "features.npz").exists():
            raise FileExistsError("Use --resume-cache to reuse the completed feature extraction")
        manifest = sample_manifest()
        arrays, sources = extract(manifest)
    head_probes(arrays, manifest)
    backbone_probes(manifest, arrays, sources)
    save("status.json", {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
                          "seconds": time.monotonic()-started, "device": "cpu", "threads": 4,
                          "source_sha256": data.sha256_file(__file__), "training_started": False})
    emit(phase="all_cpu_probes_completed", seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
