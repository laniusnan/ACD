"""Frozen CIFAR-100 seed-2051 affine/angular probe with per-image scalar oracles."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "artifacts/cifar100_same_state_seed2051"
PRIOR = ROOT / "reproductions/head_transfer_20261021"
RECOVERY = ROOT / "paper_assets/acd_tables/main_results_20261116/release_recovery"
STUDENT_SHA256 = "7f8a5d67db946b80a6f3b4fe0841e609365f4cf94063892afc55c6affa847a7b"
TEACHER_SHA256 = "d945c5ab454814944c42dd027abd89b05e89648d80771bae89334e1a1a2bc64d"
ARCHIVE_SHA256 = "c70be04c99a3f827a60304beed86c3f77d0d1ec213dd775ecf20273d571f76dd"
TEMPERATURE = 4.0
LOSS_FACTOR = 8.0
ANGULAR_SCALE = 30.0


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def restore_student():
    destination = OUTPUT / "linear_seed2051_epoch240.pth"
    if destination.exists():
        if digest(destination) != STUDENT_SHA256:
            raise ValueError("Recovered student checksum mismatch")
        return destination
    archive = RECOVERY / "downloads/424967898/head_transfer_20261021-000.tar.gz"
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Release archive checksum mismatch")
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name != f"objects/{STUDENT_SHA256}":
                continue
            if not member.isfile() or member.size != 10012093:
                raise ValueError("Unexpected checkpoint archive member")
            payload = stream.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != STUDENT_SHA256:
                raise ValueError("Student payload checksum mismatch")
            destination.write_bytes(payload)
            return destination
    raise FileNotFoundError("Requested checkpoint is not in the release archive")


def load_methods():
    spec = importlib.util.spec_from_file_location("same_state_prior_methods", PRIOR / "methods.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statistics(values):
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "min": float(values.min()), "max": float(values.max())}


def oracle(logits, teacher):
    scores = logits.double() / TEMPERATURE
    scores = scores - scores.mean(1, keepdim=True)
    target_log = (teacher.double() / TEMPERATURE).log_softmax(1)
    target = target_log.exp()
    target_mean = (target * scores).sum(1)

    def derivative(scale):
        return ((scale[:, None] * scores).softmax(1) * scores).sum(1) - target_mean

    def loss(scale):
        return LOSS_FACTOR * (target * (target_log - (scale[:, None] * scores).log_softmax(1))).sum(1)

    lower = torch.zeros(len(scores), dtype=torch.float64)
    upper = torch.ones_like(lower)
    boundary = derivative(lower) >= 0
    for iteration in range(40):
        unbracketed = (derivative(upper) < 0) & ~boundary
        if not unbracketed.any():
            break
        upper[unbracketed] *= 2
    if ((derivative(upper) < 0) & ~boundary).any():
        raise ValueError("Oracle scalar minimum not bracketed")
    for iteration in range(60):
        middle = (lower + upper) / 2
        positive = derivative(middle) >= 0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    scale = torch.where(boundary, torch.zeros_like(lower), (lower + upper) / 2)
    before, after = loss(torch.ones_like(scale)), loss(scale)
    if not torch.isfinite(after).all() or (after > before + 1e-8).any() or (after < -1e-10).any():
        raise ValueError("Invalid oracle loss")
    interior = derivative(scale)[~boundary]
    residual = float(interior.abs().max()) if len(interior) else 0.0
    if residual > 1e-10:
        raise ValueError("Oracle stationarity check failed")
    arrays = {"before": before.numpy(), "after": after.numpy(), "alpha": scale.numpy()}
    summary = {field: statistics(values) for field, values in arrays.items()}
    summary.update(boundary_zero_count=int(boundary.sum()), interior_derivative_max_abs=residual,
                   calibration_increased_count=int((after > before + 1e-8).sum()))
    return arrays, summary


def infer(args, methods, student_path, identity):
    config = json.loads((PRIOR / "study.json").read_text())
    state = torch.load(student_path, map_location="cpu", weights_only=True)
    if state["epoch"] != 240:
        raise ValueError("Expected historical epoch-240 checkpoint")
    student = methods.model_for("resnet8x4", config).eval().requires_grad_(False)
    student.load_state_dict(state["student"], strict=True)
    teacher = methods.model_for("resnet32x4", config).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True), strict=True)
    states_before = {"student": methods.tensor_digest(student.state_dict()),
                     "teacher": methods.tensor_digest(teacher.state_dict())}
    dataset = torchvision.datasets.CIFAR100(args.data_root, train=False, download=False,
                                            transform=methods.clean_transform())
    if len(dataset) != 10000:
        raise ValueError("Expected the full CIFAR-100 test set")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batches = {key: [] for key in ("teacher_logits", "affine_logits", "angular_logits", "features", "labels")}
    reconstruction_error = 0.0
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            teacher_logits, _ = teacher(images)
            affine_logits, feature_map = student(images)
            features = feature_map["pooled_feat"]
            reconstructed = functional.linear(features, student.fc.weight, student.fc.bias)
            reconstruction_error = max(reconstruction_error, float((reconstructed - affine_logits).abs().max()))
            angular_logits = ANGULAR_SCALE * functional.linear(
                functional.normalize(features, dim=1), functional.normalize(student.fc.weight, dim=1))
            for key, value in (("teacher_logits", teacher_logits), ("affine_logits", affine_logits),
                               ("angular_logits", angular_logits), ("features", features), ("labels", labels)):
                batches[key].append(value.numpy())
            if batch_index % 10 == 0:
                print(f"INFERENCE {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    states_after = {"student": methods.tensor_digest(student.state_dict()),
                    "teacher": methods.tensor_digest(teacher.state_dict())}
    if states_before != states_after or reconstruction_error > 1e-6:
        raise ValueError("Frozen-state or affine reconstruction check failed")
    arrays = {key: np.concatenate(values) for key, values in batches.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Nonfinite inference output")
    checks = {"state_before": states_before, "state_after": states_after,
              "state_unchanged": True, "affine_reconstruction_max_abs": reconstruction_error,
              "zero_feature_count": int((np.linalg.norm(arrays["features"], axis=1) == 0).sum()),
              "zero_classifier_row_count": int((student.fc.weight.norm(dim=1) == 0).sum())}
    np.savez_compressed(OUTPUT / "inference.npz", **arrays)
    (OUTPUT / "inference.json").write_text(json.dumps({"identity": identity, "checks": checks,
        "arrays_sha256": digest(OUTPUT / "inference.npz")}, indent=2) + "\n")
    return arrays, checks


def run(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2051)
    if digest(args.teacher) != TEACHER_SHA256:
        raise ValueError("Teacher differs from the original release checkpoint")
    student_path = restore_student()
    methods = load_methods()
    source_paths = [Path(__file__), PRIOR / "methods.py", PRIOR / "study.json", PRIOR / "frozen/models/resnet.py"]
    identity = {"seed": 2051, "epoch": 240, "release": "model-backup-20261027",
        "release_student_path": "reproductions/head_transfer_20261021/artifacts/res32x4_res8x4__s2051__linear/last.pth",
        "student_sha256": STUDENT_SHA256, "teacher_sha256": TEACHER_SHA256,
        "archive_sha256": ARCHIVE_SHA256, "dataset": "CIFAR-100", "split": "test", "samples": 10000,
        "ordering": "torchvision CIFAR100 test order; no shuffle or augmentation",
        "data_sha256": {name: digest(args.data_root / "cifar-100-python" / name) for name in ("test", "meta")},
        "sources_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "numpy": np.__version__, "device": "cpu",
                    "threads": args.threads, "batch_size": args.batch_size}}
    cached = OUTPUT / "inference.json"
    if cached.exists():
        cache = json.loads(cached.read_text())
        if cache["identity"] != identity or cache["arrays_sha256"] != digest(OUTPUT / "inference.npz"):
            raise ValueError("Inference cache identity differs; archive it before rerunning")
        with np.load(OUTPUT / "inference.npz") as stored:
            arrays = {key: stored[key] for key in stored.files}
        checks = cache["checks"]
    else:
        arrays, checks = infer(args, methods, student_path, identity)
    teacher = torch.from_numpy(arrays["teacher_logits"])
    report = {"identity": identity, "checks": checks, "temperature": TEMPERATURE,
        "loss_factor": LOSS_FACTOR, "angular_scale": ANGULAR_SCALE, "scalar_domain": "alpha >= 0",
        "target": "fixed original teacher softmax(logits / 4) for both representations",
        "training": "none; post-hoc head substitution on one frozen linear checkpoint", "cases": {}}
    results = {}
    for name in ("affine", "angular"):
        values, summary = oracle(torch.from_numpy(arrays[f"{name}_logits"]), teacher)
        results.update({f"{name}_{key}": value for key, value in values.items()})
        report["cases"][name] = summary
    delta = results["angular_after"] - results["affine_after"]
    results["delta"] = delta
    report["paired_delta"] = {**statistics(delta), "angular_better_fraction": float((delta < 0).mean()),
                              "definition": "angular_after - affine_after"}
    report["accuracy_percent"] = {name: float(100 * (arrays[f"{name}_logits"].argmax(1) == arrays["labels"]).mean())
                                   for name in ("teacher", "affine", "angular")}
    reference = json.loads((ROOT / "reproductions/head_story_visuals_20261027/artifacts/relations.json").read_text())
    historical = reference["models"]["linear_2051"]["summary"]["all"]
    checks["historical_absolute_errors"] = {field: abs(report["cases"]["affine"][field]["mean"] - historical[key])
                                            for field, key in (("before", "kd"), ("after", "relation_residual"))}
    if max(checks["historical_absolute_errors"].values()) > 1e-5:
        raise ValueError(f"Historical linear baseline mismatch: {checks['historical_absolute_errors']}")
    np.savez_compressed(OUTPUT / "oracle_arrays.npz", **results)
    report["oracle_arrays_sha256"] = digest(OUTPUT / "oracle_arrays.npz")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": report["cases"], "paired_delta": report["paired_delta"], "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "reproductions/cifar100_acd_rerun_20261116/data")
    parser.add_argument("--teacher", type=Path, default=ROOT / "reproductions/cifar100_acd_rerun_20261116/assets/teachers/resnet32x4.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
