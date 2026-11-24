"""Preparation, synthetic engineering checks, and gated full ImageNet-1K training."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import data
import methods


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "artifacts"


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {"python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(HERE.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configure(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable to this process; do not silently fall back to CPU")
        torch.cuda.set_device(device)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def prepare_assets(pair):
    ASSETS.mkdir(parents=True, exist_ok=True)
    metadata_path = ASSETS / "assets.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"teachers": {}}
    weights = methods.PAIRS[pair][2]
    filename = Path(urlparse(weights.url).path).name
    destination = ASSETS / filename
    prefix = filename.rsplit("-", 1)[1].split(".")[0]
    if not destination.exists():
        torch.hub.download_url_to_file(weights.url, destination, hash_prefix=prefix, progress=False)
    checksum = data.sha256_file(destination)
    if not checksum.startswith(prefix):
        raise ValueError("Teacher checkpoint does not match the official filename hash")
    metadata["teachers"][pair] = {"path": filename, "sha256": checksum, "url": weights.url,
                                   "enum": str(weights), "transform": str(weights.transforms())}
    index = ASSETS / "imagenet_class_index.json"
    if not index.exists():
        torch.hub.download_url_to_file(data.CLASS_INDEX_URL, index, progress=False)
    if data.sha256_file(index) != data.CLASS_INDEX_SHA256:
        raise ValueError("Downloaded class index differs from the pinned official metadata")
    data.class_names(index)
    metadata["class_index"] = {"url": data.CLASS_INDEX_URL, "sha256": data.sha256_file(index)}
    data.save_json(metadata_path, metadata)
    data.save_json(ASSETS / "environment.json", environment())
    return metadata


def load_teacher(pair, device):
    metadata = json.loads((ASSETS / "assets.json").read_text())["teachers"][pair]
    path = ASSETS / metadata["path"]
    expected_weights = methods.PAIRS[pair][2]
    if metadata["enum"] != str(expected_weights) or metadata["url"] != expected_weights.url:
        raise ValueError("Teacher identity changed")
    checksum = data.sha256_file(path)
    prefix = Path(urlparse(expected_weights.url).path).stem.rsplit("-", 1)[1]
    if checksum != metadata["sha256"] or not checksum.startswith(prefix):
        raise ValueError("Teacher checkpoint integrity failure")
    teacher = methods.PAIRS[pair][0](weights=None)
    teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return teacher.requires_grad_(False).eval().to(device), metadata


def save_checkpoint(path, student, optimizer, scheduler, epoch, identity):
    state = np.random.get_state()
    payload = {"student": student.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "epoch": epoch, "identity": identity,
               "python_rng": random.getstate(), "numpy_rng": [state[0], torch.from_numpy(state[1]),
                    state[2], state[3], state[4]], "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def restore_checkpoint(path, student, optimizer, scheduler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"] != identity:
        raise ValueError("Resume refuses changed source, protocol, data, environment, or teacher")
    student.load_state_dict(payload["student"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    state = payload["numpy_rng"]
    np.random.set_state((state[0], state[1].numpy(), state[2], state[3], state[4]))
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["epoch"]


def optimizer_for(student, config):
    optimizer = torch.optim.SGD(student.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"],
                                                    gamma=config["lr_decay"])
    return optimizer, scheduler


def step(student, teacher, backbone, optimizer, images, labels, arm, config):
    optimizer.zero_grad(set_to_none=True)
    if methods.ARMS[arm][1] == "ce":
        teacher_logits = images.new_zeros((len(images), 1000))
    else:
        with torch.no_grad():
            teacher_logits = teacher(images)
    logits = student(images)
    if logits.shape != (len(images), 1000):
        raise ValueError("Expected B x 1000 student logits")
    loss, diagnostics = methods.objective(logits, teacher_logits, labels, backbone, arm, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"),
                                                  error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "student_gradient_norm": float(gradient_norm),
            **{name: float(value) for name, value in diagnostics.items()}}


def loader(dataset, config, epoch, train):
    generator = torch.Generator().manual_seed(config["seed"] + 100000 + (epoch if train else 0))
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=train, drop_last=False,
                      num_workers=config["workers"], pin_memory=True, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=False)


@torch.inference_mode()
def evaluate(model, dataset, config, device):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    for images, labels in loader(dataset, config, 0, False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        predictions = model(images).float().topk(5, dim=1).indices
        correct1 += int(predictions[:, 0].eq(labels).sum())
        correct5 += int(predictions.eq(labels[:, None]).any(1).sum())
        total += len(labels)
    if total != 50000:
        raise ValueError("Official evaluation must cover all 50000 validation images")
    return {"count": total, "top1": 100 * correct1 / total, "top5": 100 * correct5 / total}


def formal_gate(config, arm, root, audit_path):
    blockers = []
    if not root or not Path(root).is_dir():
        blockers.append("authorized full ImageNet-1K data root is missing")
    if not Path(audit_path).is_file():
        blockers.append("full dataset integrity/label audit has not passed")
    if config["status"] != "frozen_after_real_data_preflight":
        blockers.append("protocol is a candidate; real-data preflight and recipe freeze required")
    if config["precision"] != "fp32":
        blockers.append("only tested FP32 is implemented; BF16 is not yet validated")
    if methods.ARMS[arm][1] == "sicd" and not methods.SICD_REFERENCE_VERIFIED:
        blockers.append("SICD frozen source missing; equation reconstruction is smoke-only")
    if not (ASSETS / "assets.json").is_file():
        blockers.append("teacher checkpoints and class index are not prepared")
    return blockers


@contextlib.contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def train(args, config):
    blockers = formal_gate(config, args.arm, args.data_root, args.audit)
    if blockers:
        evidence = {"status": "blocked_before_training", "time": now(), "arm": args.arm,
                    "blockers": blockers, "source_sha256": source_hash()}
        data.save_json(args.output / "launch_status.json", evidence)
        print(json.dumps(evidence, indent=2), flush=True)
        return 2
    configure(config["seed"], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("At least 10 GiB output filesystem headroom required")
    with run_lock(args.output):
        if (args.output / "final.json").exists():
            raise FileExistsError("Completed run cannot be overwritten")
        checkpoint = args.output / "latest.pth"
        if (args.output / "identity.json").exists() and not args.resume:
            raise FileExistsError("Existing run requires explicit --resume")
        datasets = data.audited_datasets(args.data_root, ASSETS / "imagenet_class_index.json", args.audit)
        student, backbone, initial = methods.build_student(config["pair"], methods.ARMS[args.arm][0],
                                                           config["cosine_scale"], config["seed"])
        student.to(args.device)
        teacher, teacher_identity = load_teacher(config["pair"], args.device)
        identity = {"arm": args.arm, "config": config, "source_sha256": source_hash(),
                    "data_audit_sha256": data.sha256_file(args.audit), "teacher": teacher_identity,
                    "initial": initial, "environment": environment()}
        optimizer, scheduler = optimizer_for(student, config)
        start = 0
        if args.resume:
            previous = json.loads((args.output / "identity.json").read_text())
            if previous != identity:
                raise ValueError("Resume refuses changed run identity")
            if checkpoint.exists():
                start = restore_checkpoint(checkpoint, student, optimizer, scheduler, identity)
        data.save_json(args.output / "identity.json", identity)
        teacher_metrics = evaluate(teacher, datasets["val"], config, args.device)
        expected = 76.130 if config["pair"] == "resnet50_mobilenet_v2" else 73.314
        if abs(teacher_metrics["top1"] - expected) > 2.0:
            raise ValueError(f"Teacher accuracy sanity check failed: {teacher_metrics}")
        data.save_json(args.output / "teacher_validation.json", teacher_metrics)
        for epoch in range(start, config["epochs"]):
            student.train()
            started = time.monotonic()
            samples = 0
            sums = {}
            learning_rate = optimizer.param_groups[0]["lr"]
            for batch, (images, labels) in enumerate(loader(datasets["train"], config, epoch, True)):
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                metrics = step(student, teacher, backbone, optimizer, images, labels, args.arm, config)
                samples += len(labels)
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.0) + value * len(labels)
                if batch % 100 == 0:
                    print(json.dumps({"epoch": epoch + 1, "batch": batch, **metrics}), flush=True)
            scheduler.step()
            save_checkpoint(checkpoint, student, optimizer, scheduler, epoch + 1, identity)
            record = {"epoch": epoch + 1, "samples": samples, "lr": learning_rate,
                      "seconds": time.monotonic() - started,
                      **{name: value / samples for name, value in sums.items()}}
            data.save_json(args.output / f"epoch_{epoch + 1:03d}.json", record)
            print(json.dumps(record), flush=True)
        restored_epoch = restore_checkpoint(checkpoint, student, optimizer, scheduler, identity)
        metrics = evaluate(student, datasets["val"], config, args.device)
        data.save_json(args.output / "final.json", {"status": "completed", "epoch": restored_epoch,
                       "reloaded_checkpoint_sha256": data.sha256_file(checkpoint),
                       "validation": metrics, "time": now()})
    return 0


def smoke(args, config):
    if args.steps < 2 or args.batch_size < 2:
        raise ValueError("Smoke needs batch >= 2 and at least two measured steps")
    configure(config["seed"], args.device)
    config = dict(config, batch_size=args.batch_size)
    device = torch.device(args.device)
    teacher, teacher_identity = load_teacher(config["pair"], device)
    generator = torch.Generator().manual_seed(12345)
    images = torch.randn(args.batch_size, 3, 224, 224, generator=generator).to(device)
    labels = torch.randint(0, 1000, (args.batch_size,), generator=generator).to(device)
    evidence = {"kind": "synthetic_engineering_only_not_imagenet_accuracy", "time": now(),
                "config": config, "environment": environment(), "teacher": teacher_identity,
                "source_sha256": source_hash(), "arms": {}}
    with run_lock(args.output):
        if (args.output / "smoke.json").exists():
            raise FileExistsError("Use a fresh output directory for each engineering smoke")
        for arm in args.arms:
            configure(config["seed"], device)
            student, backbone, initial = methods.build_student(config["pair"], methods.ARMS[arm][0],
                                                               config["cosine_scale"], config["seed"])
            student.to(device).train()
            optimizer, scheduler = optimizer_for(student, config)
            checkpoint = args.output / "smoke_resume.pth"
            metrics = step(student, teacher, backbone, optimizer, images, labels, arm, config)
            scheduler.step()
            save_checkpoint(checkpoint, student, optimizer, scheduler, 1, {"arm": arm})
            first = step(student, teacher, backbone, optimizer, images, labels, arm, config)
            expected_state = {name: value.detach().cpu().clone()
                              for name, value in student.state_dict().items()}
            restore_checkpoint(checkpoint, student, optimizer, scheduler, {"arm": arm})
            replay = step(student, teacher, backbone, optimizer, images, labels, arm, config)
            maximum_error = max(float((value.detach().cpu() - expected_state[name]).abs().max())
                                for name, value in student.state_dict().items())
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value.cpu(), expected_state[name], rtol=1e-5, atol=1e-6)
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("Teacher unexpectedly received gradients")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            for iteration in range(args.steps):
                metrics = step(student, teacher, backbone, optimizer, images, labels, arm, config)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            with torch.no_grad():
                teacher_logits = teacher(images)
                probability = (methods.standardize(teacher_logits) / config["temperature"]).softmax(1)
                teacher_target = probability.gather(1, labels[:, None]).mean().item()
            evidence["arms"][arm] = {"initial": initial, "metrics": metrics,
                "resume_max_absolute_error": maximum_error, "resume_loss_delta": abs(first["loss"] - replay["loss"]),
                "teacher_standardized_target_probability": teacher_target,
                "teacher_standardized_entropy": float(-(probability * probability.log()).sum(1).mean()),
                "synthetic_images_per_second": args.steps * args.batch_size / elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None}
            data.save_json(args.output / "smoke.json", evidence)
            print(json.dumps({"arm": arm, **evidence["arms"][arm]}), flush=True)
            del student, backbone, optimizer, scheduler, expected_state
            if device.type == "cuda":
                torch.cuda.empty_cache()
        checkpoint.unlink()
        evidence["status"] = "passed_synthetic_only"
        data.save_json(args.output / "smoke.json", evidence)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "smoke", "train"))
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--pair", choices=tuple(methods.PAIRS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default=os.environ.get("IMAGENET_ROOT"))
    parser.add_argument("--audit", type=Path, default=ASSETS / "dataset_audit.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=tuple(methods.ARMS), default="I7")
    parser.add_argument("--arms", nargs="+", choices=tuple(methods.ARMS), default=list(methods.ARMS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    if args.pair:
        config["pair"] = args.pair
    if args.command == "prepare":
        print(json.dumps(prepare_assets(config["pair"]), indent=2))
        return 0
    if args.command == "audit":
        if not args.data_root:
            parser.error("--data-root or IMAGENET_ROOT must point to authorized full ImageNet-1K")
        data.audit(args.data_root, ASSETS / "imagenet_class_index.json", args.audit)
        return 0
    if args.output is None:
        parser.error("--output is required for smoke/train to isolate results")
    if args.command == "train" and not args.device.startswith("cuda"):
        parser.error("Formal ImageNet training requires CUDA")
    return smoke(args, config) if args.command == "smoke" else train(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
