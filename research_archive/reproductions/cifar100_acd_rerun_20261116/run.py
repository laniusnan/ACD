"""Durable single-GPU paired reruns; explicit new protocol, final-only evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torchvision
import methods

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic(path, value, tensor=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb" if tensor else "w") as handle:
        if tensor:
            torch.save(value, handle)
        else:
            json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def identity():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + [HERE / "protocol.json", HERE / "models.json", HERE / "artifacts/assets.json"]
    paths += list((HERE / "frozen").rglob("*.py")) + list((HERE / "assets/teachers").glob("*.pth"))
    paths += [HERE / "data/cifar-100-python" / name for name in ("train", "test", "meta")]
    return {"files": {str(p.relative_to(HERE)): sha(p) for p in sorted(paths)},
            "runtime": {"python": sys.version, "torch": str(torch.__version__), "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "cuda": torch.version.cuda}}


def settings(seed, epoch=0):
    value = seed * 1000 + epoch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def recipe(pair, policy):
    return policy["recipes"]["heterogeneous" if pair in ("res32x4_sv1", "res50_mv1", "res32x4_sv2") else "homogeneous"]


def teacher(pair, config, device):
    name = config["pairs"][pair]["teacher"]
    model = methods.model(name, config)
    model.load_state_dict(torch.load(HERE / "assets/teachers" / (name + ".pth"), map_location="cpu", weights_only=True), strict=True)
    return model.to(device).requires_grad_(False).eval()


def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss = 0.; predictions = []; labels = []
    with torch.inference_mode():
        for images, targets in loader:
            z = model(images.to(device, non_blocking=True))[0]
            assert torch.isfinite(z).all()
            y = targets.to(device, non_blocking=True)
            pred = z.argmax(1)
            correct += int((pred == y).sum()); total += len(y)
            loss += float(torch.nn.functional.cross_entropy(z, y, reduction="sum"))
            predictions.extend(pred.cpu().tolist()); labels.extend(targets.tolist())
    assert total == 10000
    return {"top1": correct / total * 100, "samples": total, "ce": loss / total}, predictions, labels


def update(student, teacher_model, backbone, optimizer, batch, arm, policy, rec, device):
    images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    z = student(images)[0]
    with torch.no_grad():
        t = teacher_model(images)[0]
    loss, diag = methods.objective(z, t, targets, backbone, arm, policy, rec)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), math.inf, error_if_nonfinite=True)
    optimizer.step()
    result = {k: float(v.detach()) for k, v in diag.items()}
    result.update(loss=float(loss.detach()), gradient_norm=float(norm), correct=int((z.detach().argmax(1) == targets).sum()))
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Non-finite diagnostic")
    return result


def pairing(pair, seed, epoch, value):
    path = HERE / "artifacts/pairing" / f"{pair}_s{seed}" / f"epoch{epoch:03d}.json"
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Paired initialization/input mismatch: {path}")
    else:
        atomic(path, value)


def preflight():
    policy, config = read(HERE / "protocol.json"), read(HERE / "models.json")
    occupants = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if occupants:
        raise RuntimeError("GPU0 busy; preflight will not share or preempt")
    device = torch.device("cuda:0")
    dataset = methods.DoubleViews(HERE / "data"); assert len(dataset) == 100000
    results = {"started_at": now(), "identity": identity(), "device": torch.cuda.get_device_name(0), "pairs": {}}
    for pair in policy["pairs"]:
        settings(policy["seeds"][0])
        t = teacher(pair, config, device)
        teacher_metrics = evaluate(t, methods.test_loader(), device)[0]
        if abs(teacher_metrics["top1"] - config["pairs"][pair]["paper_teacher"]) > 2:
            raise RuntimeError(f"Teacher accuracy mismatch: {pair}: {teacher_metrics}")
        output = {"teacher": teacher_metrics, "arms": {}}
        for arm in policy["arms"]:
            settings(policy["seeds"][0])
            student, backbone, initial = methods.student(pair, policy["seeds"][0], arm, policy, config)
            student = student.to(device).train()
            # Module.to preserves Parameter identities on the installed torch runtime.
            assert all(any(p is q for q in student.parameters()) for p in backbone)
            optimizer = torch.optim.SGD(student.parameters(), **{k: recipe(pair, policy)[k] for k in ("lr",)},
                                        momentum=policy["momentum"], weight_decay=policy["weight_decay"])
            loader = methods.train_loader(dataset, policy, policy["seeds"][0], 1)
            fingerprint = hashlib.sha256(); elapsed = 0.; count = 0; last = {}
            first_batch = None
            for step, batch in enumerate(loader):
                if step == 12: break
                fingerprint.update(batch[0].numpy().tobytes()); fingerprint.update(batch[1].numpy().tobytes()); fingerprint.update(batch[2].numpy().tobytes())
                if first_batch is None: first_batch = batch
                torch.cuda.synchronize(); start = time.monotonic()
                last = update(student, t, backbone, optimizer, batch, arm, policy, recipe(pair, policy), device)
                torch.cuda.synchronize()
                if step >= 2:
                    elapsed += time.monotonic() - start; count += len(batch[1])
            # Persist and restore actual model/optimizer bytes, replay one identical update.
            snapshot = HERE / "artifacts/preflight_replay.pth"
            atomic(snapshot, {"student": student.state_dict(), "optimizer": optimizer.state_dict()}, tensor=True)
            rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
            replay1 = update(student, t, backbone, optimizer, first_batch, arm, policy, recipe(pair, policy), device)
            expected = methods.tensor_hash(student.state_dict())
            state = torch.load(snapshot, map_location=device, weights_only=True)
            student.load_state_dict(state["student"]); optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
            replay2 = update(student, t, backbone, optimizer, first_batch, arm, policy, recipe(pair, policy), device)
            assert replay1 == replay2 and methods.tensor_hash(student.state_dict()) == expected
            output["arms"][arm] = {"initial": initial, "input_sha256": fingerprint.hexdigest(),
                                   "images_per_second_compute": count / elapsed, "last_diagnostics": last, "checkpoint_replay": "exact"}
            del student, optimizer, backbone, state
            snapshot.unlink(); torch.cuda.empty_cache()
        values = list(output["arms"].values())
        assert all(v["initial"] == values[0]["initial"] and v["input_sha256"] == values[0]["input_sha256"] for v in values)
        results["pairs"][pair] = output
        atomic(HERE / "artifacts/preflight_progress.json", results)
        print(json.dumps({"preflight_pair": pair, "teacher": teacher_metrics, "arms_passed": 4}), flush=True)
        del t; torch.cuda.empty_cache()
    assert results["identity"] == identity()
    results.update(status="passed", finished_at=now())
    atomic(HERE / "artifacts/preflight.json", results)
    atomic(HERE / "artifacts/seal.json", {"identity": results["identity"], "preflight_sha256": sha(HERE / "artifacts/preflight.json")})


def train(pair, seed, arm):
    policy, config = read(HERE / "protocol.json"), read(HERE / "models.json")
    assert pair in policy["pairs"] and seed in policy["seeds"] and arm in policy["arms"]
    token = read(HERE / "artifacts/seal.json")
    assert token["identity"] == identity() and token["preflight_sha256"] == sha(HERE / "artifacts/preflight.json")
    path = HERE / "runs" / f"{pair}__s{seed}__{arm}"; path.mkdir(parents=True, exist_ok=True)
    if (path / "final.json").exists():
        return
    settings(seed)
    device = torch.device("cuda:0")
    dataset = methods.DoubleViews(HERE / "data")
    student, backbone, initial = methods.student(pair, seed, arm, policy, config)
    student = student.to(device); t = teacher(pair, config, device)
    rec = recipe(pair, policy)
    optimizer = torch.optim.SGD(student.parameters(), lr=rec["lr"], momentum=policy["momentum"], weight_decay=policy["weight_decay"])
    pairing(pair, seed, 0, initial)
    job = {"pair": pair, "seed": seed, "arm": arm}
    atomic(path / "config.json", {"job": job, "policy": policy, "recipe": rec, "initial": initial, "identity": token})
    history = []; start_epoch = 0
    if (path / "latest.pth").exists():
        state = torch.load(path / "latest.pth", map_location=device, weights_only=True)
        assert state["identity"] == token and state["job"] == job
        student.load_state_dict(state["student"]); optimizer.load_state_dict(state["optimizer"])
        history = state["history"]; start_epoch = state["epoch"]
    status = {**job, "status": "running", "pid": os.getpid(), "started_at": now(), "epoch": start_epoch}
    try:
        for epoch in range(start_epoch + 1, policy["epochs"] + 1):
            settings(seed, epoch); student.train()
            lr = rec["lr"] * policy["lr_decay"] ** sum(epoch > m for m in policy["milestones"])
            for group in optimizer.param_groups: group["lr"] = lr
            fingerprint = hashlib.sha256(); total = correct = 0; losses = 0.; started = time.monotonic()
            for step, batch in enumerate(methods.train_loader(dataset, policy, seed, epoch)):
                fingerprint.update(batch[0].numpy().tobytes()); fingerprint.update(batch[1].numpy().tobytes()); fingerprint.update(batch[2].numpy().tobytes())
                diagnostics = update(student, t, backbone, optimizer, batch, arm, policy, rec, device)
                size = len(batch[1]); total += size; correct += diagnostics["correct"]; losses += diagnostics["loss"] * size
                if step % 100 == 0:
                    status.update(epoch=epoch, batch=step + 1, heartbeat_at=now(), diagnostics=diagnostics)
                    atomic(path / "status.json", status)
                    print(json.dumps(status), flush=True)
            assert total == 100000
            pairing(pair, seed, epoch, {"input_sha256": fingerprint.hexdigest(), "samples": total})
            row = {"epoch": epoch, "lr": lr, "train_top1": correct / total * 100, "loss": losses / total,
                   "seconds": time.monotonic() - started, "input_sha256": fingerprint.hexdigest()}
            history.append(row)
            atomic(path / "latest.pth", {"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                                          "epoch": epoch, "history": history, "identity": token, "job": job}, tensor=True)
            atomic(path / "history.json", history)
            print(json.dumps({"completed_epoch": row}), flush=True)
        state = torch.load(path / "latest.pth", map_location=device, weights_only=True)
        assert state["epoch"] == 240
        student.load_state_dict(state["student"])
        metrics, predictions, labels = evaluate(student, methods.test_loader(), device)
        final = {**job, "status": "completed", "epoch": 240, "test": metrics,
                 "checkpoint_sha256": sha(path / "latest.pth"), "identity": token, "finished_at": now(),
                 "paired_epochs_verified": 240, "historical_source_parity": False}
        atomic(path / "predictions.json", {"predictions": predictions, "labels": labels})
        atomic(path / "final.json", final); atomic(path / "status.json", final)
        # Keep compact evidence outside ignored run/model directories so it can be committed.
        atomic(HERE / "results" / (path.name + ".json"), {**final, "initial": initial, "recipe": rec,
                                                        "history": history, "prediction_sha256": sha(path / "predictions.json")})
    except BaseException:
        status.update(status="failed", error=traceback.format_exc(), finished_at=now())
        atomic(path / "status.json", status)
        raise


def dispatch():
    policy = read(HERE / "protocol.json")
    assert policy["gpu_allowlist"] == [0]
    lock_path = ROOT / "queue_logs/cifar100_acd_rerun_gpu0.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [{"pair": pair, "seed": seed, "arm": arm} for seed in policy["seeds"]
                for pair in policy["pairs"] for arm in policy["arms"]]
        for index, job in enumerate(jobs):
            directory = HERE / "runs" / f"{job['pair']}__s{job['seed']}__{job['arm']}"
            if (directory / "final.json").exists(): continue
            if (directory / "status.json").exists() and read(directory / "status.json")["status"] == "failed":
                raise RuntimeError("Failed job requires diagnosis before queue restart")
            while True:
                if (HERE / "STOP").exists(): return
                pids = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                if not pids: break
                atomic(HERE / "artifacts/queue.json", {"status": "waiting_gpu0", "job": job, "heartbeat_at": now(), "occupants": pids})
                time.sleep(30)
            directory.mkdir(parents=True, exist_ok=True)
            atomic(HERE / "artifacts/queue.json", {"status": "running", "job": job, "ordinal": index + 1, "total": len(jobs), "heartbeat_at": now()})
            with (directory / "job.log").open("a") as log:
                proc = subprocess.run([sys.executable, "-u", str(HERE / "run.py"), "train", "--pair", job["pair"],
                                       "--seed", str(job["seed"]), "--arm", job["arm"]], stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                atomic(HERE / "artifacts/queue.json", {"status": "failed", "job": job, "returncode": proc.returncode, "heartbeat_at": now()})
                raise RuntimeError("Training failed; queue stopped without retry")
        atomic(HERE / "artifacts/queue.json", {"status": "completed", "total": len(jobs), "finished_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "dispatch"))
    parser.add_argument("--pair"); parser.add_argument("--seed", type=int); parser.add_argument("--arm")
    args = parser.parse_args()
    if args.action == "preflight": preflight()
    elif args.action == "train": train(args.pair, args.seed, args.arm)
    else: dispatch()
