"""Apply seed/arm scheduling restrictions without changing sealed training code."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SCHEDULE = Path(__file__).with_name("schedule.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def plan():
    policy, scope = read(HERE / "protocol.json"), read(SCHEDULE)
    assert scope["allowed_seeds"] == [3201] and scope["cancelled_seeds"] == [3202, 3203]
    assert set(scope["allowed_seeds"] + scope["cancelled_seeds"]) == set(policy["seeds"])
    assert scope["allowed_arms"] == ["acd"]
    assert set(scope["allowed_arms"] + scope["deferred_arms"]) == set(policy["arms"])
    jobs = [{"pair": pair, "seed": seed, "arm": arm} for seed in scope["allowed_seeds"]
            for pair in policy["pairs"] for arm in scope["allowed_arms"]]
    assert len(jobs) == scope["expected_jobs"] == 7
    assert len(policy["pairs"]) * len(policy["arms"]) * len(scope["cancelled_seeds"]) == scope["cancelled_jobs"] == 56
    assert len(policy["pairs"]) * len(scope["allowed_seeds"]) * len(scope["deferred_arms"]) == scope["deferred_jobs"] == 21
    return jobs, scope


def acquire_controller_lock(handle, wait_for_prior, status_path, summary):
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not wait_for_prior:
                raise
            write(status_path, {**summary, "status": "waiting_for_prior_controller_natural_exit",
                                "pid": os.getpid(), "heartbeat_at": now()})
            time.sleep(15)


def run(wait_for_prior=False):
    jobs, scope = plan()
    queue = HERE / "artifacts/queue.json"
    own_lock_path = HERE / "artifacts/single_seed_controller.lock"
    handoff_path = HERE / "artifacts/acd_only_handoff.json"
    summary = {"total": len(jobs), "allowed_seeds": scope["allowed_seeds"],
               "allowed_arms": scope["allowed_arms"], "deferred_jobs": scope["deferred_jobs"],
               "cancelled_jobs": scope["cancelled_jobs"], "process_termination": False}
    with own_lock_path.open("a") as own_lock:
        acquire_controller_lock(own_lock, wait_for_prior, handoff_path, summary)
        write(handoff_path, {**summary, "status": "controller_lock_acquired", "pid": os.getpid(), "heartbeat_at": now()})
        with (ROOT / "queue_logs/cifar100_acd_rerun_gpu0.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    current = read(queue) if queue.exists() else {}
                    current.update(total=len(jobs), allowed_seeds=scope["allowed_seeds"],
                                   allowed_arms=scope["allowed_arms"], deferred_arms=scope["deferred_arms"], deferred_jobs=scope["deferred_jobs"],
                                   cancelled_seeds=scope["cancelled_seeds"], cancelled_jobs=scope["cancelled_jobs"],
                                   scheduling_status="acd_only_handoff_pending_current_job_finishes", heartbeat_at=now())
                    write(queue, current)
                    write(HERE / "artifacts/single_seed_controller.json", {
                        "status": "waiting_for_current_job", "pid": os.getpid(), "total": len(jobs), "allowed_arms": scope["allowed_arms"], "heartbeat_at": now()})
                    time.sleep(15)
            marker = HERE / "STOP"
            if marker.exists():
                if marker.read_text().strip() != scope["handoff_marker"]:
                    write(queue, {"status": "stopped", "reason": "Independent STOP retained", "total": len(jobs), "heartbeat_at": now()})
                    return
                marker.unlink()
            for ordinal, job in enumerate(jobs, 1):
                directory = HERE / "runs" / f"{job['pair']}__s{job['seed']}__{job['arm']}"
                if (directory / "final.json").exists():
                    final = read(directory / "final.json")
                    assert final["status"] == "completed" and final["epoch"] == 240
                    assert all(final[key] == value for key, value in job.items())
                    continue
                status_path = directory / "status.json"
                if status_path.exists():
                    status = read(status_path)
                    # An old parent can exit on failure. Never turn a failed/active job into a retry.
                    write(queue, {"status": "needs_review", "job": job, "previous_status": status["status"], "total": len(jobs), "heartbeat_at": now()})
                    raise RuntimeError("Prior job lacks final.json: manual diagnosis required before scheduling")
                while True:
                    if marker.exists():
                        write(queue, {"status": "stopped", "total": len(jobs), "heartbeat_at": now()})
                        return
                    occupied = subprocess.check_output(["nvidia-smi", "-i", str(scope["gpu"]),
                        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if not occupied: break
                    write(queue, {"status": "waiting_gpu0", "job": job, "total": len(jobs), "heartbeat_at": now()})
                    time.sleep(30)
                directory.mkdir(parents=True, exist_ok=True)
                write(queue, {"status": "running", "job": job, "ordinal": ordinal, "total": len(jobs),
                              "allowed_seeds": scope["allowed_seeds"], "allowed_arms": scope["allowed_arms"],
                              "cancelled_seeds": scope["cancelled_seeds"], "cancelled_jobs": scope["cancelled_jobs"],
                              "deferred_arms": scope["deferred_arms"], "deferred_jobs": scope["deferred_jobs"], "heartbeat_at": now()})
                write(HERE / "artifacts/single_seed_controller.json", {"status": "running", "pid": os.getpid(), "job": job, "heartbeat_at": now()})
                with (directory / "job.log").open("a") as log:
                    result = subprocess.run([sys.executable, "-u", str(HERE / "run.py"), "train",
                        "--pair", job["pair"], "--seed", str(job["seed"]), "--arm", job["arm"]], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write(queue, {"status": "failed", "job": job, "total": len(jobs), "returncode": result.returncode, "heartbeat_at": now()})
                    raise RuntimeError("Training failed; no automatic retry")
            write(queue, {"status": "completed", "total": len(jobs), "allowed_seeds": scope["allowed_seeds"],
                          "allowed_arms": scope["allowed_arms"], "finished_at": now()})
            write(HERE / "artifacts/single_seed_controller.json", {"status": "completed", "pid": os.getpid(), "finished_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wait-for-prior-controller", action="store_true",
                        help="Wait for the old scheduler to exit naturally; never terminate it.")
    args = parser.parse_args()
    if args.check:
        jobs, scope = plan()
        print(json.dumps({"jobs": len(jobs), "allowed_seeds": scope["allowed_seeds"], "allowed_arms": scope["allowed_arms"],
                          "cancelled_jobs": scope["cancelled_jobs"], "deferred_jobs": scope["deferred_jobs"]}))
    else:
        run(wait_for_prior=args.wait_for_prior_controller)
