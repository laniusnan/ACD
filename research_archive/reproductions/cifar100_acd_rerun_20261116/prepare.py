"""Restore verified CIFAR teachers and freeze public data and local source files."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECOVERY = ROOT / "paper_assets/acd_tables/main_results_20261116/release_recovery"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, str(path)
    else:
        path.write_bytes(content)


def main():
    config = json.loads((ROOT / "reproductions/head_transfer_20261021/study.json").read_text())
    policy = json.loads((HERE / "protocol.json").read_text())
    pairs = {key: config["pairs"][key] for key in policy["pairs"]}
    names = {p[role] for p in pairs.values() for role in ("teacher", "student")}
    models = {name: config["models"][name] for name in sorted(names)}
    for name in {m["module"] for m in models.values()}:
        once(HERE / "frozen/models" / (name + ".py"), (ROOT / "CIFAR-100/Models" / (name + ".py")).read_bytes())
    source = (ROOT / "CIFAR-100/Utils/train.py").read_text()
    tree = ast.parse(source)
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "AutoContrast")
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == "RandAugment")
    augmentation = "import random\nimport numpy as np\nfrom PIL import Image, ImageOps, ImageEnhance, ImageDraw\n\n"
    augmentation += "\n\n".join(ast.get_source_segment(source, n) for n in tree.body[start:end+1]) + "\n"
    once(HERE / "frozen/augmentation.py", augmentation.encode())
    models_path = HERE / "models.json"
    model_config = {"pairs": pairs, "models": models}
    if models_path.exists():
        assert json.loads(models_path.read_text()) == model_config
    else:
        once(models_path, (json.dumps(model_config, indent=2) + "\n").encode())
    inventory = json.loads((RECOVERY / "release_inventory.json").read_text())
    release = next(r for r in inventory["releases"] if r["tag_name"] == "model-backup-20261027")
    manifest = json.loads(next((RECOVERY / "metadata" / str(release["id"])).glob("*_MODEL_MANIFEST.json")).read_text())
    teachers = {p["teacher"] for p in pairs.values()}
    selected = {f["sha256"]: (name, f) for name in teachers for f in manifest["files"]
                if f["path"] == f"reproductions/head_transfer_20261021/frozen/teachers/best_model_weights_{name}.pth"}
    assert len(selected) == len(teachers)
    asset = next(a for a in release["assets"] if a["name"] == "teacher-models-000.tar.gz")
    archive_path = HERE / "assets" / asset["name"]
    archive_path.parent.mkdir(exist_ok=True)
    if not archive_path.exists():
        print("Downloading verified teacher archive", flush=True)
        env = dict(os.environ, GH_CONFIG_DIR="/tmp/kd-gh-laniusnan", GH_PROMPT_DISABLED="1")
        for key in ("GH_TOKEN", "GITHUB_TOKEN"):
            env.pop(key, None)
        partial = archive_path.with_suffix(".partial")
        with partial.open("wb") as handle:
            subprocess.run(["/home/iaii2026/data/nxt/tools/github-cli/gh_2.104.0_linux_amd64/bin/gh", "api",
                            f"repos/laniusnan/KD/releases/assets/{asset['id']}", "-H", "Accept: application/octet-stream"],
                           env=env, stdout=handle, check=True, timeout=600)
        partial.rename(archive_path)
    assert archive_path.stat().st_size == asset["size"] and "sha256:" + sha(archive_path) == asset["digest"]
    found = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("objects/")
            if key not in selected:
                continue
            assert member.isfile()
            name, entry = selected[key]
            content = archive.extractfile(member).read()
            assert len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == key
            once(HERE / "assets/teachers" / (name + ".pth"), content)
            found[name] = {"sha256": key, "release": release["tag_name"], "original_path": entry["path"]}
    assert set(found) == teachers
    from torchvision.datasets import CIFAR100
    print("Preparing official CIFAR-100 data", flush=True)
    train = CIFAR100(str(HERE / "data"), train=True, download=True)
    test = CIFAR100(str(HERE / "data"), train=False, download=True)
    assert len(train) == 50000 and len(test) == 10000
    result = {"teachers": found, "data": {n: sha(HERE / "data/cifar-100-python" / n) for n in ("train", "test", "meta")},
              "frozen_sources": {str(p.relative_to(HERE)): sha(p) for p in (HERE / "frozen").rglob("*.py")},
              "augmentation_source": {"path": "CIFAR-100/Utils/train.py", "sha256": sha(ROOT / "CIFAR-100/Utils/train.py")}}
    once(HERE / "artifacts/assets.json", (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({"teachers": len(found), "train": len(train), "test": len(test)}), flush=True)


if __name__ == "__main__":
    main()
