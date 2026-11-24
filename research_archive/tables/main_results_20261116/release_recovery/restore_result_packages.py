"""Restore manifest-listed result files, verifying archives and individual objects."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
GH = "/home/iaii2026/data/nxt/tools/github-cli/gh_2.104.0_linux_amd64/bin/gh"


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-head-study", action="store_true")
    args = parser.parse_args()
    inventory = json.loads((HERE / "release_inventory.json").read_text())
    env = dict(os.environ, GH_CONFIG_DIR="/tmp/kd-gh-laniusnan", GH_PROMPT_DISABLED="1")
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        env.pop(key, None)
    recovered, archives = [], []
    for release in inventory["releases"]:
        tag = release["tag_name"]
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", tag)
        metadata = HERE / "metadata" / str(int(release["id"]))
        manifest_path = next(metadata.glob("*_MODEL_MANIFEST.json"))
        manifest = json.loads(manifest_path.read_text())
        selected = [f for f in manifest["files"] if f["group"] == "results-and-provenance" or
                    (args.include_head_study and f["group"] == "head_transfer_20261021" and
                     f["role"] in ("metadata", "protocol"))]
        checksums = {line.split()[1].lstrip("*"): line.split()[0]
                     for line in next(metadata.glob("*_SHA256SUMS")).read_text().splitlines() if line.strip()}
        for name in sorted({f["archive"] for f in selected}):
            asset = next(a for a in release["assets"] if a["name"] == name)
            assert re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            download = HERE / "downloads" / str(int(release["id"])) / name
            download.parent.mkdir(parents=True, exist_ok=True)
            if not download.is_file():
                print(f"Downloading {tag}/{name} ({asset['size']} bytes)", flush=True)
                partial = download.with_name(download.name + ".partial")
                with partial.open("wb") as handle:
                    proc = subprocess.run([GH, "api", f"repos/laniusnan/KD/releases/assets/{asset['id']}",
                                           "-H", "Accept: application/octet-stream"],
                                          env=env, stdout=handle, stderr=subprocess.PIPE, timeout=600)
                if proc.returncode:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Asset download failed (exit {proc.returncode}): {name}")
                partial.rename(download)
            digest = sha256(download)
            assert download.stat().st_size == asset["size"] and digest == checksums[name]
            if asset.get("digest"):
                assert "sha256:" + digest == asset["digest"]
            rows = [f for f in selected if f["archive"] == name]
            by_hash = {}
            for row in rows:
                by_hash.setdefault(row["sha256"], []).append(row)
            found = set()
            with tarfile.open(download, "r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not re.fullmatch(r"objects/[0-9a-f]{64}", member.name):
                        raise RuntimeError(f"Unexpected archive member: {member.name}")
                    key = member.name.split("/")[1]
                    if key not in by_hash:
                        continue
                    if key in found:
                        raise RuntimeError("Duplicate content object in archive")
                    data = archive.extractfile(member).read()
                    assert hashlib.sha256(data).hexdigest() == key
                    for row in by_hash[key]:
                        assert len(data) == row["bytes"]
                        original = PurePosixPath(row["path"])
                        assert not original.is_absolute() and ".." not in original.parts
                        # Remove only the snapshot wrapper; preserve paths inside the study.
                        relative = PurePosixPath(row["path"].split("/snapshot/", 1)[-1])
                        destination = HERE / "restored" / tag / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        assert destination.resolve().is_relative_to((HERE / "restored" / tag).resolve())
                        if destination.exists():
                            assert sha256(destination) == key
                        else:
                            destination.write_bytes(data)
                        recovered.append({"tag": tag, "archive": name, "original_path": row["path"],
                                          "path": str(destination.relative_to(HERE)), "bytes": len(data),
                                          "sha256": key, "role": row["role"]})
                    found.add(key)
            assert found == set(by_hash)
            archives.append({"tag": tag, "asset_id": asset["id"], "name": name,
                             "bytes": asset["size"], "sha256": digest, "restored_paths": len(rows)})
            print(f"Verified and restored {len(rows)} paths from {tag}/{name}", flush=True)
    result = {"restored_at": datetime.now(timezone.utc).isoformat(), "archives": archives, "files": recovered,
              "validation": "All selected objects match manifest SHA256 and size; all archives match SHA256SUMS and available GitHub digests.",
              "execution": "Only bytes restored; no checkpoint loading, source execution, training or active-workspace overwrite."}
    (HERE / "restoration_receipt.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"restored_paths": len(recovered), "archives": len(archives)}), flush=True)


if __name__ == "__main__":
    main()
