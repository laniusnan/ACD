"""Read private release inventories and retrieve small result/manifest attachments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPOSITORY = "laniusnan/KD"
DEFAULT_GH = "/home/iaii2026/data/nxt/tools/github-cli/gh_2.104.0_linux_amd64/bin/gh"
MAX_METADATA_BYTES = 32 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gh", default=DEFAULT_GH)
    parser.add_argument("--config-dir", default="/tmp/kd-gh-laniusnan")
    args = parser.parse_args()
    env = dict(os.environ, GH_CONFIG_DIR=args.config_dir, GH_PROMPT_DISABLED="1")
    # Use the requested account configuration, not an unrelated environment token.
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        env.pop(key, None)

    def request(endpoint, output=None):
        cmd = [args.gh, "api", endpoint]
        if output is not None:
            cmd += ["-H", "Accept: application/octet-stream"]
            with output.open("wb") as handle:
                proc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.PIPE, timeout=120)
        else:
            proc = subprocess.run(cmd, env=env, capture_output=True, timeout=40)
        if proc.returncode:
            if output is not None:
                output.unlink(missing_ok=True)
            # Do not echo raw auth-helper output or credentials.
            raise RuntimeError(f"GitHub API failed: {endpoint} (exit {proc.returncode}); check account login/access.")
        return None if output is not None else json.loads(proc.stdout)

    def pages(endpoint):
        collected = []
        for page in range(1, 101):
            part = request(f"{endpoint}?per_page=100&page={page}")
            collected.extend(part)
            if len(part) < 100:
                return collected
        raise RuntimeError("Pagination limit reached; inventory would be incomplete.")

    login = request("user")["login"]
    if login.lower() != "laniusnan":
        raise RuntimeError(f"Account is {login}, expected laniusnan; no assets downloaded.")
    releases = pages(f"repos/{REPOSITORY}/releases")
    downloads = []
    for release in releases:
        assets = pages(f"repos/{REPOSITORY}/releases/{release['id']}/assets")
        release["assets"] = assets
        # Release IDs and asset IDs form local paths, never untrusted archive paths.
        directory = HERE / "metadata" / str(int(release["id"]))
        for asset in assets:
            name = asset["name"]
            looks_like_metadata = (Path(name).suffix.lower() in
                                   {".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml", ".sha256"}
                                   or name.upper() in {"SHA256SUMS", "SHA256SUMS.TXT", "CHECKSUMS"})
            if not looks_like_metadata or asset["size"] > MAX_METADATA_BYTES:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            filename = str(int(asset["id"])) + "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", name)
            destination = directory / filename
            expected = asset.get("digest")
            if destination.is_file():
                digest = "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
                if destination.stat().st_size != asset["size"] or (expected and digest != expected):
                    raise RuntimeError(f"Existing metadata file differs from release: {name}; inspect before replacing.")
            else:
                temporary = destination.with_name(destination.name + ".partial")
                request(f"repos/{REPOSITORY}/releases/assets/{asset['id']}", temporary)
                digest = "sha256:" + hashlib.sha256(temporary.read_bytes()).hexdigest()
                if temporary.stat().st_size != asset["size"] or (expected and digest != expected):
                    temporary.unlink()
                    raise RuntimeError(f"Asset size/digest mismatch: {name}")
                temporary.rename(destination)
            downloads.append({"release_tag": release["tag_name"], "asset_id": asset["id"],
                              "asset_name": name, "path": str(destination.relative_to(HERE)),
                              "bytes": asset["size"], "sha256": digest.split(":", 1)[1],
                              "github_digest_verified": bool(expected)})
    inventory = {"checked_at": datetime.now(timezone.utc).isoformat(), "repository": REPOSITORY,
                 "authenticated_login": login, "releases": releases, "downloaded_metadata": downloads}
    (HERE / "release_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n")
    lines = ["# GitHub Release 附件清单", "", f"检查时间：{inventory['checked_at']}；账号：{login}。", "",
             "| Release | 附件 | 大小（bytes） | 更新时间 |", "| --- | --- | ---: | --- |"]
    for release in releases:
        for asset in release["assets"]:
            name = asset["name"].replace("|", "\\|").replace("\n", " ")
            tag = release["tag_name"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {tag} | {name} | {asset['size']} | {asset['updated_at']} |")
    lines += ["", f"已读取 {len(releases)} 个 Release，下载 {len(downloads)} 个小型文本/清单附件。",
              "模型或压缩包未自动下载；应先核对清单中的研究名、指标、配置和端点类型，再选择所需包。", ""]
    (HERE / "release_inventory.md").write_text("\n".join(lines))
    print(json.dumps({"login": login, "releases": len(releases), "downloaded_metadata": len(downloads)}))


if __name__ == "__main__":
    main()
