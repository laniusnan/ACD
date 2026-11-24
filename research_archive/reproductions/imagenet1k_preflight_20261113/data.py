"""Strict audits for an already authorized, extracted ImageNet-1K ImageFolder."""

import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


COUNTS = {"train": 1281167, "val": 50000}
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
CLASS_INDEX_SHA256 = "a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def transform(train):
    operations = [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
                  transforms.RandomHorizontalFlip()] if train else [
                      transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                      transforms.CenterCrop(224)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def class_names(index_path):
    index = json.loads(Path(index_path).read_text())
    if set(index) != {str(number) for number in range(1000)}:
        raise ValueError("Class index must contain exactly indices 0..999")
    names = [index[str(number)][0] for number in range(1000)]
    if len(set(names)) != 1000 or names != sorted(names):
        raise ValueError("Class order does not match lexicographically ordered ImageNet WNIDs")
    return names


def open_split(root, split, names):
    dataset = datasets.ImageFolder(Path(root) / split, transform=transform(split == "train"))
    if dataset.classes != names or len(dataset) != COUNTS[split]:
        raise ValueError(f"Not full ImageNet-1K {split}: classes={len(dataset.classes)}, images={len(dataset)}")
    if split == "val":
        counts = [0] * 1000
        for label in dataset.targets:
            counts[label] += 1
        if counts != [50] * 1000:
            raise ValueError("Validation requires 50 images per class with verified labels")
    return dataset


def stat_digest(dataset, root):
    digest = hashlib.sha256()
    for path, label in dataset.samples:
        info = Path(path).stat()
        digest.update(json.dumps([str(Path(path).relative_to(root)), label, info.st_size,
                                  info.st_mtime_ns]).encode())
    return digest.hexdigest()


def audit(root, index_path, output):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Authorized ImageNet-1K root is missing: {root}")
    if sha256_file(index_path) != CLASS_INDEX_SHA256:
        raise ValueError("Class index differs from the pinned official metadata")
    names = class_names(index_path)
    result = {"root": str(root), "class_index_sha256": sha256_file(index_path),
              "classes": names, "decoded_all": True, "content_hashed_all": True, "splits": {}}
    for split in COUNTS:
        dataset = open_split(root, split, names)
        before = stat_digest(dataset, root)
        manifest = hashlib.sha256()
        for number, (path, label) in enumerate(dataset.samples, 1):
            with Image.open(path) as image:
                image.convert("RGB").load()
            manifest.update(json.dumps([str(Path(path).relative_to(root)), label,
                                        sha256_file(path)]).encode())
            if number % 10000 == 0:
                print(json.dumps({"audit_split": split, "checked": number}), flush=True)
        after = stat_digest(dataset, root)
        if before != after:
            raise RuntimeError("Dataset changed during integrity audit")
        result["splits"][split] = {"count": len(dataset), "stat_sha256": after,
                                    "content_manifest_sha256": manifest.hexdigest()}
    save_json(output, result)
    return result


def audited_datasets(root, index_path, audit_path):
    evidence = json.loads(Path(audit_path).read_text())
    root = Path(root).resolve()
    if sha256_file(index_path) != CLASS_INDEX_SHA256:
        raise ValueError("Class index differs from the pinned official metadata")
    names = class_names(index_path)
    if evidence.get("root") != str(root) or evidence.get("class_index_sha256") != sha256_file(index_path):
        raise ValueError("Data audit path or class mapping mismatch")
    if not evidence.get("decoded_all") or not evidence.get("content_hashed_all"):
        raise ValueError("Full image decode and content integrity audit required")
    result = {}
    for split in COUNTS:
        dataset = open_split(root, split, names)
        if evidence["splits"][split]["stat_sha256"] != stat_digest(dataset, root):
            raise ValueError(f"Dataset {split} changed after audit")
        result[split] = dataset
    return result
