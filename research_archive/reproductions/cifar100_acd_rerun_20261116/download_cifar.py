"""Download the same official CIFAR archive in verified HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
SIZE = 169001437
MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def main():
    root = HERE / "data"; root.mkdir(exist_ok=True)
    destination = root / "cifar-100-python.tar.gz"
    if destination.exists() and hashlib.md5(destination.read_bytes()).hexdigest() == MD5:
        print("Official archive already verified", flush=True); return
    if destination.exists():
        destination.rename(root / "cifar-100-python.single-connection.partial")
    parts = root / "download_parts"; parts.mkdir(exist_ok=True)

    def download(index):
        start, end = index * SIZE // 8, (index + 1) * SIZE // 8 - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1: return path
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
                    assert response.status == 206 and response.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
                    while chunk := response.read(64 * 1024): handle.write(chunk)
                assert path.stat().st_size == end - start + 1
                return path
            except Exception:
                if attempt == 2: raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, i) for i in range(8)]
        pending = futures
        while pending:
            _, pending = wait(pending, timeout=15)
            count = sum(p.stat().st_size for p in parts.iterdir())
            print(f"Official CIFAR download: {count}/{SIZE} bytes ({100*count/SIZE:.1f}%)", flush=True)
        paths = [f.result() for f in futures]
    temporary = root / "cifar-100-python.merged.partial"
    digest = hashlib.md5()
    with temporary.open("wb") as handle:
        for path in paths:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk); handle.write(chunk)
    assert temporary.stat().st_size == SIZE and digest.hexdigest() == MD5
    temporary.rename(destination)
    print("Complete: official CIFAR-100 MD5 verified", flush=True)


if __name__ == "__main__": main()
