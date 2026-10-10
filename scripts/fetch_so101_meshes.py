"""Download the SO-101 follower meshes at the pinned upstream commit and verify checksums."""

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{COMMIT}/Simulation/SO101"


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "third_party" / "so101"
    manifest = (model_dir / "assets.sha256").read_text().splitlines()
    (model_dir / "assets").mkdir(exist_ok=True)
    failures = []
    for digest, relpath in (line.split() for line in manifest if line.strip()):
        name = Path(relpath).name
        path = model_dir / relpath
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            with urllib.request.urlopen(f"{BASE}/assets/{name}", timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                failures.append(name)
                continue
            path.write_bytes(data)
        print(f"OK {name}")
    if failures:
        sys.exit(f"Checksum mismatch, not written: {', '.join(failures)}")
    print(f"Meshes ready in {model_dir / 'assets'}")


if __name__ == "__main__":
    main()
