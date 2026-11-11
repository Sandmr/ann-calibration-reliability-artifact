"""Create a SHA-256 manifest for frozen evidence or template artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or ".venv" in path.parts:
            continue
        if path.resolve() == args.output.resolve():
            continue
        if any(part in {"__pycache__", "stage4_pre_patch"} for part in path.parts):
            continue
        records.append({"path": path.relative_to(args.root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    args.output.write_text(json.dumps({"root": str(args.root), "files": records}, indent=2), encoding="utf-8")
    print(f"manifest_files={len(records)}")


if __name__ == "__main__":
    main()
