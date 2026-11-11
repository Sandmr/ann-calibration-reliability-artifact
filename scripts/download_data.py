"""Download the public ANN-Benchmarks HDF5 workloads with checksum checks."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

DATASETS = {
    "sift1m": ("https://ann-benchmarks.com/sift-128-euclidean.hdf5", "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984", "sift-128-euclidean.hdf5"),
    "glove25": ("https://ann-benchmarks.com/glove-25-angular.hdf5", "51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391", "glove-25-angular.hdf5"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--dataset", choices=["sift1m", "glove25", "all"], default="all")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, (url, expected, filename) in names.items():
        target = args.output / filename
        if not target.exists():
            print(f"downloading {name} from {url}")
            with urlopen(url) as response, target.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {target}: {actual} != {expected}")
        print(f"verified {target} sha256={actual}")

if __name__ == "__main__":
    main()
