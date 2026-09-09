"""
Builds the lightweight "Colab bundle" used to reproduce the paper's detection
results from the released weights, without downloading the full 18 GB dataset.

Rationale
---------
`python src/main.py --config=<CFG> --use_weights=True` only iterates over the
*test* split: training and validation loaders are constructed but never read.
Moreover, the inductive masking of Exp0-Exp3 is applied to the *training* split
only (see `loaders_constructor` in `src/datasets.py`), so the compiled test
snapshots are byte-identical across the four experiments. The bundle therefore
ships a single copy of the compiled test snapshots per dataset and exposes the
per-experiment folder names as symlinks.

Usage
-----
Run this once, on a machine that holds the full uncompressed dataset and where
the datasets have already been compiled (see the "Compile datasets" section of
the README):

    python tools/make_colab_bundle.py --dataset LANL --out ./bundles
    python tools/make_colab_bundle.py --dataset OPTC --out ./bundles

Then upload the produced archives and put their URLs in the Colab notebook.
"""

import argparse
import gzip
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import ROOT, LANL_COMPILED_TEST_PATH, TEST_PATH  # noqa: E402

TEST_PATHS = {"LANL": LANL_COMPILED_TEST_PATH, "OPTC": TEST_PATH}
DEFAULT_EXPERIMENTS = ["exp0", "exp1", "exp2", "exp3"]


def shrink_snapshot(src: Path, dst: Path) -> None:
    """
    Rewrite one compiled snapshot as a gzip-compressed pickle with narrower dtypes.

    This is lossless with respect to what the model consumes: the loader casts the
    edge index to int64, the edge features to float32 and the labels to int at load
    time, so storing them as int32/float32/int8 changes nothing downstream. The
    loader detects the gzip magic bytes and decompresses transparently.
    """
    with open(src, "rb") as f:
        edge_index, edge_feats, labels, *rest = pickle.load(f)

    edge_index = np.asarray(edge_index).astype(np.int32, copy=False)
    edge_feats = np.asarray(edge_feats).astype(np.float32, copy=False)
    labels = np.asarray(labels).astype(np.int8, copy=False)

    with gzip.open(dst, "wb", compresslevel=6) as f:
        pickle.dump((edge_index, edge_feats, labels, *rest), f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def shrink_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.pkl")):
        shrink_snapshot(f, dst / f.name)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(path: Path) -> str:
    files = sorted(p for p in path.glob("*.pkl"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(digest(f).encode())
    return h.hexdigest()


def human(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["LANL", "OPTC"], required=True)
    parser.add_argument("--out", default="bundles", help="Output directory for the archive")
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Store the snapshots verbatim instead of shrinking them (larger archive)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment suffixes to expose (default: exp0 exp1 exp2 exp3)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    test_root = Path(TEST_PATHS[dataset])
    print(f"Dataset root:      {ROOT}")
    print(f"Compiled test dir: {test_root}")

    available = {}
    for exp in args.experiments:
        name = f"{dataset}_{exp}"
        d = test_root / name
        if d.is_dir() and any(d.glob("*.pkl")):
            available[name] = d
        else:
            print(f"  [skip] {name}: not compiled at {d}")

    if not available:
        sys.exit(
            f"No compiled test snapshots found under {test_root}.\n"
            f"Compile them first, e.g.:\n"
            f"  python src/datasets.py --dataset={dataset} "
            f"--dataset_name={dataset}_exp0 --inductive_experiment=Exp0"
        )

    # The test split does not depend on the inductive experiment, so identical
    # folders are stored once and symlinked.
    print("\nHashing compiled test splits (this checks they are indeed identical)...")
    digests = {name: dir_digest(d) for name, d in available.items()}
    for name, dg in digests.items():
        print(f"  {name}: {len(list(available[name].glob('*.pkl'))):4d} snapshots  sha256={dg[:16]}...")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / dataset / "compiled" / "test"
        stage.mkdir(parents=True)

        canonical = {}  # digest -> folder name kept as a real copy
        for name, d in available.items():
            dg = digests[name]
            if dg in canonical:
                (stage / name).symlink_to(canonical[dg], target_is_directory=True)
                print(f"  {name} -> symlink to {canonical[dg]}")
            elif args.compress:
                shrink_dir(d, stage / name)
                canonical[dg] = name
                print(f"  {name} -> shrunk {human(dir_size(d))} "
                      f"-> {human(dir_size(stage / name))}")
            else:
                shutil.copytree(d, stage / name)
                canonical[dg] = name
                print(f"  {name} -> copied ({human(dir_size(d))})")

        archive = out_dir / f"lares_colab_bundle_{dataset}.tar.gz"
        print(f"\nCreating {archive} ...")
        subprocess.check_call(
            ["tar", "-czf", str(archive), "-C", tmp, dataset]
        )

    print(f"\nDone: {archive}  ({human(archive.stat().st_size)})")
    print(f"sha256: {digest(archive)}")
    print(
        "\nUpload this archive and set the matching URL in "
        "colab/LARES_artifact.ipynb (BUNDLE_URLS)."
    )


if __name__ == "__main__":
    main()
