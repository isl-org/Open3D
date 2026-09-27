#!/usr/bin/env python3
"""Download and safely unpack the TUM fr1/xyz RGB-D sequence."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
SEQUENCE = "rgbd_dataset_freiburg1_xyz"


def download(destination: Path) -> Path:
    """Stream the public TUM archive and keep an existing valid extraction."""
    destination.mkdir(parents=True, exist_ok=True)
    sequence_dir = destination / SEQUENCE
    if (sequence_dir / "groundtruth.txt").is_file():
        print(f"Dataset already ready: {sequence_dir}")
        return sequence_dir

    archive = destination / f"{SEQUENCE}.tgz"
    partial = archive.with_suffix(".tgz.part")
    if not archive.exists():
        print(f"Downloading {URL}\nThis sequence is about 0.5 GB; please wait...")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(archive)

    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    if not (sequence_dir / "groundtruth.txt").is_file():
        raise RuntimeError("Extraction finished but groundtruth.txt is missing")
    print(f"Ready: {sequence_dir}")
    return sequence_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    download(args.destination.resolve())
