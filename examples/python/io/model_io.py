# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Load a multi-material mesh model from USD (experimental).

Downloads Apple's sample biplane USDZ asset, reads it with
``open3d.io.read_triangle_model``, and visualizes embedded PBR materials.

See also: ``docs/jupyter/geometry/file_io.ipynb`` (USD import notes).
"""

import sys
import urllib.request
from pathlib import Path

import open3d as o3d

# Apple AR Quick Look sample (USDZ).
BIPLANE_USDZ_URL = (
    "https://developer.apple.com/augmented-reality/quick-look/models/"
    "biplane/toy_biplane_realistic.usdz")
BIPLANE_FILENAME = "toy_biplane_realistic.usdz"


def download_biplane_usdz() -> Path:
    """Download the sample USDZ if not already present under Open3D's data dir."""
    dataset = o3d.data.Dataset("toy_biplane_realistic")
    dest = Path(dataset.download_dir) / BIPLANE_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest

    print(f"Downloading {BIPLANE_USDZ_URL} ...")
    urllib.request.urlretrieve(BIPLANE_USDZ_URL, dest)
    return dest


def main():
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = download_biplane_usdz()

    print(f"Reading triangle mesh model: {model_path}")
    model = o3d.io.read_triangle_model(str(model_path))
    print(f"  sub-meshes: {len(model.meshes)}")
    print(f"  materials: {len(model.materials)}")

    o3d.visualization.draw(
        [{"name": model_path.stem, "geometry": model}],
        title="Open3D: " + model_path.name,
    )


if __name__ == "__main__":
    main()
