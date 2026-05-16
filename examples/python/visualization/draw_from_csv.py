# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""
Load a table of 3D assets and per-row transforms from a CSV file, then display
all geometries in one :func:`open3d.visualization.draw` window.

CSV format (header required)::

    filename,scale,rx_deg,ry_deg,rz_deg,tx,ty,tz

- ``filename``: path to a mesh, point cloud, or 3D Gaussian splat file. May be
  relative to the CSV file's directory.
- ``scale``: uniform scale factor.
- ``rx_deg``, ``ry_deg``, ``rz_deg``: Euler angles in degrees, XYZ order, applied
  about the origin.
- ``tx``, ``ty``, ``tz``: translation, applied after scale and rotation.

Transforms are applied in order: **scale** → **rotate** → **translate** (separate
operations, not a single 4x4 ``transform()`` on splats; for 3DGS, use
``t.geometry.PointCloud`` scale / rotate / translate, which update splat
attributes consistently).

Line-set-only files are skipped with a warning.

Example::

    filename,scale,rx_deg,ry_deg,rz_deg,tx,ty,tz
    mipnerf360_garden_crop_table.ply,1,0,0,0,0,0,0
    vase-f1992_13_2-150k-4096.glb,0.5,180,0,0,-0.2,0.2,1.0
    Lycaste_virginalis-150k-4096_std.glb,0.75,180,0,0,0,0.425,0.8
    nike.splat,0.075,-15,0,0,0,0.47,1.0
"""

import csv
import sys
from pathlib import Path
from typing import Tuple
import requests

import numpy as np
import open3d as o3d


def _download_example_assets():
    """Download example assets and create a manifest CSV with scale - rotation - translation to setup the example scene."""
    asset_urls = {
        "mipnerf360_garden_crop_table.ply":
            "https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.ply",
        "vase-f1992_13_2-150k-4096.glb":
            "https://3d-api.si.edu/content/document/3d_package:a05dc7c9-7b6f-43f8-8830-69fe98718e4f/resources/vase-f1992_13_2-150k-4096.glb",
        "Lycaste_virginalis-150k-4096_std.glb":
            "https://3d-api.si.edu/content/document/3d_package:5ff6e90a-4ddb-4eea-a69c-40970f85fbcb/resources/Lycaste_virginalis-150k-4096_std.glb",
        "nike.splat":
            "https://huggingface.co/cakewalk/splat-data/resolve/8fa962a5c7088fff3149a658718b89c5eb2c9c26/nike.splat?download=true",
    }
    dataset = o3d.data.Dataset("3dgs_example_assets")
    out_path = Path(dataset.download_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    CHUNK = 64 * 1024 * 1024  # 64MB
    for name, url in asset_urls.items():
        if (out_path / name).is_file():
            continue
        # Stream download in chunks to handle large files
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        print(f"Downloading {name}", end="", flush=True)
        with open(out_path / name, "wb") as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                print(".", end="", flush=True)
        print()
    with open(out_path / "3dgs_example_scene.csv", "w", newline="") as f:
        f.write("""filename,scale,rx_deg,ry_deg,rz_deg,tx,ty,tz
mipnerf360_garden_crop_table.ply,1,0,0,0,0,0,0
vase-f1992_13_2-150k-4096.glb,0.5,180,0,0,-0.2,0.2,1.0
Lycaste_virginalis-150k-4096_std.glb,0.75,180,0,0,0,0.425,0.8
nike.splat,0.075,-15,0,0,0,0.47,1.0
""")
    return out_path / "3dgs_example_scene.csv"


def _apply_srt_geometry(
    geometry,
    srt: Tuple[float, float, float, float, float, float, float],
) -> None:
    scale, rx, ry, rz, tx, ty, tz = srt
    rotation = np.array(
        o3d.geometry.get_rotation_matrix_from_xyz(
            np.deg2rad(np.array([rx, ry, rz], dtype=np.float64))))
    translation = np.array([tx, ty, tz], dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
    geometry.scale(scale, center)
    geometry.rotate(rotation, center)
    geometry.translate(translation)


def _load_and_transform_row(
    path: Path,
    srt: Tuple[float, float, float, float, float, float, float],
):
    """Load geometry from *path* and return a list of draw() dicts (one entry)."""
    if not path.is_file():
        print(f"[Open3D][warning] File not found: {path!s}. Skipping.",
              file=sys.stderr)
        return None
    gtype = o3d.io.read_file_geometry_type(str(path))
    if gtype & o3d.io.CONTAINS_GAUSSIAN_SPLATS:
        t_pcd = o3d.t.io.read_point_cloud(str(path))
        _apply_srt_geometry(t_pcd, srt)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "gaussianSplat"
        return [{"name": path.stem, "geometry": t_pcd, "material": mat}]

    if gtype & o3d.io.CONTAINS_TRIANGLES:
        model = o3d.io.read_triangle_model(str(path))
        for mesh_info in model.meshes:
            _apply_srt_geometry(mesh_info.mesh, srt)
        return [{"name": path.stem, "geometry": model}]

    if gtype & o3d.io.CONTAINS_POINTS:
        pcd = o3d.io.read_point_cloud(str(path))
        _apply_srt_geometry(pcd, srt)
        return [{"name": path.stem, "geometry": pcd}]

    if gtype & o3d.io.CONTAINS_LINES:
        print(
            f"[Open3D][info] Skipping line-set-only file: {path!s}",
            file=sys.stderr,
        )
        return None

    print(
        f"[Open3D][warning] Unknown geometry flags {gtype!r} for {path!s}. Skipping.",
        file=sys.stderr,
    )
    return None


def _read_manifest(csv_path: Path):
    rows: list[dict] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "filename", "scale", "rx_deg", "ry_deg", "rz_deg", "tx", "ty", "tz"
        ]
        for col in required:
            if col not in reader.fieldnames or reader.fieldnames is None:
                raise ValueError(
                    f"CSV must have columns: {', '.join(required)}; got: {reader.fieldnames}"
                )
        for i, row in enumerate(reader, start=2):
            if not any(str(v).strip() for v in row.values() if v is not None):
                continue
            if row["filename"].strip().startswith("#"):
                continue
            rows.append({"_line": i, **row})
    return rows


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if len(sys.argv) == 1:
        print(
            "No CSV provided, downloading example assets and using generated manifest."
        )
        csv_path = _download_example_assets()
    else:
        csv_path = Path(sys.argv[1]).resolve()
    base_dir = csv_path.parent
    try:
        manifest = _read_manifest(csv_path)
    except ValueError as e:
        print(f"[Open3D][error] {e}", file=sys.stderr)
        print(__doc__)
        return 1

    draw_list: list[dict] = []
    for row in manifest:
        relp = Path(row["filename"].strip())
        path = relp if relp.is_absolute() else (base_dir / relp)
        srt = tuple(
            float(row[x].strip())
            for x in ["scale", "rx_deg", "ry_deg", "rz_deg", "tx", "ty", "tz"])
        out = _load_and_transform_row(path, srt)
        if not out:
            continue
        object_name = f"l{row['_line']}_{path.stem}"
        for d in out:
            d["name"] = object_name
        draw_list.extend(out)

    if not draw_list:
        print("No entries to display.", file=sys.stderr)
        return 1
    o3d.visualization.draw(
        draw_list,
        show_ui=True,
        title=csv_path.stem,
        show_skybox=False,
        bg_color=(0.0, 0.0, 0.0, 1.0),
        ibl_intensity=100000,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
