# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
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
    /Users/ssheorey/Downloads/point_cloud_29999_only_table.ply,1,0,0,0,0,0,0
    /Users/ssheorey/Downloads/vase-f1992_13_2-150k-4096.glb,0.5,180,0,0,-0.2,0.2,1.0
    /Users/ssheorey/Downloads/Lycaste_virginalis-150k-4096_std.glb,0.75,180,0,0,0,0.425,0.8
    /Users/ssheorey/Downloads/nike.splat,0.075,-15,0,0,0,0.47,1.0
    #/Users/ssheorey/Downloads/garden_30K.ply,1,0,0,0,0,0,0
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d


def _parse_floats(row: Dict[str, str], keys: List[str]) -> Tuple[float, ...]:
    return tuple(float(row[k].strip()) for k in keys)


def _euler_deg_xyz_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    return o3d.geometry.get_rotation_matrix_from_xyz(
        np.deg2rad(np.array([rx_deg, ry_deg, rz_deg], dtype=np.float64)))


def _apply_srt_legacy(
    geometry: o3d.geometry.Geometry,
    scale: float,
    R: np.ndarray,
    translation: np.ndarray,
) -> None:
    center = np.zeros(3, dtype=np.float64)
    geometry.scale(scale, center)
    geometry.rotate(R, center)
    geometry.translate(translation)


def _apply_srt_tensor_point_cloud(
    pcd: o3d.t.geometry.PointCloud,
    scale: float,
    R: np.ndarray,
    translation: np.ndarray,
) -> None:
    if pcd.is_empty():
        return
    dtype = pcd.point["positions"].dtype
    dev = pcd.device
    zero = o3d.core.Tensor([0.0, 0.0, 0.0], dtype=dtype, device=dev)
    R_t = o3d.core.Tensor(R, dtype=dtype, device=dev)
    t_np = np.asarray(translation, dtype=np.float64)
    t_t = o3d.core.Tensor(t_np, dtype=dtype, device=dev)
    pcd.scale(float(scale), center=zero)
    pcd.rotate(R_t, center=zero)
    pcd.translate(t_t, relative=True)


def _load_and_transform_row(
    path: Path,
    scale: float,
    R: np.ndarray,
    tvec: np.ndarray,
):
    """Load geometry from *path* and return a list of draw() dicts (one entry)."""
    gtype = o3d.io.read_file_geometry_type(str(path))
    if gtype & o3d.io.CONTAINS_GAUSSIAN_SPLATS:
        t_pcd = o3d.t.io.read_point_cloud(str(path))
        if t_pcd.is_empty():
            print(
                f"[Open3D][warning] Empty tensor point cloud after read: {path!s}",
                file=sys.stderr,
            )
            return None
        _apply_srt_tensor_point_cloud(t_pcd, scale, R, tvec)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "gaussianSplat"
        return [{"name": path.stem, "geometry": t_pcd, "material": mat}]

    if gtype & o3d.io.CONTAINS_TRIANGLES:
        model = o3d.io.read_triangle_model(str(path))
        for mesh_info in model.meshes:
            _apply_srt_legacy(mesh_info.mesh, scale, R, tvec)
        return [{"name": path.stem, "geometry": model}]

    if gtype & o3d.io.CONTAINS_POINTS:
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_points():
            print(
                f"[Open3D][warning] No points in point cloud: {path!s}",
                file=sys.stderr,
            )
            return None
        _apply_srt_legacy(pcd, scale, R, tvec)
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
            rows.append({"_line": i, **row})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw multiple 3D assets with transforms from a CSV manifest."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to a CSV with columns: filename, scale, "
        "rx_deg, ry_deg, rz_deg, tx, ty, tz",
    )
    args = parser.parse_args()
    csv_path = args.csv.resolve()
    base_dir = csv_path.parent
    try:
        manifest = _read_manifest(csv_path)
    except ValueError as e:
        print(f"[Open3D][error] {e}", file=sys.stderr)
        return 1

    draw_list: list[dict] = []
    for row in manifest:
        if row["filename"].strip().startswith('#'):
            continue
        relp = Path(row["filename"].strip())
        path = relp if relp.is_absolute() else (base_dir / relp)
        if not path.is_file():
            print(
                f"[Open3D][error] line {row['_line']}: file not found: {path}",
                file=sys.stderr,
            )
            return 1

        scale, rx, ry, rz, tx, ty, tz = _parse_floats(
            row, ["scale", "rx_deg", "ry_deg", "rz_deg", "tx", "ty", "tz"]
        )
        R = _euler_deg_xyz_to_R(rx, ry, rz)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        out = _load_and_transform_row(path, scale, R, tvec)
        if not out:
            continue
        object_name = f"l{row['_line']}_{path.stem}"
        for d in out:
            d["name"] = object_name
        draw_list.extend(out)

    if not draw_list:
        print("No entries to display.", file=sys.stderr)
        return 1
    o3d.visualization.draw(draw_list, show_ui=True, title=csv_path.name, 
        show_skybox=False, bg_color=(0.0, 0.0, 0.0, 1.0), ibl_intensity=100000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
