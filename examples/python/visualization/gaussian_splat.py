# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# To run this example, download some sample Gaussian Splats. You can use these
# samples to get started:
# curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.ply
# curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.splat

import sys

import open3d as o3d


def print_usage() -> None:
    print("Visualize Gaussian Splat from PLY or SPLAT file.")
    print(
        "Usage: gaussian_splat.py <filename.[ply|splat]> [occ_cull] [sh_degree] [min_alpha] [antialias]"
    )
    print("  occ_cull:   0 or 1 (default 0); enables occlusion culling")
    print("  sh_degree:  integer 0..2 (default 2)")
    print("  min_alpha:  float 0..1 (default 0)")
    print(
        "  antialias:  0 or 1 (default 0); enables density compensation to correct small-splat over-brightening"
    )


def _parse_optional_int(value: str, default: int, name: str) -> int:
    try:
        return int(value)
    except ValueError:
        print(
            f"[Open3D][warning] Invalid {name} '{value}', using default {default}."
        )
        return default


def _parse_optional_float(value: str, default: float, name: str) -> float:
    try:
        return float(value)
    except ValueError:
        print(
            f"[Open3D][warning] Invalid {name} '{value}', using default {default}."
        )
        return default


def main() -> int:
    if len(sys.argv) < 2 or any(a in ("-h", "--help") for a in sys.argv[1:]):
        print_usage()
        return 1

    filename = sys.argv[1]

    gsplat = o3d.t.io.read_point_cloud(filename)
    if gsplat.is_empty():
        print(f"[Open3D][warning] Failed to read file {filename}")
        return 1

    print(gsplat)
    # Parse optional args: occ_cull, sh_degree, min_alpha, antialias
    occ_cull = True
    sh_degree = 2
    min_alpha = 0.0
    antialias = False

    if len(sys.argv) >= 3:
        occ_cull = _parse_optional_int(sys.argv[2], occ_cull, "occ_cull") != 0
    if len(sys.argv) >= 4:
        sh_degree = _parse_optional_int(sys.argv[3], sh_degree, "sh_degree")
    if len(sys.argv) >= 4:
        min_alpha = _parse_optional_float(sys.argv[3], min_alpha, "min_alpha")
    if len(sys.argv) >= 5:
        antialias = _parse_optional_int(sys.argv[4], 0, "antialias") != 0

    # Clamp values to sane ranges.
    sh_degree = max(0, min(2, sh_degree))
    min_alpha = max(0.0, min(1.0, min_alpha))

    print(
        f"[Open3D][info] Using occ_cull={occ_cull} sh_degree={sh_degree} min_alpha={min_alpha} antialias={antialias}"
    )

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "gaussianSplat"
    mat.gaussian_splat_sh_degree = sh_degree
    mat.gaussian_splat_min_alpha = min_alpha
    mat.gaussian_splat_antialias = antialias
    mat.gaussian_splat_occlusion_cull = occ_cull

    o3d.visualization.draw([
        {
            "name": filename,
            "geometry": gsplat,
            "material": mat,
        },
    ],
                           title="Gaussian Splat",
                           width=1024,
                           height=768,
                           show_ui=True,
                           near_plane=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
