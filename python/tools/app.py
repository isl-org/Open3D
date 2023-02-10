# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys

import open3d as o3d


def main():
    args = [os.path.abspath(__file__)]
    if len(sys.argv) > 1:
        args.append(sys.argv[1])
    o3d.visualization.app.run_viewer(args)


if __name__ == "__main__":
    main()
