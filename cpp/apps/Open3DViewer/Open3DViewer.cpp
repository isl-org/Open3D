// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>

#include "open3d/Open3D.h"
#include "open3d/visualization/app/Viewer.h"

using namespace open3d::visualization::app;

#if __APPLE__
// Open3DViewer_mac.mm
#else
int main(int argc, const char *argv[]) {
    RunViewer(argc, argv);
    return 0;
}
#endif  // __APPLE__
