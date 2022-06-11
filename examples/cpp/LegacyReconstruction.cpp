// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "LegacyReconstructionUtil.h"
#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > LegacyOfflineReconstruction [options]");
    utility::LogInfo("      Given an RGBD image sequence, perform the following steps:");
    utility::LogInfo("      1. Make fragments from the RGBD image sequence.");
    utility::LogInfo("      2. Register multiple fragments.");
    utility::LogInfo("      3. Refine rough registration.");
    utility::LogInfo("      4. Integrate the whole RGBD sequence to make final mesh or point clouds.");
    utility::LogInfo("      5. (Optional) Run slac optimization for fragments.");
    utility::LogInfo("      6. (Optional) Run slac optimization for fragments.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --color_folder_path");
    utility::LogInfo("    --depth_folder_path");
    utility::LogInfo("    --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("    --voxel_size [=0.0058 (m)]");
    utility::LogInfo("    --depth_scale [=1000.0]");
    utility::LogInfo("    --max_depth [=3.0]");
    utility::LogInfo("    --pointcloud [file path to save the extracted pointcloud]");
    utility::LogInfo("    --mesh [file path to save the extracted mesh]");
    utility::LogInfo("Description:");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;
    
}