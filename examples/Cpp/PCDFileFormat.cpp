// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;
    using namespace flann;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintOpen3DVersion();
        // clang-format off
        utility::LogInfo("Usage:");
        utility::LogInfo("    > PCDFileFormat [filename] [ascii|binary|compressed]");
        utility::LogInfo("    The program will :");
        utility::LogInfo("    1. load the pointcloud in [filename].");
        utility::LogInfo("    2. visualize the point cloud.");
        utility::LogInfo("    3. if a save method is specified, write the point cloud into data.pcd.");
        // clang-format on
        return 0;
    }

    auto cloud_ptr = io::CreatePointCloudFromFile(argv[1]);
    visualization::DrawGeometries({cloud_ptr}, "TestPCDFileFormat", 1920, 1080);

    if (argc >= 3) {
        std::string method(argv[2]);
        if (method == "ascii") {
            io::WritePointCloud("data.pcd", *cloud_ptr, true);
        } else if (method == "binary") {
            io::WritePointCloud("data.pcd", *cloud_ptr, false, false);
        } else if (method == "compressed") {
            io::WritePointCloud("data.pcd", *cloud_ptr, false, true);
        }
    }

    return 0;
}
