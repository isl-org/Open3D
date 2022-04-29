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

#include <iostream>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    CameraPoseTrajectory [trajectory_file]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    data::MultiDownloadDataset dataset(
            "BedroomRGBDImages",
            {{"https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220301-data/bedroom01.zip"},
             {"https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220301-data/bedroom02.zip"},
             {"https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220301-data/bedroom03.zip"},
             {"https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220301-data/bedroom04.zip"},
             {"https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220301-data/bedroom05.zip"}},
            {"", "", "", "", ""}, false);

    std::cout << "Prefix: " << dataset.GetPrefix() << std::endl;
    std::cout << "Download Dir: " << dataset.GetExtractDir() << std::endl;
    std::cout << "Extract Dir: " << dataset.GetExtractDir() << std::endl;
    return 0;
}
