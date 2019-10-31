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

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage :");
    utility::LogInfo("    > ProgramOptions [--help] [--switch] [--int i] [--double d] [--string str] [--vector (x,y,z,...)]");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    utility::LogInfo("Switch is {}.",
                     utility::ProgramOptionExists(argc, argv, "--switch")
                             ? "ON"
                             : "OFF");
    utility::LogInfo("Int is {:d}",
                     utility::GetProgramOptionAsInt(argc, argv, "--int"));
    utility::LogInfo("Double is {:.10f}",
                     utility::GetProgramOptionAsDouble(argc, argv, "--double"));
    utility::LogInfo("String is {}",
                     utility::GetProgramOptionAsString(argc, argv, "--string"));
    std::vector<std::string> strs;
    utility::SplitString(
            strs, utility::GetProgramOptionAsString(argc, argv, "--string"),
            ",.", true);
    for (auto &str : strs) {
        utility::LogInfo("\tSubstring : {}", str);
    }
    Eigen::VectorXd vec =
            utility::GetProgramOptionAsEigenVectorXd(argc, argv, "--vector");
    utility::LogInfo("Vector is (");
    for (auto i = 0; i < vec.size(); i++) {
        if (i == 0) {
            utility::LogInfo("{:.2f}", vec(i));
        } else {
            utility::LogInfo("{:.2f}", vec(i));
        }
    }
    utility::LogInfo(")");
    return 0;
}
