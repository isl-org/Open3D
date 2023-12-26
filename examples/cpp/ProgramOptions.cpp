// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ProgramOptions [-h|--help] [--switch] [--int i] [--double d] [--string str] [--vector (x,y,z,...)]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    if (argc == 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
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
    std::vector<std::string> strs = utility::SplitString(
            utility::GetProgramOptionAsString(argc, argv, "--string"), ",.",
            true);
    for (auto& str : strs) {
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
