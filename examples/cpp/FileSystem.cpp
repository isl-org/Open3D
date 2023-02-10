// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > FileSystem ls [dir]");
    utility::LogInfo("    > FileSystem mkdir [dir]");
    utility::LogInfo("    > FileSystem rmdir [dir]");
    utility::LogInfo("    > FileSystem rmfile [file]");
    utility::LogInfo("    > FileSystem fileexists [file]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;
    if (!(argc == 2 || argc == 3) ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string directory, function;
    function = std::string(argv[1]);
    if (argc == 2) {
        directory = ".";
    } else {
        directory = std::string(argv[2]);
    }

    if (function == "ls") {
        std::vector<std::string> filenames;
        ListFilesInDirectory(directory, filenames);

        for (const auto &filename : filenames) {
            std::cout << filename << std::endl;
            std::cout << "parent dir name is : "
                      << GetFileParentDirectory(filename) << std::endl;
            std::cout << "file name only is : "
                      << GetFileNameWithoutDirectory(filename) << std::endl;
            std::cout << "extension name is : "
                      << GetFileExtensionInLowerCase(filename) << std::endl;
            std::cout << "file name without extension is : "
                      << GetFileNameWithoutExtension(filename) << std::endl;
            std::cout << std::endl;
        }
    } else if (function == "mkdir") {
        bool success = MakeDirectoryHierarchy(directory);
        std::cout << "mkdir " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "rmdir") {
        bool success = DeleteDirectory(directory);
        std::cout << "rmdir " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "rmfile") {
        bool success = RemoveFile(directory);
        std::cout << "rmfile " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "fileexists") {
        bool success = FileExists(directory);
        std::cout << "fileexists " << (success ? "yes" : "no") << std::endl;
    }
    return 1;
}
