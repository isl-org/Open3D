// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <tinyfiledialogs/tinyfiledialogs.h>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > FileDialog [save|load]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);
    char const *pattern = "*.*";
    if (option == "load") {
        char const *str = tinyfd_openFileDialog("Find a file to load", "", 0,
                                                NULL, NULL, 1);
        utility::LogInfo("{}", str);
    } else if (option == "save") {
        char const *str = tinyfd_saveFileDialog("Find a file to save", "", 1,
                                                &pattern, NULL);
        utility::LogInfo("{}", str);
    }
    return 0;
}
