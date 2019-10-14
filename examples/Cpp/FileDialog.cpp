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

#include <tinyfiledialogs/tinyfiledialogs.h>

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    utility::LogInfo("Usage :");
    utility::LogInfo("    > FileDialog [save|load]");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    if (argc == 1) {
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
