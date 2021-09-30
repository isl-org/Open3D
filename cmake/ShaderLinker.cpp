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

#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("ShaderLinker:\n"
               "Unexpected number of arguments %d\n"
               "Expected usage \"ShaderLinker <output> <input1> [<input2> "
               "...]\"\n",
               argc);
        return 1;
    }

    FILE *file_out = fopen(argv[1], "w");
    if (file_out == 0) {
        printf("ShaderLinker:\n"
               "Cannot open file %s\n",
               argv[1]);
        return 1;
    }

    fprintf(file_out, "// ----------------------------------------------------------------------------\n");
    fprintf(file_out, "// -                        Open3D: www.open3d.org                            -\n");
    fprintf(file_out, "// ----------------------------------------------------------------------------\n");
    fprintf(file_out, "// The MIT License (MIT)\n");
    fprintf(file_out, "//\n");
    fprintf(file_out, "// Copyright (c) 2018-2021 www.open3d.org\n");
    fprintf(file_out, "//\n");
    fprintf(file_out, "// Permission is hereby granted, free of charge, to any person obtaining a copy\n");
    fprintf(file_out, "// of this software and associated documentation files (the \"Software\"), to deal\n");
    fprintf(file_out, "// in the Software without restriction, including without limitation the rights\n");
    fprintf(file_out, "// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n");
    fprintf(file_out, "// copies of the Software, and to permit persons to whom the Software is\n");
    fprintf(file_out, "// furnished to do so, subject to the following conditions:\n");
    fprintf(file_out, "//\n");
    fprintf(file_out, "// The above copyright notice and this permission notice shall be included in\n");
    fprintf(file_out, "// all copies or substantial portions of the Software.\n");
    fprintf(file_out, "//\n");
    fprintf(file_out, "// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n");
    fprintf(file_out, "// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n");
    fprintf(file_out, "// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n");
    fprintf(file_out, "// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n");
    fprintf(file_out, "// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\n");
    fprintf(file_out, "// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS\n");
    fprintf(file_out, "// IN THE SOFTWARE.\n");
    fprintf(file_out, "// ----------------------------------------------------------------------------\n");
    
    fprintf(file_out, "// Automatically generated header file for shader.\n");
    fprintf(file_out, "\n");
    fprintf(file_out, "#pragma once\n");
    fprintf(file_out, "\n");
    fprintf(file_out, "namespace open3d {\n");
    fprintf(file_out, "namespace visualization {\n");
    fprintf(file_out, "namespace glsl {\n");
    fprintf(file_out, "// clang-format off\n");
    fprintf(file_out, "\n");

    char buffer[1024];
    for (int i = 2; i < argc; ++i) {
        FILE *file_in = fopen(argv[i], "r");
        if (file_in == nullptr) {
            printf("ShaderLinker:\n"
                   "Cannot open file %s\n",
                   argv[i]);
            continue;
        }

        // Skip first 3 comment lines which only contain license information
        for (int i = 0; i < 3; ++i) {
            auto ignored = fgets(buffer, sizeof(buffer), file_in);
            (void)ignored;
        }

        // Copy content into "linked" file
        while (fgets(buffer, sizeof(buffer), file_in)) {
            fprintf(file_out, "%s", buffer);
        }
        fprintf(file_out, "\n");

        fclose(file_in);
    }

    fprintf(file_out, "// clang-format on\n");
    fprintf(file_out, "}  // namespace open3d::glsl\n");
    fprintf(file_out, "}  // namespace open3d::visualization\n");
    fprintf(file_out, "}  // namespace open3d\n");
    fprintf(file_out, "\n");

    fclose(file_out);

    return 0;
}
