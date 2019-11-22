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

#include <cstdio>
#include <cstdlib>
#include <string>

void PrintHelp() {
    printf("Usage:\n");
    printf("    > EncodeShader target_file\n");
    printf("      Generate a header file with license information.\n");
    printf("\n");
    printf("    > EncodeShader target_file source_shader_file\n");
    printf("      Append the shader file to the target header file.\n");
}

void WriteFileHeader(FILE *file) {
    fprintf(file, "// Automatically generated header file for shader.\n");
    fprintf(file, "// See LICENSE.txt for full license statement.\n");
    fprintf(file, "\n");
    fprintf(file, "#pragma once\n");
    fprintf(file, "\n");
}

std::string MakeString(const std::string &line) {
    std::string str;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == '"') {
            str += "\\\"";
        } else if (c == '\\') {
            str += "\\\\";
        } else {
            str += c;
        }
    }

    size_t r_pos = str.find('\r');
    if (r_pos != std::string::npos) {
        str = str.substr(0, r_pos);
    }

    size_t n_pos = str.find('\n');
    if (n_pos != std::string::npos) {
        str = str.substr(0, n_pos);
    }

    return str;
}

void WriteStringHeader(const std::string &string_name, FILE *file) {
    fprintf(file, "namespace open3d {\n\n");
    fprintf(file, "namespace visualization {\n\n");
    fprintf(file, "namespace glsl {\n\n");
    fprintf(file, "const char * const %s = \n", string_name.c_str());
}

void WriteStringFooter(FILE *file) {
    fprintf(file, ";\n");
    fprintf(file, "\n}  // namespace open3d::glsl\n");
    fprintf(file, "\n}  // namespace open3d::visualization\n");
    fprintf(file, "\n}  // namespace open3d\n");
    fprintf(file, "\n");
}

int main(int argc, char **args) {
    if (argc <= 1) {
        PrintHelp();
        return 0;
    }

    if (argc == 2) {
        FILE *file_out = fopen(args[1], "w");
        if (file_out == 0) {
            printf("Cannot open file %s\n", args[1]);
        }
        WriteFileHeader(file_out);
        fclose(file_out);
    }

    if (argc >= 3) {
        FILE *file_out = fopen(args[1], "a");
        if (file_out == 0) {
            printf("Cannot open file %s\n", args[1]);
        }

        FILE *file_in = fopen(args[2], "r");
        if (file_in == 0) {
            printf("Cannot open file %s\n", args[2]);
        }

        const std::string file_in_name(args[2]);
        size_t dot_pos = file_in_name.find_last_of(".");
        if (dot_pos == std::string::npos || dot_pos == 0) {
            printf("Illegal file extension.");
            return 0;
        }
        std::string string_name = file_in_name.substr(0, dot_pos);
        const size_t last_slash_idx = string_name.find_last_of("\\/");
        if (last_slash_idx != std::string::npos) {
            string_name = string_name.substr(last_slash_idx + 1);
        }

        fprintf(file_out, "// clang-format off\n");
        WriteStringHeader(string_name, file_out);
        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), file_in)) {
            std::string line = MakeString(std::string(buffer));
            fprintf(file_out, "\"%s\\n\"\n", line.c_str());
        }
        WriteStringFooter(file_out);
        fprintf(file_out, "// clang-format on\n");

        fclose(file_in);
        fclose(file_out);
    }

    return 0;
}
