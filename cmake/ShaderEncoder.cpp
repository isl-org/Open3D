// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <string>

std::string MakeString(const std::string &line) {
    std::string str;
    for (size_t i = 0; i < line.size(); ++i) {
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

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("ShaderEncoder:\n"
               "Unexpected number of arguments %d\n"
               "Expected usage \"ShaderEncoder <output> <input>\"\n",
               argc);
        return 1;
    }

    FILE *file_out = fopen(argv[1], "w");
    if (file_out == nullptr) {
        printf("ShaderEncoder:\n"
               "Cannot open file %s\n",
               argv[1]);
        return 1;
    }

    FILE *file_in = fopen(argv[2], "r");
    if (file_in == nullptr) {
        printf("ShaderEncoder:\n"
               "Cannot open file %s\n",
               argv[2]);

        fclose(file_out);
        return 1;
    }

    const std::string file_in_name(argv[2]);
    size_t dot_pos = file_in_name.find_last_of(".");
    if (dot_pos == std::string::npos || dot_pos == 0) {
        printf("ShaderEncoder:\n"
               "Illegal file extension");

        fclose(file_in);
        fclose(file_out);
        return 1;
    }

    std::string filename = file_in_name.substr(0, dot_pos);
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (last_slash_idx != std::string::npos) {
        filename = filename.substr(last_slash_idx + 1);
    }
    if (filename.empty()) {
        printf("ShaderEncoder:\n"
               "Illegal file name");

        fclose(file_in);
        fclose(file_out);
        return 1;
    }

    fprintf(file_out, "// Automatically generated header file for shader.\n");
    fprintf(file_out, "// See LICENSE.txt for full license statement.\n");
    fprintf(file_out, "\n");

    fprintf(file_out, "const char * const %s = \n", filename.c_str());

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file_in)) {
        std::string line = MakeString(std::string(buffer));
        fprintf(file_out, "\"%s\\n\"\n", line.c_str());
    }

    fprintf(file_out, ";\n");

    fclose(file_in);
    fclose(file_out);

    return 0;
}
