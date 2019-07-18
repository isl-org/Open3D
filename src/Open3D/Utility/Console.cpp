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

#include "Open3D/Utility/Console.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/time.h>

#include "Open3D/Utility/Helper.h"

namespace open3d {

namespace utility {

void Logger::ChangeConsoleColor(TextColor text_color, int highlight_text) {
#ifdef _WIN32
    const WORD EMPHASIS_MASK[2] = {0, FOREGROUND_INTENSITY};
    const WORD COLOR_MASK[8] = {
            0,
            FOREGROUND_RED,
            FOREGROUND_GREEN,
            FOREGROUND_GREEN | FOREGROUND_RED,
            FOREGROUND_BLUE,
            FOREGROUND_RED | FOREGROUND_BLUE,
            FOREGROUND_GREEN | FOREGROUND_BLUE,
            FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED};
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(
            h, EMPHASIS_MASK[highlight_text] | COLOR_MASK[(int)text_color]);
#else
    printf("%c[%d;%dm", 0x1B, highlight_text, (int)text_color + 30);
#endif
}

void Logger::ResetConsoleColor() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(
            h, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
#else
    printf("%c[0;m", 0x1B);
#endif
}

std::string GetCurrentTimeStamp() {
    std::time_t t = std::time(nullptr);
    return fmt::format("{:%Y-%m-%d-%H-%M-%S}", *std::localtime(&t));
}

std::string GetProgramOptionAsString(
        int argc,
        char **argv,
        const std::string &option,
        const std::string &default_value /* = ""*/) {
    char **itr = std::find(argv, argv + argc, option);
    if (itr != argv + argc && ++itr != argv + argc) {
        return std::string(*itr);
    }
    return default_value;
}

int GetProgramOptionAsInt(int argc,
                          char **argv,
                          const std::string &option,
                          const int default_value /* = 0*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0) {
        return default_value;
    }
    char *end;
    errno = 0;
    long l = std::strtol(str.c_str(), &end, 0);
    if ((errno == ERANGE && l == LONG_MAX) || l > INT_MAX) {
        return default_value;
    } else if ((errno == ERANGE && l == LONG_MIN) || l < INT_MIN) {
        return default_value;
    } else if (*end != '\0') {
        return default_value;
    }
    return (int)l;
}

double GetProgramOptionAsDouble(int argc,
                                char **argv,
                                const std::string &option,
                                const double default_value /* = 0.0*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0) {
        return default_value;
    }
    char *end;
    errno = 0;
    double l = std::strtod(str.c_str(), &end);
    if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
        return default_value;
    } else if (*end != '\0') {
        return default_value;
    }
    return l;
}

Eigen::VectorXd GetProgramOptionAsEigenVectorXd(
        int argc,
        char **argv,
        const std::string &option,
        const Eigen::VectorXd default_value /* =
        Eigen::VectorXd::Zero()*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0 || (!(str.front() == '(' && str.back() == ')') &&
                              !(str.front() == '[' && str.back() == ']') &&
                              !(str.front() == '<' && str.back() == '>'))) {
        return default_value;
    }
    std::vector<std::string> tokens;
    SplitString(tokens, str.substr(1, str.length() - 2), ",");
    Eigen::VectorXd vec(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        char *end;
        errno = 0;
        double l = std::strtod(tokens[i].c_str(), &end);
        if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
            return default_value;
        } else if (*end != '\0') {
            return default_value;
        }
        vec(i) = l;
    }
    return vec;
}

bool ProgramOptionExists(int argc, char **argv, const std::string &option) {
    return std::find(argv, argv + argc, option) != argv + argc;
}

bool ProgramOptionExistsAny(int argc,
                            char **argv,
                            const std::vector<std::string> &options) {
    for (const auto &option : options) {
        if (ProgramOptionExists(argc, argv, option)) {
            return true;
        }
    }
    return false;
}

}  // namespace utility
}  // namespace open3d
