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

#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

#define FMT_STRING_ALIAS 1
#include "fmt/format.h"

#define DEFAULT_IO_BUFFER_SIZE 1024

namespace open3d {
namespace utility {

enum class VerbosityLevel {
    VerboseOff = 0,
    VerboseFatal = 1,
    VerboseError = 2,
    VerboseWarning = 3,
    VerboseInfo = 4,
    VerboseDebug = 5,
};

enum class TextColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7
};

/// Internal function to change text color for the console
/// Note there is no security check for parameters.
/// \param text_color, from 0 to 7, they are black, red, green, yellow, blue,
/// magenta, cyan, white
/// \param emphasis_text is 0 or 1
inline void ChangeConsoleColor(TextColor text_color, int highlight_text) {
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

inline void ResetConsoleColor() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(
            h, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
#else
    printf("%c[0;m", 0x1B);
#endif
}

static VerbosityLevel global_verbosity_level;

void SetVerbosityLevel(VerbosityLevel verbosity_level);
VerbosityLevel GetVerbosityLevel();


inline void VPrintFatal(const char *format, fmt::format_args args) {
  if (global_verbosity_level >= VerbosityLevel::VerboseFatal) {
    ChangeConsoleColor(TextColor::Red, 1);
    fmt::print("[Open3D FATAL] ");
    fmt::vprint(format, args);
    ResetConsoleColor();
    throw std::runtime_error("");
  }
}

template <typename... Args>
inline void NewPrintFatal(const char *format, const Args & ... args) {
  VPrintFatal(format, fmt::make_format_args(args...));
}

inline void VPrintError(const char *format, fmt::format_args args) {
  if (global_verbosity_level >= VerbosityLevel::VerboseError) {
    ChangeConsoleColor(TextColor::Red, 1);
    fmt::print("[Open3D ERROR] ");
    fmt::vprint(format, args);
    ResetConsoleColor();
  }
}

template <typename... Args>
inline void NewPrintError(const char *format, const Args & ... args) {
  VPrintError(format, fmt::make_format_args(args...));
}

inline void VPrintWarning(const char *format, fmt::format_args args) {
  if (global_verbosity_level >= VerbosityLevel::VerboseWarning) {
    ChangeConsoleColor(TextColor::Yellow, 1);
    fmt::print("[Open3D WARNING] ");
    fmt::vprint(format, args);
    ResetConsoleColor();
  }
}

template <typename... Args>
inline void NewPrintWarning(const char *format, const Args & ... args) {
  VPrintWarning(format, fmt::make_format_args(args...));
}

inline void VPrintInfo(const char *format, fmt::format_args args) {
  if (global_verbosity_level >= VerbosityLevel::VerboseInfo) {
    // fmt::print("[Open3D INFO] ");
    fmt::vprint(format, args);
  }
}

template <typename... Args>
inline void NewPrintInfo(const char *format, const Args & ... args) {
  VPrintInfo(format, fmt::make_format_args(args...));
}

inline void VPrintDebug(const char *format, fmt::format_args args) {
  if (global_verbosity_level >= VerbosityLevel::VerboseDebug) {
    fmt::print("[Open3D DEBUG] ");
    fmt::vprint(format, args);
  }
}

template <typename... Args>
inline void NewPrintDebug(const char *format, const Args & ... args) {
  VPrintDebug(format, fmt::make_format_args(args...));
}


void ResetConsoleProgress(const int64_t expected_count,
                          const std::string &progress_info = "");

void AdvanceConsoleProgress();

std::string GetCurrentTimeStamp();

std::string GetProgramOptionAsString(int argc,
                                     char **argv,
                                     const std::string &option,
                                     const std::string &default_value = "");

int GetProgramOptionAsInt(int argc,
                          char **argv,
                          const std::string &option,
                          const int default_value = 0);

double GetProgramOptionAsDouble(int argc,
                                char **argv,
                                const std::string &option,
                                const double default_value = 0.0);

Eigen::VectorXd GetProgramOptionAsEigenVectorXd(
        int argc,
        char **argv,
        const std::string &option,
        const Eigen::VectorXd default_value = Eigen::VectorXd::Zero(0));

bool ProgramOptionExists(int argc, char **argv, const std::string &option);

bool ProgramOptionExistsAny(int argc,
                            char **argv,
                            const std::vector<std::string> &options);
}  // namespace utility
}  // namespace open3d
