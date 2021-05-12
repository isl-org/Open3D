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

#include "open3d/utility/Console.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/chrono.h>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace utility {

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

struct Logger::Impl {
    // The current print function.
    std::function<void(const std::string &)> print_fcn_;

    // The default print function (that prints to console).
    static std::function<void(const std::string &)> console_print_fcn_;

    // Verbosity level.
    VerbosityLevel verbosity_level_;

    // Colorize and reset the color of a string, does not work on Windows,
    std::string ColorString(const std::string &text,
                            TextColor text_color,
                            int highlight_text) const {
        std::ostringstream msg;
#ifndef _WIN32
        msg << fmt::sprintf("%c[%d;%dm", 0x1B, highlight_text,
                            (int)text_color + 30);
#endif
        msg << text;
#ifndef _WIN32
        msg << fmt::sprintf("%c[0;m", 0x1B);
#endif
        return msg.str();
    }
};

std::function<void(const std::string &)> Logger::Impl::console_print_fcn_ =
        [](const std::string &msg) { std::cout << msg << std::endl; };

Logger::Logger() : impl_(new Logger::Impl()) {
    impl_->print_fcn_ = Logger::Impl::console_print_fcn_;
    impl_->verbosity_level_ = VerbosityLevel::Info;
}

Logger &Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::VError [[noreturn]] (const char *file_name,
                                  int line_number,
                                  const char *function_name,
                                  bool force_console_log,
                                  const char *format,
                                  fmt::format_args args) const {
    std::string err_msg = fmt::vformat(format, args);
    err_msg = fmt::format("[Open3D Error] ({}) {}:{}: {}\n", function_name,
                          file_name, line_number, err_msg);
    err_msg = impl_->ColorString(err_msg, TextColor::Red, 1);

    throw std::runtime_error(err_msg);
}

void Logger::VWarning(const char *file_name,
                      int line_number,
                      const char *function_name,
                      bool force_console_log,
                      const char *format,
                      fmt::format_args args) const {
    if (impl_->verbosity_level_ >= VerbosityLevel::Warning) {
        std::string err_msg = fmt::vformat(format, args);
        err_msg = fmt::format("[Open3D WARNING] {}", err_msg);
        err_msg = impl_->ColorString(err_msg, TextColor::Yellow, 1);
        if (force_console_log) {
            Logger::Impl::console_print_fcn_(err_msg);
        } else {
            impl_->print_fcn_(err_msg);
        }
    }
}

void Logger::VInfo(const char *file_name,
                   int line_number,
                   const char *function_name,
                   bool force_console_log,
                   const char *format,
                   fmt::format_args args) const {
    if (impl_->verbosity_level_ >= VerbosityLevel::Info) {
        std::string err_msg = fmt::vformat(format, args);
        err_msg = fmt::format("[Open3D INFO] {}", err_msg);
        if (force_console_log) {
            Logger::Impl::console_print_fcn_(err_msg);
        } else {
            impl_->print_fcn_(err_msg);
        }
    }
}

void Logger::VDebug(const char *file_name,
                    int line_number,
                    const char *function_name,
                    bool force_console_log,
                    const char *format,
                    fmt::format_args args) const {
    if (impl_->verbosity_level_ >= VerbosityLevel::Debug) {
        std::string err_msg = fmt::vformat(format, args);
        err_msg = fmt::format("[Open3D DEBUG] {}", err_msg);
        if (force_console_log) {
            Logger::Impl::console_print_fcn_(err_msg);
        } else {
            impl_->print_fcn_(err_msg);
        }
    }
}

void Logger::SetPrintFunction(
        std::function<void(const std::string &)> print_fcn) {
    impl_->print_fcn_ = print_fcn;
}

void Logger::SetVerbosityLevel(VerbosityLevel verbosity_level) {
    impl_->verbosity_level_ = verbosity_level;
}

VerbosityLevel Logger::GetVerbosityLevel() const {
    return impl_->verbosity_level_;
}

ConsoleProgressBar::ConsoleProgressBar(size_t expected_count,
                                       const std::string &progress_info,
                                       bool active) {
    Reset(expected_count, progress_info, active);
}

void ConsoleProgressBar::Reset(size_t expected_count,
                               const std::string &progress_info,
                               bool active) {
    expected_count_ = expected_count;
    current_count_ = static_cast<size_t>(-1);  // Guaranteed to wraparound
    progress_info_ = progress_info;
    progress_pixel_ = 0;
    active_ = active;
    operator++();
}

ConsoleProgressBar &ConsoleProgressBar::operator++() {
    SetCurrentCount(current_count_ + 1);
    return *this;
}

void ConsoleProgressBar::SetCurrentCount(size_t n) {
    current_count_ = n;
    if (!active_) {
        return;
    }
    if (current_count_ >= expected_count_) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else {
        size_t new_progress_pixel =
                int(current_count_ * resolution_ / expected_count_);
        if (new_progress_pixel > progress_pixel_) {
            progress_pixel_ = new_progress_pixel;
            int percent = int(current_count_ * 100 / expected_count_);
            fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                       std::string(progress_pixel_, '='),
                       std::string(resolution_ - 1 - progress_pixel_, ' '),
                       percent);
            fflush(stdout);
        }
    }
}

void SetVerbosityLevel(VerbosityLevel level) {
    Logger::GetInstance().SetVerbosityLevel(level);
}

VerbosityLevel GetVerbosityLevel() {
    return Logger::GetInstance().GetVerbosityLevel();
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
    std::vector<std::string> tokens =
            SplitString(str.substr(1, str.length() - 2), ",");
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
