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
#include <iostream>
#include <string>
#include <vector>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY 1
#endif
#ifndef FMT_STRING_ALIAS
#define FMT_STRING_ALIAS 1
#endif
// NVCC does not support deprecated attribute on Windows prior to v11.
#if defined(__CUDACC__) && defined(_MSC_VER) && __CUDACC_VER_MAJOR__ < 11
#ifndef FMT_DEPRECATED
#define FMT_DEPRECATED
#endif
#endif
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#define DEFAULT_IO_BUFFER_SIZE 1024

// Compiler-specific function macro.
// Ref: https://stackoverflow.com/a/4384825
#ifdef _WIN32
#define __FN__ __FUNCSIG__
#else
#define __FN__ __PRETTY_FUNCTION__
#endif

// Mimic 'macro in namespace' by concatenating utility:: and _LogError.
// Ref: https://stackoverflow.com/a/11791202
// We avoid using (format, ...) since in this case __VA_ARGS__ can be
// empty, and the behavior of pruning trailing comma with ##__VA_ARGS__ is not
// officially standard.
// Ref: https://stackoverflow.com/a/28074198
// __PRETTY_FUNCTION__ has to be converted, otherwise a bug regarding [noreturn]
// will be triggered.
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94742
#define LogError(...) \
    _LogError(__FILE__, __LINE__, (const char *)__FN__, __VA_ARGS__)
#define LogErrorConsole LogError

namespace open3d {
namespace utility {

enum class VerbosityLevel {
    /// LogError throws now a runtime_error with the given error message. This
    /// should be used if there is no point in continuing the given algorithm at
    /// some point and the error is not returned in another way (e.g., via a
    /// bool/int as return value).
    Error = 0,
    /// LogWarning is used if an error occured, but the error is also signaled
    /// via a return value (i.e., there is no need to throw an exception). This
    /// warning should further be used, if the algorithms encounters a state
    /// that does not break its continuation, but the output is likely not to be
    /// what the user expected.
    Warning = 1,
    /// LogInfo is used to inform the user with expected output, e.g, pressed a
    /// key in the visualizer prints helping information.
    Info = 2,
    /// LogDebug is used to print debug/additional information on the state of
    /// the algorithm.
    Debug = 3,
};

class Logger {
public:
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

    Logger(Logger const &) = delete;

    void operator=(Logger const &) = delete;

    static Logger &i() {
        static Logger instance;
        return instance;
    }

    void VError [[noreturn]] (const char *file_name,
                              int line_number,
                              const char *function_name,
                              const char *format,
                              fmt::format_args args) const {
        std::string err_msg = fmt::vformat(format, args);
        err_msg = fmt::format("[Open3D Error] ({}:{}): {}", function_name,
                              file_name, line_number, err_msg);
        err_msg = ColorString(err_msg, TextColor::Red, 1);
        if (print_fcn_overwritten_) {
            // In Jupyter, print_fcn_ is replaced by Pybind11's py::print() and
            // prints the error message inside Jupyter cell.
            Logger::print_fcn_(err_msg);
        }
        throw std::runtime_error(err_msg);
    }

    void VWarning(const char *format,
                  fmt::format_args args,
                  bool force_console_log = false) const {
        if (verbosity_level_ >= VerbosityLevel::Warning) {
            std::string err_msg = fmt::vformat(format, args);
            err_msg = fmt::format("[Open3D WARNING] {}", err_msg);
            err_msg = ColorString(err_msg, TextColor::Yellow, 1);
            if (force_console_log) {
                Logger::console_print_fcn_(err_msg);
            } else {
                print_fcn_(err_msg);
            }
        }
    }

    void VInfo(const char *format,
               fmt::format_args args,
               bool force_console_log = false) const {
        if (verbosity_level_ >= VerbosityLevel::Info) {
            std::string err_msg = fmt::vformat(format, args);
            err_msg = fmt::format("[Open3D INFO] {}", err_msg);
            if (force_console_log) {
                Logger::console_print_fcn_(err_msg);
            } else {
                print_fcn_(err_msg);
            }
        }
    }

    void VDebug(const char *format,
                fmt::format_args args,
                bool force_console_log = false) const {
        if (verbosity_level_ >= VerbosityLevel::Debug) {
            std::string err_msg = fmt::vformat(format, args);
            err_msg = fmt::format("[Open3D DEBUG] {}", err_msg);
            if (force_console_log) {
                Logger::console_print_fcn_(err_msg);
            } else {
                print_fcn_(err_msg);
            }
        }
    }

    void OverwritePrintFunction(
            std::function<void(const std::string &)> print_fcn) {
        print_fcn_ = print_fcn;
        print_fcn_overwritten_ = true;
    }

    void SetVerbosityLevel(VerbosityLevel verbosity_level) {
        verbosity_level_ = verbosity_level;
    }

    VerbosityLevel GetVerbosityLevel() const { return verbosity_level_; }

private:
    Logger()
        : print_fcn_(Logger::console_print_fcn_),
          verbosity_level_(VerbosityLevel::Info) {}

    /// Internal function to change text color for the console
    /// Note there is no safety check for parameters.
    /// \param text_color from 0 to 7, they are black, red, green, yellow,
    /// blue, magenta, cyan, white
    /// \param highlight_text is 0 or 1
    void ChangeConsoleColor(TextColor text_color, int highlight_text) const;
    void ResetConsoleColor() const;
    /// Colorize and reset the color of a string, does not work on Windows
    std::string ColorString(const std::string &text,
                            TextColor text_color,
                            int highlight_text) const;
    std::function<void(const std::string &)> print_fcn_;
    static std::function<void(const std::string &)> console_print_fcn_;
    VerbosityLevel verbosity_level_;
    bool print_fcn_overwritten_ = false;
};

/// Set global verbosity level of Open3D
///
/// \param level Messages with equal or less than verbosity_level verbosity will
/// be printed.
inline void SetVerbosityLevel(VerbosityLevel level) {
    Logger::i().SetVerbosityLevel(level);
}

/// Get global verbosity level of Open3D.
inline VerbosityLevel GetVerbosityLevel() {
    return Logger::i().GetVerbosityLevel();
}

template <typename... Args>
inline void _LogError [[noreturn]] (const char *file_name,
                                    int line_number,
                                    const char *function_name,
                                    const char *format,
                                    Args &&... args) {
    Logger::i().VError(file_name, line_number, function_name, format,
                       fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogWarning(const char *format, const Args &... args) {
    Logger::i().VWarning(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogWarningConsole(const char *format, const Args &... args) {
    Logger::i().VWarning(format, fmt::make_format_args(args...), true);
}

template <typename... Args>
inline void LogInfo(const char *format, const Args &... args) {
    Logger::i().VInfo(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogInfoConsole(const char *format, const Args &... args) {
    Logger::i().VInfo(format, fmt::make_format_args(args...), true);
}

template <typename... Args>
inline void LogDebug(const char *format, const Args &... args) {
    Logger::i().VDebug(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogDebugConsole(const char *format, const Args &... args) {
    Logger::i().VDebug(format, fmt::make_format_args(args...), true);
}

class VerbosityContextManager {
public:
    VerbosityContextManager(VerbosityLevel level) : level_(level) {}

    void enter() {
        level_backup_ = Logger::i().GetVerbosityLevel();
        Logger::i().SetVerbosityLevel(level_);
    }

    void exit() { Logger::i().SetVerbosityLevel(level_backup_); }

private:
    VerbosityLevel level_;
    VerbosityLevel level_backup_;
};

class ConsoleProgressBar {
public:
    ConsoleProgressBar(size_t expected_count,
                       const std::string &progress_info,
                       bool active = false) {
        reset(expected_count, progress_info, active);
    }

    void reset(size_t expected_count,
               const std::string &progress_info,
               bool active) {
        expected_count_ = expected_count;
        current_count_ = static_cast<size_t>(-1);  // Guaranteed to wraparound
        progress_info_ = progress_info;
        progress_pixel_ = 0;
        active_ = active;
        operator++();
    }

    ConsoleProgressBar &operator++() {
        SetCurrentCount(current_count_ + 1);
        return *this;
    }

    void SetCurrentCount(size_t n) {
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

private:
    const size_t resolution_ = 40;
    size_t expected_count_;
    size_t current_count_;
    std::string progress_info_;
    size_t progress_pixel_;
    bool active_;
};

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
