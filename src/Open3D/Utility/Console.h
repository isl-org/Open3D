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

#define FMT_HEADER_ONLY 1
#define FMT_STRING_ALIAS 1
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#define DEFAULT_IO_BUFFER_SIZE 1024

namespace open3d {
namespace utility {

enum class VerbosityLevel {
    Off = 0,
    Fatal = 1,
    Error = 2,
    Warning = 3,
    Info = 4,
    Debug = 5,
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

    Logger() : verbosity_level_(VerbosityLevel::Info) {}
    Logger(Logger const &) = delete;
    void operator=(Logger const &) = delete;

    static Logger &i() {
        static Logger instance;
        return instance;
    }

    /// Internal function to change text color for the console
    /// Note there is no security check for parameters.
    /// \param text_color, from 0 to 7, they are black, red, green, yellow,
    /// blue, magenta, cyan, white \param emphasis_text is 0 or 1
    void ChangeConsoleColor(TextColor text_color, int highlight_text);
    void ResetConsoleColor();

    void VFatal(const char *format, fmt::format_args args) {
        if (verbosity_level_ >= VerbosityLevel::Fatal) {
            ChangeConsoleColor(TextColor::Red, 1);
            fmt::print("[Open3D FATAL] ");
            fmt::vprint(format, args);
            ResetConsoleColor();
            exit(-1);
        }
    }

    void VError(const char *format, fmt::format_args args) {
        if (verbosity_level_ >= VerbosityLevel::Error) {
            ChangeConsoleColor(TextColor::Red, 1);
            fmt::print("[Open3D ERROR] ");
            fmt::vprint(format, args);
            ResetConsoleColor();
        }
    }

    void VWarning(const char *format, fmt::format_args args) {
        if (verbosity_level_ >= VerbosityLevel::Warning) {
            ChangeConsoleColor(TextColor::Yellow, 1);
            fmt::print("[Open3D WARNING] ");
            fmt::vprint(format, args);
            ResetConsoleColor();
        }
    }

    void VInfo(const char *format, fmt::format_args args) {
        if (verbosity_level_ >= VerbosityLevel::Info) {
            fmt::print("[Open3D INFO] ");
            fmt::vprint(format, args);
        }
    }

    void VDebug(const char *format, fmt::format_args args) {
        if (verbosity_level_ >= VerbosityLevel::Debug) {
            fmt::print("[Open3D DEBUG] ");
            fmt::vprint(format, args);
        }
    }

    template <typename... Args>
    void Fatal(const char *format, const Args &... args) {
        VFatal(format, fmt::make_format_args(args...));
    }

    template <typename... Args>
    void Error(const char *format, const Args &... args) {
        VError(format, fmt::make_format_args(args...));
    }

    template <typename... Args>
    void Warning(const char *format, const Args &... args) {
        VWarning(format, fmt::make_format_args(args...));
    }

    template <typename... Args>
    void Info(const char *format, const Args &... args) {
        VInfo(format, fmt::make_format_args(args...));
    }

    template <typename... Args>
    void Debug(const char *format, const Args &... args) {
        VDebug(format, fmt::make_format_args(args...));
    }

    template <typename... Args>
    void Fatalf(const char *format, const Args &... args) {
        if (verbosity_level_ >= VerbosityLevel::Fatal) {
            ChangeConsoleColor(TextColor::Red, 1);
            fmt::print("[Open3D FATAL] ");
            fmt::printf(format, args...);
            ResetConsoleColor();
            exit(-1);
        }
    }

    template <typename... Args>
    void Errorf(const char *format, const Args &... args) {
        if (verbosity_level_ >= VerbosityLevel::Error) {
            ChangeConsoleColor(TextColor::Red, 1);
            fmt::print("[Open3D ERROR] ");
            fmt::printf(format, args...);
            ResetConsoleColor();
        }
    }

    template <typename... Args>
    void Warningf(const char *format, const Args &... args) {
        if (verbosity_level_ >= VerbosityLevel::Warning) {
            ChangeConsoleColor(TextColor::Yellow, 1);
            fmt::print("[Open3D WARNING] ");
            fmt::printf(format, args...);
            ResetConsoleColor();
        }
    }

    template <typename... Args>
    void Infof(const char *format, const Args &... args) {
        if (verbosity_level_ >= VerbosityLevel::Info) {
            fmt::print("[Open3D INFO] ");
            fmt::printf(format, args...);
        }
    }

    template <typename... Args>
    void Debugf(const char *format, const Args &... args) {
        if (verbosity_level_ >= VerbosityLevel::Debug) {
            fmt::print("[Open3D DEBUG] ");
            fmt::printf(format, args...);
        }
    }

public:
    VerbosityLevel verbosity_level_;
};

inline void SetVerbosityLevel(VerbosityLevel level) {
    Logger::i().verbosity_level_ = level;
}

inline VerbosityLevel GetVerbosityLevel() {
    return Logger::i().verbosity_level_;
}

template <typename... Args>
inline void LogFatal(const char *format, const Args &... args) {
    Logger::i().VFatal(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogError(const char *format, const Args &... args) {
    Logger::i().VError(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogWarning(const char *format, const Args &... args) {
    Logger::i().VWarning(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogInfo(const char *format, const Args &... args) {
    Logger::i().VInfo(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogDebug(const char *format, const Args &... args) {
    Logger::i().VDebug(format, fmt::make_format_args(args...));
}

template <typename... Args>
inline void LogFatalf(const char *format, const Args &... args) {
    Logger::i().Fatalf(format, args...);
}

template <typename... Args>
inline void LogErrorf(const char *format, const Args &... args) {
    Logger::i().Errorf(format, args...);
}

template <typename... Args>
inline void LogWarningf(const char *format, const Args &... args) {
    Logger::i().Warningf(format, args...);
}

template <typename... Args>
inline void LogInfof(const char *format, const Args &... args) {
    Logger::i().Infof(format, args...);
}

template <typename... Args>
inline void LogDebugf(const char *format, const Args &... args) {
    Logger::i().Debugf(format, args...);
}

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
        current_count_ = -1;
        progress_info_ = progress_info;
        progress_pixel_ = 0;
        active_ = active;
        operator++();
    }

    ConsoleProgressBar &operator++() {
        current_count_++;
        if (!active_) {
            return *this;
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
        return *this;
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
