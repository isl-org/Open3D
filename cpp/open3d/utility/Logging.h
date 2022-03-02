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

#pragma once

#include <functional>
#include <memory>
#include <string>

// NVCC does not support deprecated attribute on Windows prior to v11.
#if defined(__CUDACC__) && defined(_MSC_VER) && __CUDACC_VER_MAJOR__ < 11
#ifndef FMT_DEPRECATED
#define FMT_DEPRECATED
#endif
#endif

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#define DEFAULT_IO_BUFFER_SIZE 1024

#include "open3d/Macro.h"

// Mimic "macro in namespace" by concatenating `utility::` and a macro.
// Ref: https://stackoverflow.com/a/11791202
//
// We avoid using (format, ...) since in this case __VA_ARGS__ can be
// empty, and the behavior of pruning trailing comma with ##__VA_ARGS__ is not
// officially standard.
// Ref: https://stackoverflow.com/a/28074198
//
// __PRETTY_FUNCTION__ has to be converted, otherwise a bug regarding [noreturn]
// will be triggered.
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94742

// LogError throws now a runtime_error with the given error message. This
// should be used if there is no point in continuing the given algorithm at
// some point and the error is not returned in another way (e.g., via a
// bool/int as return value).
//
// Usage  : utility::LogError(format_string, arg0, arg1, ...);
// Example: utility::LogError("name: {}, age: {}", "dog", 5);
#define LogError(...)                     \
    Logger::LogError_(__FILE__, __LINE__, \
                      static_cast<const char *>(OPEN3D_FUNCTION), __VA_ARGS__)

// LogWarning is used if an error occurs, but the error is also signaled
// via a return value (i.e., there is no need to throw an exception). This
// warning should further be used, if the algorithms encounters a state
// that does not break its continuation, but the output is likely not to be
// what the user expected.
//
// Usage  : utility::LogWarning(format_string, arg0, arg1, ...);
// Example: utility::LogWarning("name: {}, age: {}", "dog", 5);
#define LogWarning(...)                                             \
    Logger::LogWarning_(__FILE__, __LINE__,                         \
                        static_cast<const char *>(OPEN3D_FUNCTION), \
                        __VA_ARGS__)

// LogInfo is used to inform the user with expected output, e.g, pressed a
// key in the visualizer prints helping information.
//
// Usage  : utility::LogInfo(format_string, arg0, arg1, ...);
// Example: utility::LogInfo("name: {}, age: {}", "dog", 5);
#define LogInfo(...)                     \
    Logger::LogInfo_(__FILE__, __LINE__, \
                     static_cast<const char *>(OPEN3D_FUNCTION), __VA_ARGS__)

// LogDebug is used to print debug/additional information on the state of
// the algorithm.
//
// Usage  : utility::LogDebug(format_string, arg0, arg1, ...);
// Example: utility::LogDebug("name: {}, age: {}", "dog", 5);
#define LogDebug(...)                     \
    Logger::LogDebug_(__FILE__, __LINE__, \
                      static_cast<const char *>(OPEN3D_FUNCTION), __VA_ARGS__)

namespace open3d {
namespace utility {

enum class VerbosityLevel {
    /// LogError throws now a runtime_error with the given error message. This
    /// should be used if there is no point in continuing the given algorithm at
    /// some point and the error is not returned in another way (e.g., via a
    /// bool/int as return value).
    Error = 0,
    /// LogWarning is used if an error occurs, but the error is also signaled
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

/// Logger class should be used as a global singleton object (GetInstance()).
class Logger {
public:
    Logger(Logger const &) = delete;
    void operator=(Logger const &) = delete;

    /// Get Logger global singleton instance.
    static Logger &GetInstance();

    /// Overwrite the default print function, this is useful when you want to
    /// redirect prints rather than printing to stdout. For example, in Open3D's
    /// python binding, the default print function is replaced with py::print().
    ///
    /// \param print_fcn The function for printing. It should take a string
    /// input and returns nothing.
    void SetPrintFunction(std::function<void(const std::string &)> print_fcn);

    /// Reset the print function to the default one (print to console).
    void ResetPrintFunction();

    /// Get the print function used by the Logger.
    const std::function<void(const std::string &)> GetPrintFunction();

    /// Set global verbosity level of Open3D.
    ///
    /// \param verbosity_level Messages with equal or less than verbosity_level
    /// verbosity will be printed.
    void SetVerbosityLevel(VerbosityLevel verbosity_level);

    /// Get global verbosity level of Open3D.
    VerbosityLevel GetVerbosityLevel() const;

    template <typename... Args>
    static void LogError_ [[noreturn]] (const char *file,
                                        int line,
                                        const char *function,
                                        const char *format,
                                        Args &&... args) {
        if (sizeof...(Args) > 0) {
            Logger::GetInstance().VError(
                    file, line, function,
                    FormatArgs(format, fmt::make_format_args(args...)));
        } else {
            Logger::GetInstance().VError(file, line, function,
                                         std::string(format));
        }
    }
    template <typename... Args>
    static void LogWarning_(const char *file,
                            int line,
                            const char *function,
                            const char *format,
                            Args &&... args) {
        if (Logger::GetInstance().GetVerbosityLevel() >=
            VerbosityLevel::Warning) {
            if (sizeof...(Args) > 0) {
                Logger::GetInstance().VWarning(
                        file, line, function,
                        FormatArgs(format, fmt::make_format_args(args...)));
            } else {
                Logger::GetInstance().VWarning(file, line, function,
                                               std::string(format));
            }
        }
    }
    template <typename... Args>
    static void LogInfo_(const char *file,
                         int line,
                         const char *function,
                         const char *format,
                         Args &&... args) {
        if (Logger::GetInstance().GetVerbosityLevel() >= VerbosityLevel::Info) {
            if (sizeof...(Args) > 0) {
                Logger::GetInstance().VInfo(
                        file, line, function,
                        FormatArgs(format, fmt::make_format_args(args...)));
            } else {
                Logger::GetInstance().VInfo(file, line, function,
                                            std::string(format));
            }
        }
    }
    template <typename... Args>
    static void LogDebug_(const char *file,
                          int line,
                          const char *function,
                          const char *format,
                          Args &&... args) {
        if (Logger::GetInstance().GetVerbosityLevel() >=
            VerbosityLevel::Debug) {
            if (sizeof...(Args) > 0) {
                Logger::GetInstance().VDebug(
                        file, line, function,
                        FormatArgs(format, fmt::make_format_args(args...)));
            } else {
                Logger::GetInstance().VDebug(file, line, function,
                                             std::string(format));
            }
        }
    }

private:
    Logger();
    static std::string FormatArgs(const char *format, fmt::format_args args) {
        std::string err_msg = fmt::vformat(format, args);
        return err_msg;
    }
    void VError [[noreturn]] (const char *file,
                              int line,
                              const char *function,
                              const std::string &message) const;
    void VWarning(const char *file,
                  int line,
                  const char *function,
                  const std::string &message) const;
    void VInfo(const char *file,
               int line,
               const char *function,
               const std::string &message) const;
    void VDebug(const char *file,
                int line,
                const char *function,
                const std::string &message) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Set global verbosity level of Open3D
///
/// \param level Messages with equal or less than verbosity_level verbosity will
/// be printed.
void SetVerbosityLevel(VerbosityLevel level);

/// Get global verbosity level of Open3D.
VerbosityLevel GetVerbosityLevel();

class VerbosityContextManager {
public:
    VerbosityContextManager(VerbosityLevel level) : level_(level) {}

    void Enter() {
        level_backup_ = Logger::GetInstance().GetVerbosityLevel();
        Logger::GetInstance().SetVerbosityLevel(level_);
    }

    void Exit() { Logger::GetInstance().SetVerbosityLevel(level_backup_); }

private:
    VerbosityLevel level_;
    VerbosityLevel level_backup_;
};

}  // namespace utility
}  // namespace open3d
