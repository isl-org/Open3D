// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

// NVCC does not support deprecated attribute on Windows prior to v11.
#if defined(__CUDACC__) && defined(_MSC_VER) && __CUDACC_VER_MAJOR__ < 11
#ifndef FMT_DEPRECATED
#define FMT_DEPRECATED
#endif
#endif

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#if FMT_VERSION >= 100000
#include <fmt/std.h>
#endif

#define DEFAULT_IO_BUFFER_SIZE 1024

#include "open3d/Macro.h"

namespace open3d {
namespace utility {

enum class VerbosityLevel {
    /// LogError throws a runtime_error with the given error message. This
    /// should be used if there is no point in continuing the given algorithm at
    /// some point and the error is not returned in another way (e.g., via a
    /// bool/int as return value).
    Error = 0,
    /// LogWarning is used if an error occurs, but the error is also signaled
    /// via a return value (i.e., there is no need to throw an exception). This
    /// warning should further be used if the algorithm encounters a state
    /// that does not break its continuation, but the output is likely not to be
    /// what the user expected.
    Warning = 1,
    /// LogInfo is used to inform the user with expected output, e.g., pressing a
    /// key in the visualizer prints helpful information.
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
                                        Args &&...args) {
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
                            Args &&...args) {
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
                         Args &&...args) {
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
                          Args &&...args) {
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

namespace detail {

/// Captures the format string and caller source location at the call site.
///
/// Because a C++ variadic parameter pack (`Args &&...args`) consumes all
/// trailing arguments, defaulted source-location parameters cannot be placed
/// after `Args...`. Instead, `LogLocation` wraps the first argument (`format`)
/// via a non-explicit converting constructor whose default arguments
/// (`__builtin_FILE()`, `__builtin_LINE()`, `__builtin_FUNCTION()`) are
/// evaluated at the caller's call site, leaving `Args &&...args` to deduce any
/// remaining format arguments.
///
/// Ref: https://twdev.blog/2023/03/variadic_templates/#implicit-type-conversion
/// Ref: https://stackoverflow.com/a/66402319
/// Ref: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
struct LogLocation {
    const char *format;
    const char *file;
    int line;
    const char *function;

    constexpr LogLocation(const char *fmt,
                          const char *f = __builtin_FILE(),
                          int l = __builtin_LINE(),
                          const char *fn = __builtin_FUNCTION()) noexcept
        : format(fmt), file(f), line(l), function(fn) {}
};

}  // namespace detail

/// LogError throws a runtime_error with the given error message. This
/// should be used if there is no point in continuing the given algorithm at
/// some point and the error is not returned in another way (e.g., via a
/// bool/int as return value).
///
/// Usage  : utility::LogError(format_string, arg0, arg1, ...);
/// Example: utility::LogError("name: {}, age: {}", "dog", 5);
template <typename... Args>
[[noreturn]] inline void LogError(detail::LogLocation loc, Args &&...args) {
    Logger::LogError_(loc.file, loc.line, loc.function, loc.format,
                      std::forward<Args>(args)...);
}

/// LogWarning is used if an error occurs, but the error is also signaled
/// via a return value (i.e., there is no need to throw an exception). This
/// warning should further be used if the algorithm encounters a state
/// that does not break its continuation, but the output is likely not to be
/// what the user expected.
///
/// Usage  : utility::LogWarning(format_string, arg0, arg1, ...);
/// Example: utility::LogWarning("name: {}, age: {}", "dog", 5);
template <typename... Args>
inline void LogWarning(detail::LogLocation loc, Args &&...args) {
    Logger::LogWarning_(loc.file, loc.line, loc.function, loc.format,
                        std::forward<Args>(args)...);
}

/// LogInfo is used to inform the user with expected output, e.g., pressing a
/// key in the visualizer prints helpful information.
///
/// Usage  : utility::LogInfo(format_string, arg0, arg1, ...);
/// Example: utility::LogInfo("name: {}, age: {}", "dog", 5);
template <typename... Args>
inline void LogInfo(detail::LogLocation loc, Args &&...args) {
    Logger::LogInfo_(loc.file, loc.line, loc.function, loc.format,
                     std::forward<Args>(args)...);
}

/// LogDebug is used to print debug/additional information on the state of
/// the algorithm.
///
/// Usage  : utility::LogDebug(format_string, arg0, arg1, ...);
/// Example: utility::LogDebug("name: {}, age: {}", "dog", 5);
template <typename... Args>
inline void LogDebug(detail::LogLocation loc, Args &&...args) {
    Logger::LogDebug_(loc.file, loc.line, loc.function, loc.format,
                      std::forward<Args>(args)...);
}

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
