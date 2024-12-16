// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Logging.h"

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

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

void Logger::VError [[noreturn]] (const char *file,
                                  int line,
                                  const char *function,
                                  const std::string &message) const {
    std::string err_msg = fmt::format("[Open3D Error] ({}) {}:{}: {}\n",
                                      function, file, line, message);
    err_msg = impl_->ColorString(err_msg, TextColor::Red, 1);
#ifdef _MSC_VER  // Uncaught exception error messages not shown in Windows
    std::cerr << err_msg << std::endl;
#endif
    throw std::runtime_error(err_msg);
}

void Logger::VWarning(const char *file,
                      int line,
                      const char *function,
                      const std::string &message) const {
    std::string err_msg = fmt::format("[Open3D WARNING] {}", message);
    err_msg = impl_->ColorString(err_msg, TextColor::Yellow, 1);
    impl_->print_fcn_(err_msg);
}

void Logger::VInfo(const char *file,
                   int line,
                   const char *function,
                   const std::string &message) const {
    std::string err_msg = fmt::format("[Open3D INFO] {}", message);
    impl_->print_fcn_(err_msg);
}

void Logger::VDebug(const char *file,
                    int line,
                    const char *function,
                    const std::string &message) const {
    std::string err_msg = fmt::format("[Open3D DEBUG] {}", message);
    impl_->print_fcn_(err_msg);
}

void Logger::SetPrintFunction(
        std::function<void(const std::string &)> print_fcn) {
    impl_->print_fcn_ = print_fcn;
}

const std::function<void(const std::string &)> Logger::GetPrintFunction() {
    return impl_->print_fcn_;
}

void Logger::ResetPrintFunction() {
    impl_->print_fcn_ = impl_->console_print_fcn_;
}

void Logger::SetVerbosityLevel(VerbosityLevel verbosity_level) {
    impl_->verbosity_level_ = verbosity_level;
}

VerbosityLevel Logger::GetVerbosityLevel() const {
    return impl_->verbosity_level_;
}

void SetVerbosityLevel(VerbosityLevel level) {
    Logger::GetInstance().SetVerbosityLevel(level);
}

VerbosityLevel GetVerbosityLevel() {
    return Logger::GetInstance().GetVerbosityLevel();
}

}  // namespace utility
}  // namespace open3d
