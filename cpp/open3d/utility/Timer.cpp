// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Timer.h"

#include <chrono>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

Timer::Timer()
    : start_time_in_milliseconds_(0.0), end_time_in_milliseconds_(0.0) {}

Timer::~Timer() {}

double Timer::GetSystemTimeInMilliseconds() {
    std::chrono::duration<double, std::milli> current_time =
            std::chrono::high_resolution_clock::now().time_since_epoch();
    return current_time.count();
}

void Timer::Start() {
    start_time_in_milliseconds_ = GetSystemTimeInMilliseconds();
}

void Timer::Stop() {
    end_time_in_milliseconds_ = GetSystemTimeInMilliseconds();
}

double Timer::GetDurationInMillisecond() const {
    return end_time_in_milliseconds_ - start_time_in_milliseconds_;
}

double Timer::GetDurationInSecond() const {
    return (end_time_in_milliseconds_ - start_time_in_milliseconds_) / 1000.0;
}

std::tuple<int, int, double> Timer::GetDurationInHMS() const {
    double duration = GetDurationInSecond();
    int hours = static_cast<int>(duration / 3600.0);
    int minutes = static_cast<int>((duration - hours * 3600.0) / 60.0);
    double seconds = duration - hours * 3600.0 - minutes * 60.0;
    return std::make_tuple(hours, minutes, seconds);
}

void Timer::Print(const std::string &timer_info) const {
    LogInfo("{} {:.2f} ms.", timer_info,
            end_time_in_milliseconds_ - start_time_in_milliseconds_);
}

ScopeTimer::ScopeTimer(const std::string &scope_timer_info /* = ""*/)
    : scope_timer_info_(scope_timer_info) {
    Timer::Start();
}

ScopeTimer::~ScopeTimer() {
    Timer::Stop();
    Timer::Print(scope_timer_info_ + " took");
}

FPSTimer::FPSTimer(const std::string &fps_timer_info /* = ""*/,
                   int expectation /* = -1*/,
                   double time_to_print /* = 3000.0*/,
                   int events_to_print /* = 100*/)
    : fps_timer_info_(fps_timer_info),
      expectation_(expectation),
      time_to_print_(time_to_print),
      events_to_print_(events_to_print),
      event_fragment_count_(0),
      event_total_count_(0) {
    Start();
}

void FPSTimer::Signal() {
    event_fragment_count_++;
    event_total_count_++;
    Stop();
    if (GetDurationInMillisecond() >= time_to_print_ ||
        event_fragment_count_ >= events_to_print_) {
        // print and reset
        if (expectation_ == -1) {
            LogInfo("{} at {:.2f} fps.", fps_timer_info_,
                    double(event_fragment_count_ + 1) /
                            GetDurationInMillisecond() * 1000.0);
        } else {
            LogInfo("{} at {:.2f} fps (progress {:.2f}%).",
                    fps_timer_info_.c_str(),
                    double(event_fragment_count_ + 1) /
                            GetDurationInMillisecond() * 1000.0,
                    (double)event_total_count_ * 100.0 / (double)expectation_);
        }
        Start();
        event_fragment_count_ = 0;
    }
}

}  // namespace utility
}  // namespace open3d
