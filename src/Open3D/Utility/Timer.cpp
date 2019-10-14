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

#include "Open3D/Utility/Timer.h"

#include <chrono>

#include "Open3D/Utility/Console.h"

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

double Timer::GetDuration() const {
    return end_time_in_milliseconds_ - start_time_in_milliseconds_;
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
    if (GetDuration() >= time_to_print_ ||
        event_fragment_count_ >= events_to_print_) {
        // print and reset
        if (expectation_ == -1) {
            LogInfo("{} at {:.2f} fps.", fps_timer_info_,
                    double(event_fragment_count_ + 1) / GetDuration() * 1000.0);
        } else {
            LogInfo("{} at {:.2f} fps (progress {:.2f}%).",
                    fps_timer_info_.c_str(),
                    double(event_fragment_count_ + 1) / GetDuration() * 1000.0,
                    (double)event_total_count_ * 100.0 / (double)expectation_);
        }
        Start();
        event_fragment_count_ = 0;
    }
}

}  // namespace utility
}  // namespace open3d
