// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace open3d {
namespace utility {

class Timer {
public:
    Timer();
    ~Timer();

public:
    static double GetSystemTimeInMilliseconds();

public:
    void Start();
    void Stop();
    void Print(const std::string &timer_info) const;
    double GetDurationInSecond() const;
    double GetDurationInMillisecond() const;
    std::tuple<int, int, double> GetDurationInHMS() const;

private:
    double start_time_in_milliseconds_;
    double end_time_in_milliseconds_;
};

class ScopeTimer : public Timer {
public:
    ScopeTimer(const std::string &scope_timer_info = "");
    ~ScopeTimer();

private:
    std::string scope_timer_info_;
};

class FPSTimer : public Timer {
public:
    FPSTimer(const std::string &fps_timer_info = "",
             int expectation = -1,
             double time_to_print = 3000.0,
             int events_to_print = 100);

    /// Function to signal an event
    /// It automatically prints FPS information when duration is more than
    /// time_to_print_, or event has been signaled events_to_print_ times.
    void Signal();

private:
    std::string fps_timer_info_;
    int expectation_;
    double time_to_print_;
    int events_to_print_;
    int event_fragment_count_;
    int event_total_count_;
};

}  // namespace utility
}  // namespace open3d
