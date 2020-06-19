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

#include "open3d/visualization/gui/Task.h"

#include <atomic>
#include <thread>

#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
enum class ThreadState { NOT_STARTED, RUNNING, FINISHED };
}

struct Task::Impl {
    std::function<void()> func_;
    std::thread thread_;
    ThreadState state_;
    std::atomic<bool> is_finished_running_;
};

Task::Task(std::function<void()> f) : impl_(new Task::Impl) {
    impl_->func_ = f;
    impl_->state_ = ThreadState::NOT_STARTED;
    impl_->is_finished_running_ = false;
}

Task::~Task() {
    // TODO: if able to cancel, do so here
    WaitToFinish();
}

void Task::Run() {
    if (impl_->state_ != ThreadState::NOT_STARTED) {
        utility::LogWarning("Attempting to Run() already-started Task");
        return;
    }

    auto thread_main = [this]() {
        impl_->func_();
        impl_->is_finished_running_ = true;
    };
    impl_->thread_ = std::thread(thread_main);  // starts thread
    impl_->state_ = ThreadState::RUNNING;
}

bool Task::IsFinished() const {
    switch (impl_->state_) {
        case ThreadState::NOT_STARTED:
            return true;
        case ThreadState::RUNNING:
            return impl_->is_finished_running_;
        case ThreadState::FINISHED:
            return true;
    }
    utility::LogError("Unexpected thread state");
}

void Task::WaitToFinish() {
    if (impl_->state_ == ThreadState::RUNNING) {
        impl_->thread_.join();
        impl_->state_ = ThreadState::FINISHED;
    }
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
