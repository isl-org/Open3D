// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#ifdef USE_SPNAV
#include "open3d/visualization/utility/SpaceMouse.h"
#include "open3d/utility/Logging.h"

#include <spnav.h>


namespace open3d {
namespace visualization {

std::shared_ptr<SpaceMouse> SpaceMouse::me_;

void SpaceMouse::Exit() {
    if(me_) {
        me_->Stop();
    }
}
SpaceMouse* SpaceMouse::GetInstance() {
    if (!me_) {
        me_ = std::shared_ptr<SpaceMouse>(new SpaceMouse());
    }
    return me_.get();
}
SpaceMouse::SpaceMouse() : evt_{} {
    if (spnav_open() < 0) {
        return;
    }
    thread_ = std::make_shared<std::thread>([this]() { this->Wait();});
    ready_ = true;
    atexit(SpaceMouse::Exit);
}
SpaceMouse::~SpaceMouse() {
    Stop();
}
void SpaceMouse::Recv(void *data) {
    auto evt = (spnav_event *)data;
    auto &e = evt_;
    std::unique_lock<std::mutex> locker(mt_);
    if (evt->type == SPNAV_EVENT_MOTION) {
        e.type = SpaceMouseEvent::MOTION;
        e.motion.rx = evt->motion.rx;
        e.motion.ry = evt->motion.ry;
        e.motion.rz = evt->motion.rz;
        e.motion.x = evt->motion.x;
        e.motion.y = evt->motion.y;
        e.motion.z = evt->motion.z;
        e.motion.period = evt->motion.period;
    } else {
        e.type = SpaceMouseEvent::BUTTON;
        e.button.btn_num = evt->button.bnum;
        e.button.press = (evt->button.press != 0);
    }
    has_evt_ = true;
}
void SpaceMouse::Stop() {
    if (ready_) {
        ready_ = false;
        spnav_close();
        thread_->join();
        thread_.reset();
    }
}
void SpaceMouse::Wait() {
    spnav_event evt;
    while(ready_) {
        auto typ = spnav_poll_event(&evt);
        if (typ != 0) {
            Recv(&evt);
            spnav_remove_events(typ);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
}
bool SpaceMouse::Poll(SpaceMouseEvent &e) {
    if (ready_ && has_evt_) {
        std::unique_lock<std::mutex> locker(mt_);
        has_evt_ = false;
        e = evt_;
        return true;
    }
    return false;
}
}  // namespace visualization
}  // namespace open3d
#endif
