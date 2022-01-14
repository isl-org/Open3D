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

#pragma once

#ifdef USE_SPNAV
#include <memory>
#include <map>
#include <functional>
#include <thread>
#include <mutex>


namespace open3d {
namespace visualization {

struct SpaceMouseEvent {
    enum Type {MOTION, BUTTON};
    Type type;
    union {
        struct {
            int x, y, z;
            int rx, ry, rz;
            unsigned int period;
        } motion;
        struct {
            bool press;
            int btn_num;
        } button;
    };
    void adjust(int coeff) {
        if (type == MOTION and coeff > 0) {
            motion.rx /= coeff;
            motion.ry /= coeff;
            motion.rz /= coeff;
            motion.x /= coeff;
            motion.y /= coeff;
            motion.z /= coeff;
        }
    }
    void adjust(int rx, int ry, int rz, int x, int y, int z) {
        if (type == MOTION) {
            if (rx > 0) motion.rx /= rx;
            if (ry > 0) motion.ry /= ry;
            if (rz > 0) motion.rz /= rz;
            if (x > 0) motion.x /= x;
            if (y > 0) motion.y /= y;
            if (z > 0) motion.z /= z;
        }
    }
};

class SpaceMouse {
public:
    using Callback = std::function<void(const SpaceMouseEvent &)>;
    static SpaceMouse* GetInstance();
    static void Exit();
    ~SpaceMouse();
    bool Poll(SpaceMouseEvent &e);
    void Wait();
    void Stop();

private:
    void Recv(void *);
    SpaceMouse();

private:
    bool ready_ = false;
    std::shared_ptr<std::thread> thread_;
    SpaceMouseEvent evt_;
    bool has_evt_ = false;
    std::mutex mt_;
    static std::shared_ptr<SpaceMouse> me_;
};

}  // namespace visualization
}  // namespace open3d
#endif
