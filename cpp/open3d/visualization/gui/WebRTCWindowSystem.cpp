// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/gui/WebRTCWindowSystem.h"

#include <chrono>
#include <sstream>
#include <thread>

#include "open3d/io/ImageIO.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/utility/Draw.h"

namespace open3d {
namespace visualization {
namespace gui {

struct WebRTCWindowSystem::Impl {};

WebRTCWindowSystem::WebRTCWindowSystem()
    : BitmapWindowSystem(BitmapWindowSystem::Rendering::HEADLESS) {
    auto draw_callback = [](gui::Window *window,
                            std::shared_ptr<geometry::Image> im) -> void {
        static int image_id = 0;
        utility::LogInfo("draw_callback called, image id {}", image_id);
        io::WriteImage(fmt::format("headless_{}.jpg", image_id), *im);
        image_id++;
    };
    SetOnWindowDraw(draw_callback);
}

WebRTCWindowSystem::~WebRTCWindowSystem() {}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
