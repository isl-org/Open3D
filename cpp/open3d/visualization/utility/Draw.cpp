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

#include "open3d/visualization/utility/Draw.h"

#include <chrono>
#include <sstream>
#include <thread>

#include "open3d/io/ImageIO.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/WebRTCWindowSystem.h"

namespace open3d {
namespace visualization {

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<geometry::Geometry3D> g,
                       bool vis /*= true*/) {
    this->name = n;
    this->geometry = g;
    this->is_visible = vis;
}

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<t::geometry::Geometry> tg,
                       bool vis /*= true*/) {
    this->name = n;
    this->tgeometry = tg;
    this->is_visible = vis;
}

// ----------------------------------------------------------------------------
void Draw(const std::vector<std::shared_ptr<geometry::Geometry3D>> &geometries,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(geometries.size());
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), geometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name /*= "Open3D"*/,
        int width /*= 1024*/,
        int height /*= 768*/,
        const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(tgeometries.size());
    for (size_t i = 0; i < tgeometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), tgeometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    auto &o3d_app = gui::Application::GetInstance();
    auto webrtc_window = std::make_shared<gui::WebRTCWindowSystem>();
    o3d_app.SetWindowSystem(webrtc_window);
    o3d_app.Initialize();

    auto draw = std::make_shared<visualizer::O3DVisualizer>(window_name, width,
                                                            height);
    for (auto &o : objects) {
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry);
        } else {
            draw->AddGeometry(o.name, o.tgeometry);
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }

    for (auto &act : actions) {
        draw->AddAction(act.name, act.callback);
    }

    draw->ResetCameraToDefault();

    gui::Application::GetInstance().AddWindow(draw);

    auto emulate_mouse_events = [webrtc_window, draw]() -> void {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            utility::LogInfo("emulate_mouse_events called");

            // clang-format off
            // MouseEvent{type: Type::BUTTON_DOWN, x: 139, y: 366, modifiers: 0, button.button: MouseButton::LEFT}
            // MouseEvent{type: Type::DRAG, x: 149, y: 362, modifiers: 0, move.buttons : 1}
            // MouseEvent{type: Type::DRAG, x: 209, y: 338, modifiers: 0, move.buttons : 1}
            // MouseEvent{type: Type::DRAG, x: 259, y: 319, modifiers: 0, move.buttons : 1}
            // MouseEvent{type: Type::BUTTON_UP, x: 263, y: 318, modifiers: 0, button.button: MouseButton::LEFT}
            // clang-format on

            gui::MouseEvent me;

            me = gui::MouseEvent{gui::MouseEvent::Type::BUTTON_DOWN, 139, 366,
                                 0};
            me.button.button = gui::MouseButton::LEFT;
            webrtc_window->PostMouseEvent(draw->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            me = gui::MouseEvent{gui::MouseEvent::Type::DRAG, 149, 362, 0};
            me.move.buttons = 1;
            webrtc_window->PostMouseEvent(draw->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            me = gui::MouseEvent{gui::MouseEvent::Type::DRAG, 209, 338, 0};
            me.move.buttons = 1;
            webrtc_window->PostMouseEvent(draw->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            me = gui::MouseEvent{gui::MouseEvent::Type::DRAG, 259, 319, 0};
            me.move.buttons = 1;
            webrtc_window->PostMouseEvent(draw->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            me = gui::MouseEvent{gui::MouseEvent::Type::BUTTON_UP, 263, 318, 0};
            me.button.button = gui::MouseButton::LEFT;
            webrtc_window->PostMouseEvent(draw->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    };
    draw.reset();  // so we don't hold onto the pointer after Run() cleans up
    std::thread thead(emulate_mouse_events);

    gui::Application::GetInstance().Run();
}

}  // namespace visualization
}  // namespace open3d
