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

#include "open3d/visualization/gui/Events.h"

#include <json/json.h>

#include <cstdint>
#include <sstream>
#include <string>

#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace gui {

std::string MouseEvent::ToString() const {
    std::stringstream ss;
    ss << "MouseEvent{";
    ss << "type: ";
    if (type == Type::MOVE) {
        ss << "Type::MOVE";
    } else if (type == Type::BUTTON_DOWN) {
        ss << "Type::BUTTON_DOWN";
    } else if (type == Type::DRAG) {
        ss << "Type::DRAG";
    } else if (type == Type::BUTTON_UP) {
        ss << "Type::BUTTON_UP";
    } else if (type == Type::WHEEL) {
        ss << "Type::WHEEL";
    } else {
        ss << "ERROR";
    }
    ss << ", x: " << x;
    ss << ", y: " << y;
    ss << ", modifiers: " << modifiers;
    if (type == Type::MOVE || type == Type::DRAG) {
        ss << ", move.buttons : " << move.buttons;
    } else if (type == Type::BUTTON_DOWN || type == Type::BUTTON_UP) {
        ss << ", button.button: ";
        if (button.button == MouseButton::NONE) {
            ss << "MouseButton::NONE";
        } else if (button.button == MouseButton::LEFT) {
            ss << "MouseButton::LEFT";
        } else if (button.button == MouseButton::MIDDLE) {
            ss << "MouseButton::MIDDLE";
        } else if (button.button == MouseButton::RIGHT) {
            ss << "MouseButton::RIGHT";
        } else if (button.button == MouseButton::BUTTON4) {
            ss << "MouseButton::BUTTON4";
        } else if (button.button == MouseButton::BUTTON5) {
            ss << "MouseButton::BUTTON5";
        } else {
            ss << "ERROR";
        }
        ss << ", button.count: " << button.count;
    } else if (type == Type::WHEEL) {
        ss << ", wheel.dx: " << wheel.dx;
        ss << ", wheel.dy: " << wheel.dy;
        ss << ", wheel.isTrackpad: " << wheel.isTrackpad;
    }
    ss << "}";
    return ss.str();
}

bool MouseEvent::ToJson(Json::Value &value) const {
    // Unimplemented.
    return true;
}

bool MouseEvent::FromJson(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "MouseEvent::ConvertFromJsonValue failed: not an object.");
        return false;
    }
    return true;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
