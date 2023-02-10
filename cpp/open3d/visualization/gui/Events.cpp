// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Events.h"

#include <json/json.h>

#include <cstdint>
#include <sstream>
#include <string>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace gui {

std::string MouseEvent::ToString() const {
    std::stringstream ss;
    ss << "MouseEvent{";
    ss << "type: ";
    if (type == Type::MOVE) {
        ss << "MOVE";
    } else if (type == Type::BUTTON_DOWN) {
        ss << "BUTTON_DOWN";
    } else if (type == Type::DRAG) {
        ss << "DRAG";
    } else if (type == Type::BUTTON_UP) {
        ss << "BUTTON_UP";
    } else if (type == Type::WHEEL) {
        ss << "WHEEL";
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
            ss << "NONE";
        } else if (button.button == MouseButton::LEFT) {
            ss << "LEFT";
        } else if (button.button == MouseButton::MIDDLE) {
            ss << "MIDDLE";
        } else if (button.button == MouseButton::RIGHT) {
            ss << "RIGHT";
        } else if (button.button == MouseButton::BUTTON4) {
            ss << "BUTTON4";
        } else if (button.button == MouseButton::BUTTON5) {
            ss << "BUTTON5";
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

bool MouseEvent::FromJson(const Json::Value& value) {
    if (!value.isObject()) {
        utility::LogWarning("MouseEvent::FromJson failed: Not an object.");
        return false;
    }

    std::string class_name = value.get("class_name", "").asString();
    if (class_name != "MouseEvent") {
        utility::LogWarning(
                "MouseEvent::FromJson failed: Incorrect class name {}.",
                class_name);
        return false;
    }

    std::string type_name = value.get("type", "").asString();
    if (type_name == "MOVE") {
        this->type = MouseEvent::Type::MOVE;
    } else if (type_name == "BUTTON_DOWN") {
        this->type = MouseEvent::Type::BUTTON_DOWN;
    } else if (type_name == "BUTTON_UP") {
        this->type = MouseEvent::Type::BUTTON_UP;
    } else if (type_name == "DRAG") {
        this->type = MouseEvent::Type::DRAG;
    } else if (type_name == "WHEEL") {
        this->type = MouseEvent::Type::WHEEL;
    } else {
        utility::LogWarning(
                "MouseEvent::FromJson failed: Incorrect type name {}.",
                type_name);
        return false;
    }
    this->x = value.get("x", 0).asInt();
    this->y = value.get("y", 0).asInt();
    this->modifiers = value.get("modifiers", 0).asInt();

    if (this->type == Type::MOVE || this->type == Type::DRAG) {
        this->move.buttons = value["move"].get("buttons", 0).asInt();
    } else if (this->type == Type::BUTTON_DOWN ||
               this->type == Type::BUTTON_UP) {
        std::string button_name = value["button"].get("button", "").asString();
        if (button_name == "NONE") {
            this->button.button = MouseButton::NONE;
        } else if (button_name == "LEFT") {
            this->button.button = MouseButton::LEFT;
        } else if (button_name == "MIDDLE") {
            this->button.button = MouseButton::MIDDLE;
        } else if (button_name == "RIGHT") {
            this->button.button = MouseButton::RIGHT;
        } else if (button_name == "BUTTON4") {
            this->button.button = MouseButton::BUTTON4;
        } else if (button_name == "BUTTON5") {
            this->button.button = MouseButton::BUTTON5;
        } else {
            utility::LogWarning(
                    "MouseEvent::FromJson failed: Incorrect button name {}.",
                    button_name);
            return false;
        }
        this->button.count = value["button"].get("count", 1).asInt();
    } else if (this->type == Type::WHEEL) {
        this->wheel.dx = value["wheel"].get("dx", 0.f).asFloat();
        this->wheel.dy = value["wheel"].get("dy", 0.f).asFloat();
        this->wheel.isTrackpad =
                value["wheel"].get("isTrackpad", false).asBool();
    }

    return true;
}

MouseEvent MouseEvent::MakeMoveEvent(const Type type,
                                     const int x,
                                     const int y,
                                     const int modifiers,
                                     const int buttons) {
    MouseEvent me;
    me.type = type;
    me.x = x;
    me.y = y;
    me.modifiers = modifiers;
    me.move.buttons = buttons;
    return me;
}

MouseEvent MouseEvent::MakeButtonEvent(const Type type,
                                       const int x,
                                       const int y,
                                       const int modifiers,
                                       const MouseButton button,
                                       const int count) {
    MouseEvent me;
    me.type = type;
    me.x = x;
    me.y = y;
    me.modifiers = modifiers;
    me.button.button = button;
    me.button.count = count;
    return me;
}

MouseEvent MouseEvent::MakeWheelEvent(const Type type,
                                      const int x,
                                      const int y,
                                      const int modifiers,
                                      const float dx,
                                      const float dy,
                                      const bool isTrackpad) {
    MouseEvent me;
    me.type = type;
    me.x = x;
    me.y = y;
    me.modifiers = modifiers;
    me.wheel.dx = dx;
    me.wheel.dy = dy;
    me.wheel.isTrackpad = isTrackpad;
    return me;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
