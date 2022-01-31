// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/utility/Logging.h"
#include "pybind/visualization/gui/gui.h"
#include "pybind11/functional.h"

namespace open3d {
namespace visualization {
namespace gui {

void pybind_gui_events(py::module& m) {
    py::enum_<MouseButton> buttons(m, "MouseButton", "Mouse button identifiers",
                                   py::arithmetic());
    buttons.value("NONE", MouseButton::NONE)
            .value("LEFT", MouseButton::LEFT)
            .value("MIDDLE", MouseButton::MIDDLE)
            .value("RIGHT", MouseButton::RIGHT)
            .value("BUTTON4", MouseButton::BUTTON4)
            .value("BUTTON5", MouseButton::BUTTON5)
            .export_values();

    py::enum_<KeyModifier> key_mod(m, "KeyModifier", "Key modifier identifiers",
                                   py::arithmetic());
    key_mod.value("NONE", KeyModifier::NONE)
            .value("SHIFT", KeyModifier::SHIFT)
            .value("CTRL", KeyModifier::CTRL)
            .value("ALT", KeyModifier::ALT)
            .value("META", KeyModifier::META)
            .export_values();

    py::class_<MouseEvent> mouse_event(m, "MouseEvent",
                                       "Object that stores mouse events");
    py::enum_<MouseEvent::Type> mouse_event_type(mouse_event, "Type",
                                                 py::arithmetic());
    mouse_event_type.value("MOVE", MouseEvent::Type::MOVE)
            .value("BUTTON_DOWN", MouseEvent::Type::BUTTON_DOWN)
            .value("DRAG", MouseEvent::Type::DRAG)
            .value("BUTTON_UP", MouseEvent::Type::BUTTON_UP)
            .value("WHEEL", MouseEvent::Type::WHEEL)
            .export_values();
    mouse_event.def_readwrite("type", &MouseEvent::type, "Mouse event type")
            .def_readwrite("x", &MouseEvent::x,
                           "x coordinate  of the mouse event")
            .def_readwrite("y", &MouseEvent::y,
                           "y coordinate of the mouse event")
            .def(
                    "is_modifier_down",
                    [](const MouseEvent& e, KeyModifier mod) {
                        return ((e.modifiers & int(mod)) != 0);
                    },
                    "Convenience function to more easily deterimine if a "
                    "modifier "
                    "key is down")
            .def(
                    "is_button_down",
                    [](const MouseEvent& e, MouseButton b) {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            return false;
                        } else if (e.type == MouseEvent::Type::BUTTON_DOWN) {
                            return (e.button.button == b);
                        } else {
                            return ((e.move.buttons & int(b)) != 0);
                        }
                    },
                    "Convenience function to more easily deterimine if a mouse "
                    "button is pressed")
            .def_readwrite("modifiers", &MouseEvent::modifiers,
                           "ORed mouse modifiers")
            .def_property(
                    "buttons",
                    [](const MouseEvent& e) -> int {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            return int(MouseButton::NONE);
                        } else if (e.type == MouseEvent::Type::BUTTON_DOWN) {
                            return int(e.button.button);
                        } else {
                            return e.move.buttons;
                        }
                    },
                    [](MouseEvent& e, int new_value) {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            ;  // no button value for wheel events
                        } else if (e.type == MouseEvent::Type::BUTTON_DOWN) {
                            e.button.button = MouseButton(new_value);
                        } else {
                            e.move.buttons = new_value;
                        }
                    },
                    "ORed mouse buttons")
            .def_property(
                    "wheel_dx",
                    [](const MouseEvent& e) -> int {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            return e.wheel.dx;
                        } else {
                            return 0;
                        }
                    },
                    [](MouseEvent& e, int new_value) {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            e.wheel.dx = new_value;
                        } else {
                            utility::LogWarning(
                                    "Cannot set MouseEvent.wheel_dx unless "
                                    "MouseEvent.type == MouseEvent.Type.WHEEL");
                        }
                    },
                    "Mouse wheel horizontal motion")
            .def_property(
                    "wheel_dy",
                    [](const MouseEvent& e) -> int {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            return e.wheel.dy;
                        } else {
                            return 0;
                        }
                    },
                    [](MouseEvent& e, int new_value) {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            e.wheel.dy = new_value;
                        } else {
                            utility::LogWarning(
                                    "Cannot set MouseEvent.wheel_dy unless "
                                    "MouseEvent.type == MouseEvent.Type.WHEEL");
                        }
                    },
                    "Mouse wheel vertical motion")
            .def_property(
                    "wheel_is_trackpad",
                    [](const MouseEvent& e) -> bool {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            return e.wheel.isTrackpad;
                        } else {
                            return false;
                        }
                    },
                    [](MouseEvent& e, bool new_value) {
                        if (e.type == MouseEvent::Type::WHEEL) {
                            e.wheel.isTrackpad = new_value;
                        } else {
                            utility::LogWarning(
                                    "Cannot set MouseEvent.wheel_is_trackpad "
                                    "unless MouseEvent.type == "
                                    "MouseEvent.Type.WHEEL");
                        }
                    },
                    "Is mouse wheel event from a trackpad");

    py::enum_<KeyName> key_name(m, "KeyName",
                                "Names of keys. Used by KeyEvent.key");
    key_name.value("NONE", KeyName::KEY_NONE)
            .value("BACKSPACE", KeyName::KEY_BACKSPACE)
            .value("TAB", KeyName::KEY_TAB)
            .value("ENTER", KeyName::KEY_ENTER)
            .value("ESCAPE", KeyName::KEY_ESCAPE)
            .value("SPACE", KeyName::KEY_SPACE)
            .value("EXCLAMATION_MARK", KeyName::KEY_EXCLAMATION)
            .value("DOUBLE_QUOTE", KeyName::KEY_DOUBLE_QUOTE)
            .value("HASH", KeyName::KEY_HASH)
            .value("DOLLAR_SIGN", KeyName::KEY_DOLLAR_SIGN)
            .value("PERCENT", KeyName::KEY_PERCENT)
            .value("AMPERSAND", KeyName::KEY_AMPERSAND)
            .value("QUOTE", KeyName::KEY_SINGLE_QUOTE)
            .value("LEFT_PAREN", KeyName::KEY_LEFT_PAREN)
            .value("RIGHT_PAREN", KeyName::KEY_RIGHT_PAREN)
            .value("ASTERISK", KeyName::KEY_ASTERISK)
            .value("PLUS", KeyName::KEY_PLUS)
            .value("COMMA", KeyName::KEY_COMMA)
            .value("MINUS", KeyName::KEY_MINUS)
            .value("PERIOD", KeyName::KEY_PERIOD)
            .value("SLASH", KeyName::KEY_SLASH)
            .value("ZERO", KeyName::KEY_0)
            .value("ONE", KeyName::KEY_1)
            .value("TWO", KeyName::KEY_2)
            .value("THREE", KeyName::KEY_3)
            .value("FOUR", KeyName::KEY_4)
            .value("FIVE", KeyName::KEY_5)
            .value("SIX", KeyName::KEY_6)
            .value("SEVEN", KeyName::KEY_7)
            .value("EIGHT", KeyName::KEY_8)
            .value("NINE", KeyName::KEY_9)
            .value("COLON", KeyName::KEY_COLON)
            .value("SEMICOLON", KeyName::KEY_SEMICOLON)
            .value("LESS_THAN", KeyName::KEY_LESS_THAN)
            .value("EQUALS", KeyName::KEY_EQUALS)
            .value("GREATER_THAN", KeyName::KEY_GREATER_THAN)
            .value("QUESTION_MARK", KeyName::KEY_QUESTION_MARK)
            .value("AT", KeyName::KEY_AT)
            .value("LEFT_BRACKET", KeyName::KEY_LEFT_BRACKET)
            .value("BACKSLASH", KeyName::KEY_BACKSLASH)
            .value("RIGHT_BRACKET", KeyName::KEY_RIGHT_BRACKET)
            .value("CARET", KeyName::KEY_CARET)
            .value("UNDERSCORE", KeyName::KEY_UNDERSCORE)
            .value("BACKTICK", KeyName::KEY_BACKTICK)
            .value("A", KeyName::KEY_A)
            .value("B", KeyName::KEY_B)
            .value("C", KeyName::KEY_C)
            .value("D", KeyName::KEY_D)
            .value("E", KeyName::KEY_E)
            .value("F", KeyName::KEY_F)
            .value("G", KeyName::KEY_G)
            .value("H", KeyName::KEY_H)
            .value("I", KeyName::KEY_I)
            .value("J", KeyName::KEY_J)
            .value("K", KeyName::KEY_K)
            .value("L", KeyName::KEY_L)
            .value("M", KeyName::KEY_M)
            .value("N", KeyName::KEY_N)
            .value("O", KeyName::KEY_O)
            .value("P", KeyName::KEY_P)
            .value("Q", KeyName::KEY_Q)
            .value("R", KeyName::KEY_R)
            .value("S", KeyName::KEY_S)
            .value("T", KeyName::KEY_T)
            .value("U", KeyName::KEY_U)
            .value("V", KeyName::KEY_V)
            .value("W", KeyName::KEY_W)
            .value("X", KeyName::KEY_X)
            .value("Y", KeyName::KEY_Y)
            .value("Z", KeyName::KEY_Z)
            .value("LEFT_BRACE", KeyName::KEY_LEFT_BRACE)
            .value("PIPE", KeyName::KEY_PIPE)
            .value("RIGHT_BRACE", KeyName::KEY_RIGHT_BRACE)
            .value("TILDE", KeyName::KEY_TILDE)
            .value("DELETE", KeyName::KEY_DELETE)
            .value("LEFT_SHIFT", KeyName::KEY_LSHIFT)
            .value("RIGHT_SHIFT", KeyName::KEY_RSHIFT)
            .value("LEFT_CONTROL", KeyName::KEY_LCTRL)
            .value("RIGHT_CONTROL", KeyName::KEY_RCTRL)
            .value("ALT", KeyName::KEY_ALT)
            .value("META", KeyName::KEY_META)
            .value("CAPS_LOCK", KeyName::KEY_CAPSLOCK)
            .value("LEFT", KeyName::KEY_LEFT)
            .value("RIGHT", KeyName::KEY_RIGHT)
            .value("UP", KeyName::KEY_UP)
            .value("DOWN", KeyName::KEY_DOWN)
            .value("INSERT", KeyName::KEY_INSERT)
            .value("HOME", KeyName::KEY_HOME)
            .value("END", KeyName::KEY_END)
            .value("PAGE_UP", KeyName::KEY_PAGEUP)
            .value("PAGE_DOWN", KeyName::KEY_PAGEDOWN)
            .value("F1", KeyName::KEY_F1)
            .value("F2", KeyName::KEY_F2)
            .value("F3", KeyName::KEY_F3)
            .value("F4", KeyName::KEY_F4)
            .value("F5", KeyName::KEY_F5)
            .value("F6", KeyName::KEY_F6)
            .value("F7", KeyName::KEY_F7)
            .value("F8", KeyName::KEY_F8)
            .value("F9", KeyName::KEY_F9)
            .value("F10", KeyName::KEY_F10)
            .value("F11", KeyName::KEY_F11)
            .value("F12", KeyName::KEY_F12)
            .value("UNKNOWN", KeyName::KEY_UNKNOWN)
            .export_values();

    py::class_<KeyEvent> key_event(m, "KeyEvent",
                                   "Object that stores mouse events");
    py::enum_<KeyEvent::Type> key_event_type(key_event, "Type",
                                             py::arithmetic());
    key_event_type.value("DOWN", KeyEvent::Type::DOWN)
            .value("UP", KeyEvent::Type::UP)
            .export_values();
    key_event.def_readwrite("type", &KeyEvent::type, "Key event type")
            .def_readwrite("key", &KeyEvent::key,
                           "This is the actual key that was pressed, not the "
                           "character generated by the key. This event is "
                           "not suitable for text entry")
            .def_readwrite("is_repeat", &KeyEvent::isRepeat,
                           "True if this key down event comes from a key "
                           "repeat");
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
