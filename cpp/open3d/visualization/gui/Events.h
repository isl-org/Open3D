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

#pragma once

#include <cstdint>
#include <sstream>
#include <string>

#include "open3d/utility/IJsonConvertible.h"

/// @cond
namespace Json {
class Value;
}  // namespace Json
/// @endcond

namespace open3d {
namespace visualization {
namespace gui {

enum class MouseButton {
    NONE = 0,
    LEFT = (1 << 0),
    MIDDLE = (1 << 1),
    RIGHT = (1 << 2),
    BUTTON4 = (1 << 3),
    BUTTON5 = (1 << 4)
};

// The key modifiers are labeled by functionality; for instance,
// Ctrl on Windows and Command on macOS have roughly the same functionality.
enum class KeyModifier {
    NONE = 0,
    SHIFT = (1 << 0),
    CTRL = (1 << 1),  // win/linux: ctrl, macOS: command
    ALT = (1 << 2),   // win/linux: alt, macOS: ctrl
    META = (1 << 3),  // win/linux: windows key, macOS: option
};

struct MouseEvent {
    enum Type { MOVE, BUTTON_DOWN, DRAG, BUTTON_UP, WHEEL };

    static MouseEvent MakeMoveEvent(const Type type,
                                    const int x,
                                    const int y,
                                    const int modifiers,
                                    const int buttons);

    static MouseEvent MakeButtonEvent(const Type type,
                                      const int x,
                                      const int y,
                                      const int modifiers,
                                      const MouseButton button,
                                      const int count);

    static MouseEvent MakeWheelEvent(const Type type,
                                     const int x,
                                     const int y,
                                     const int modifiers,
                                     const float dx,
                                     const float dy,
                                     const bool isTrackpad);

    Type type;
    int x;
    int y;
    int modifiers;  // KeyModifiers ORed together
    union {
        struct {
            int buttons;  // MouseButtons ORed together
        } move;           // includes drag
        struct {
            MouseButton button;
            int count;
        } button;
        struct {
            float dx;  // macOS gives fractional values, and is required
            float dy;  //   for the buttery-smooth trackpad scrolling on macOS
            bool isTrackpad;
        } wheel;
    };

    bool FromJson(const Json::Value &value);
    std::string ToString() const;
};

struct TickEvent {
    double dt;
};

enum KeyName {
    KEY_NONE = 0,
    KEY_BACKSPACE = 8,
    KEY_TAB = 9,
    KEY_ENTER = 10,
    KEY_ESCAPE = 27,
    KEY_SPACE = 32,
    KEY_EXCLAMATION = 33,
    KEY_DOUBLE_QUOTE = 34,
    KEY_HASH = 35,
    KEY_DOLLAR_SIGN = 36,
    KEY_PERCENT = 37,
    KEY_AMPERSAND = 38,
    KEY_SINGLE_QUOTE = 39,
    KEY_LEFT_PAREN = 40,
    KEY_RIGHT_PAREN = 41,
    KEY_ASTERISK = 42,
    KEY_PLUS = 43,
    KEY_COMMA = 44,
    KEY_MINUS = 45,
    KEY_PERIOD = 46,
    KEY_SLASH = 47,
    KEY_0 = 48,
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_COLON = 58,
    KEY_SEMICOLON = 59,
    KEY_LESS_THAN = 60,
    KEY_EQUALS = 61,
    KEY_GREATER_THAN = 62,
    KEY_QUESTION_MARK = 63,
    KEY_AT = 64,
    KEY_LEFT_BRACKET = 91,
    KEY_BACKSLASH = 92,
    KEY_RIGHT_BRACKET = 93,
    KEY_CARET = 94,
    KEY_UNDERSCORE = 95,
    KEY_BACKTICK = 96,
    KEY_A = 97,
    KEY_B,
    KEY_C,
    KEY_D,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_M,
    KEY_N,
    KEY_O,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_S,
    KEY_T,
    KEY_U,
    KEY_V,
    KEY_W,
    KEY_X,
    KEY_Y,
    KEY_Z,
    KEY_LEFT_BRACE = 123,
    KEY_PIPE = 124,
    KEY_RIGHT_BRACE = 125,
    KEY_TILDE = 126,
    KEY_DELETE = 127,
    KEY_LSHIFT = 256,
    KEY_RSHIFT,
    KEY_LCTRL,
    KEY_RCTRL,
    KEY_ALT,
    KEY_META,
    KEY_CAPSLOCK,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    KEY_DOWN,
    KEY_INSERT,
    KEY_HOME,
    KEY_END,
    KEY_PAGEUP,
    KEY_PAGEDOWN,
    KEY_F1 = 290,
    KEY_F2,
    KEY_F3,
    KEY_F4,
    KEY_F5,
    KEY_F6,
    KEY_F7,
    KEY_F8,
    KEY_F9,
    KEY_F10,
    KEY_F11,
    KEY_F12,
    KEY_UNKNOWN = 1000
};

struct KeyEvent {
    enum Type { DOWN, UP };
    Type type;
    // This is the actual key that was pressed, not the character that
    // was generated (use TextInputEvent for that). Values correspond
    // to ASCII values where applicable.
    uint32_t key;
    bool isRepeat;
};

struct TextInputEvent {
    const char *utf8;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
