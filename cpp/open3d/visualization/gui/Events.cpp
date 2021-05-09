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

#include "open3d/visualization/gui/Events.h"

namespace open3d {
namespace visualization {
namespace gui {

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
