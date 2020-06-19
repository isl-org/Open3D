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

#include "open3d/visualization/gui/Gui.h"

#include <algorithm>

namespace open3d {
namespace visualization {
namespace gui {

Point::Point() : x(0), y(0) {}

Point::Point(int x_, int y_) : x(x_), y(y_) {}

// ----------------------------------------------------------------------------
Size::Size() : width(0), height(0) {}

Size::Size(int w, int h) : width(w), height(h) {}

// ----------------------------------------------------------------------------
Rect::Rect() : x(0), y(0), width(0), height(0) {}

Rect::Rect(int x_, int y_, int w_, int h_)
    : x(x_), y(y_), width(w_), height(h_) {}

int Rect::GetTop() const { return this->y; }

int Rect::GetBottom() const { return this->y + this->height; }

int Rect::GetLeft() const { return this->x; }

int Rect::GetRight() const { return this->x + this->width; }

bool Rect::Contains(int x, int y) const {
    return (x >= this->x && x <= GetRight() && y >= this->y &&
            y <= GetBottom());
}

bool Rect::Contains(const Point& pt) const { return Contains(pt.x, pt.y); }

Rect Rect::UnionedWith(const Rect& r) const {
    auto newX = std::min(this->x, r.x);
    auto newY = std::min(this->y, r.y);
    auto w = std::max(GetRight() - newX, r.GetRight() - newX);
    auto h = std::max(GetBottom() - newY, r.GetBottom() - newY);
    return Rect(newX, newY, w, h);
}

bool Rect::operator==(const Rect& other) const {
    return (this->x == other.x && this->y == other.y &&
            this->width == other.width && this->height == other.height);
}

bool Rect::operator!=(const Rect& other) const { return !operator==(other); }

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
