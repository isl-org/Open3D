// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
