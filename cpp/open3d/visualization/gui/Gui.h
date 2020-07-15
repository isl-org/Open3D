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

#pragma once

namespace open3d {
namespace visualization {
namespace gui {

struct Point {
    int x;
    int y;

    Point();
    Point(int x_, int y_);
};

struct Size {
    int width;
    int height;

    Size();
    Size(int w, int h);
};

struct Rect {
    int x;
    int y;
    int width;
    int height;

    Rect();
    Rect(int x_, int y_, int w_, int h_);

    int GetTop() const;
    int GetBottom() const;
    int GetLeft() const;
    int GetRight() const;

    bool Contains(int x, int y) const;
    bool Contains(const Point& pt) const;

    Rect UnionedWith(const Rect& r) const;

    bool operator==(const Rect& other) const;
    bool operator!=(const Rect& other) const;
};

enum class BorderShape { NONE = 0, RECT, ROUNDED_RECT };

enum class Alignment : unsigned int {
    LEFT = 1,
    HCENTER = 2,
    RIGHT = 3,
    TOP = (1 << 4),
    VCENTER = (2 << 4),
    BOTTOM = (3 << 4),
    CENTER = (2 | (2 << 4))
};
constexpr Alignment operator|(Alignment x, Alignment y) {
    return Alignment((unsigned int)(x) | (unsigned int)(y));
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
