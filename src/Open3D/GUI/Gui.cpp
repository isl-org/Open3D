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

#include "Gui.h"

namespace open3d {
namespace gui {

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

}  // namespace gui
}  // namespace open3d
