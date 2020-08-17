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

#include "open3d/visualization/gui/Color.h"

#include <algorithm>
#include <cmath>

namespace open3d {
namespace visualization {
namespace gui {

Color::Color() : rgba_{0.0f, 0.0f, 0.0f, 1.0f} {}

Color::Color(float r, float g, float b, float a /*= 1.0*/)
    : rgba_{r, g, b, a} {}

Color::Color(const Eigen::Vector3f& rgb)
    : rgba_{rgb.x(), rgb.y(), rgb.z(), 1.0f} {}

bool Color::operator==(const Color& rhs) const {
    return (this->rgba_[0] == rhs.rgba_[0] && this->rgba_[1] == rhs.rgba_[1] &&
            this->rgba_[2] == rhs.rgba_[2] && this->rgba_[3] == rhs.rgba_[3]);
}

bool Color::operator!=(const Color& rhs) const {
    return !this->operator==(rhs);
}

float Color::GetRed() const { return rgba_[0]; }
float Color::GetGreen() const { return rgba_[1]; }
float Color::GetBlue() const { return rgba_[2]; }
float Color::GetAlpha() const { return rgba_[3]; }

void Color::SetColor(const float r,
                     const float g,
                     const float b,
                     const float a /*= 1.0 */) {
    rgba_[0] = r;
    rgba_[1] = g;
    rgba_[2] = b;
    rgba_[3] = a;
}

const float* Color::GetPointer() const { return rgba_; }
float* Color::GetMutablePointer() { return rgba_; }

Color Color::Lightened(float amount) {
    amount = std::max(0.0f, std::min(1.0f, amount));
    return Color((1.0f - amount) * GetRed() + amount * 1.0f,
                 (1.0f - amount) * GetGreen() + amount * 1.0f,
                 (1.0f - amount) * GetBlue() + amount * 1.0f, GetAlpha());
}

unsigned int Color::ToABGR32() const {
    unsigned int a = (unsigned int)std::round(GetAlpha() * 255.0f);
    unsigned int b = (unsigned int)std::round(GetBlue() * 255.0f);
    unsigned int g = (unsigned int)std::round(GetGreen() * 255.0f);
    unsigned int r = (unsigned int)std::round(GetRed() * 255.0f);
    return ((a << 24) | (b << 16) | (g << 8) | r);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
