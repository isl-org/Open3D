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

#include <Eigen/Geometry>

namespace open3d {
namespace visualization {
namespace gui {

class Color {
public:
    Color();
    Color(float r, float g, float b, float a = 1.0);
    Color(const Eigen::Vector3f& rgb);  // not explicit: want auto-convert

    float GetRed() const;
    float GetGreen() const;
    float GetBlue() const;
    float GetAlpha() const;

    void SetColor(float r, float g, float b, float a = 1.0);

    const float* GetPointer() const;
    float* GetMutablePointer();

    /// Returns a lighter color.
    /// \param amount is between 0 and 1, with 0 being the same color and
    /// 1 being white.
    Color Lightened(float amount);

    unsigned int ToABGR32() const;

    bool operator==(const Color& rhs) const;
    bool operator!=(const Color& rhs) const;

private:
    float rgba_[4];
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
