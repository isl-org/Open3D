// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
