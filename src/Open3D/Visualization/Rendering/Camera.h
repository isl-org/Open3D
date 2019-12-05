// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

namespace open3d
{
namespace visualization
{

class Camera
{
public:
    enum eFovType {
        VERTICAL_FOV,
        HORIZONTAL_FOV
    };

    virtual void SetProjection(double fov, double aspect, double near, double far, eFovType fovType) = 0;

    virtual void LookAt(const Eigen::Vector3f& center,
                        const Eigen::Vector3f& eye,
                        const Eigen::Vector3f& up) = 0;

    virtual double GetNear() const = 0;
    virtual double GetFar() const = 0;
    virtual double GetFoV() const = 0;
    virtual double GetAspect() const = 0;

    virtual Eigen::Vector3f GetPosition() = 0;
    virtual Eigen::Vector3f GetForwardVector() = 0;
    virtual Eigen::Vector3f GetLeftVector() = 0;
    virtual Eigen::Vector3f GetUpVector() = 0;
};

}
}