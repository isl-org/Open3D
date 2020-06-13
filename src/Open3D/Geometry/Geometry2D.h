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

#include <Eigen/Core>

#include "Open3D/Geometry/Geometry.h"

namespace open3d {
namespace geometry {

/// \class Geometry2D
///
/// \brief The base geometry class for 2D geometries.
///
/// Main class for 2D geometries, Derives all data from Geometry Base class.
class Geometry2D : public Geometry {
public:
    ~Geometry2D() override {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type  type of object based on GeometryType
    Geometry2D(GeometryType type) : Geometry(type, 2) {}

public:
    Geometry& Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector2d GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector2d GetMaxBound() const = 0;
};

}  // namespace geometry
}  // namespace open3d
