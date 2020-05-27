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

#include "Open3D/Core/TensorList.h"
#include "Open3D/TGeometry/Geometry.h"

namespace open3d {
namespace tgeometry {

/// \class Geometry3D
///
/// \brief The base geometry class for 3D geometries.
///
/// Main class for 3D geometries.
class Geometry3D : public Geometry {
public:
    ~Geometry3D() override {}

protected:
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    Geometry3D& Clear() override = 0;

    bool IsEmpty() const override = 0;

    /// Returns min bounds for geometry coordinates.
    virtual Tensor GetMinBound() const = 0;

    /// Returns max bounds for geometry coordinates.
    virtual Tensor GetMaxBound() const = 0;

    /// Returns the center of the geometry coordinates.
    virtual Tensor GetCenter() const = 0;

    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    ///
    /// \param transformation A Tensor of shape (4, 4). The transformation
    /// matrix,
    virtual Geometry3D& Transform(const Tensor& transformation) = 0;

    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A Tensor of shape (3,). The translation offset to the
    /// geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual Geometry3D& Translate(const Tensor& translation,
                                  bool relative = true) = 0;

    /// \brief Apply scaling to the geometry coordinates.
    /// Given a scaling factor \f$s\f$, and center \f$c\f$, a given point
    /// \f$p\f$ is transformed according to \f$s (p - c) + c\f$.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center A Tensor of shape (3,). The scale center used to resize
    /// the geometry.
    virtual Geometry3D& Scale(double scale, const Tensor& center) = 0;

    /// \brief Apply rotation to the geometry coordinates and normals.
    /// Given a rotation matrix \f$R\f$, and center \f$c\f$, a given point
    /// \f$p\f$ is transformed according to \f$R (p - c) + c\f$.
    ///
    /// \param R A Tensor of shape (3, 3). The rotation matrix.
    /// \param center A Tensor of shape (3,). Rotation center that is used for
    /// the rotation.
    virtual Geometry3D& Rotate(const Tensor& R, const Tensor& center) = 0;
};

}  // namespace tgeometry
}  // namespace open3d
