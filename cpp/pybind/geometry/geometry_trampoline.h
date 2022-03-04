// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/Geometry.h"
#include "open3d/geometry/Geometry2D.h"
#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "pybind/geometry/geometry.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace geometry {

template <class GeometryBase = Geometry>
class PyGeometry : public GeometryBase {
public:
    using GeometryBase::GeometryBase;
    GeometryBase& Clear() override {
        PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, );
    }
    bool IsEmpty() const override {
        PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
    }
};

template <class Geometry3DBase = Geometry3D>
class PyGeometry3D : public PyGeometry<Geometry3DBase> {
public:
    using PyGeometry<Geometry3DBase>::PyGeometry;
    Eigen::Vector3d GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
    }
    Eigen::Vector3d GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
    }
    Eigen::Vector3d GetCenter() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
    }
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override {
        PYBIND11_OVERLOAD_PURE(AxisAlignedBoundingBox, Geometry3DBase, );
    }
    OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override {
        PYBIND11_OVERLOAD_PURE(OrientedBoundingBox, Geometry3DBase, robust);
    }
    Geometry3DBase& Transform(const Eigen::Matrix4d& transformation) override {
        PYBIND11_OVERLOAD_PURE(Geometry3DBase&, Geometry3DBase, transformation);
    }
};

template <class Geometry2DBase = Geometry2D>
class PyGeometry2D : public PyGeometry<Geometry2DBase> {
public:
    using PyGeometry<Geometry2DBase>::PyGeometry;
    Eigen::Vector2d GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, Geometry2DBase, );
    }
    Eigen::Vector2d GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, Geometry2DBase, );
    }
};

}  // namespace geometry
}  // namespace open3d
