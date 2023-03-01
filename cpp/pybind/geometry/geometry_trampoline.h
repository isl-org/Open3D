// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    OrientedBoundingBox GetMinimalOrientedBoundingBox(
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
