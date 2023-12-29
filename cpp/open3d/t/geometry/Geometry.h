// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/core/Device.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class Geometry
///
/// \brief The base geometry class.
class Geometry : public core::IsDevice {
public:
    /// \enum GeometryType
    ///
    /// \brief Specifies possible geometry types.
    enum class GeometryType {
        /// Unspecified geometry type.
        Unspecified = 0,
        /// PointCloud
        PointCloud = 1,
        /// VoxelGrid
        VoxelGrid = 2,
        /// Octree
        Octree = 3,
        /// LineSet
        LineSet = 4,
        /// MeshBase
        MeshBase = 5,
        /// TriangleMesh
        TriangleMesh = 6,
        /// HalfEdgeTriangleMesh
        HalfEdgeTriangleMesh = 7,
        /// Image
        Image = 8,
        /// RGBDImage
        RGBDImage = 9,
        /// TetraMesh
        TetraMesh = 10,
        /// OrientedBoundingBox
        OrientedBoundingBox = 11,
        /// AxisAlignedBoundingBox
        AxisAlignedBoundingBox = 12,
    };

public:
    virtual ~Geometry() {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type Specifies the type of geometry of the object constructed.
    /// \param dimension Specifies whether the dimension is 2D or 3D.
    Geometry(GeometryType type, int dimension)
        : geometry_type_(type), dimension_(dimension) {}

public:
    /// Clear all elements in the geometry.
    virtual Geometry& Clear() = 0;

    /// Returns true iff the geometry is empty.
    virtual bool IsEmpty() const = 0;

    /// Returns the device of the geometry.
    virtual core::Device GetDevice() const = 0;

    /// Returns one of registered geometry types.
    GeometryType GetGeometryType() const { return geometry_type_; }

    /// Returns whether the geometry is 2D or 3D.
    int Dimension() const { return dimension_; }

    std::string GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

private:
    /// Type of geometry from GeometryType.
    GeometryType geometry_type_ = GeometryType::Unspecified;

    /// Number of dimensions of the geometry.
    int dimension_;
    std::string name_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
