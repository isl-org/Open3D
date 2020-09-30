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

namespace open3d {
namespace t {
namespace geometry {

/// \class Geometry
///
/// \brief The base geometry class.
class Geometry {
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

    /// Returns one of registered geometry types.
    GeometryType GetGeometryType() const { return geometry_type_; }

    /// Returns whether the geometry is 2D or 3D.
    int Dimension() const { return dimension_; }

private:
    /// Type of geometry from GeometryType.
    GeometryType geometry_type_ = GeometryType::Unspecified;

    /// Number of dimensions of the geometry.
    int dimension_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
