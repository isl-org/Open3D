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

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <memory>
#include <vector>

#include "Open3D/Geometry/MeshBase.h"
#include "Open3D/Utility/Eigen.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace geometry {

class PointCloud;
class TriangleMesh;

/// \class TetraMesh
///
/// \brief Tetra mesh contains vertices and tetrahedra represented by the
/// indices to the vertices.
class TetraMesh : public MeshBase {
public:
    /// \brief Default Constructor.
    TetraMesh() : MeshBase(Geometry::GeometryType::TetraMesh) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param vertices Vertex coordinates.
    /// \param tetras List of tetras denoted by the index of points forming the
    /// tetra.
    TetraMesh(const std::vector<Eigen::Vector3d> &vertices,
              const std::vector<Eigen::Vector4i, utility::Vector4i_allocator>
                      &tetras)
        : MeshBase(Geometry::GeometryType::TetraMesh, vertices),
          tetras_(tetras) {}
    ~TetraMesh() override {}

public:
    TetraMesh &Clear() override;

public:
    TetraMesh &operator+=(const TetraMesh &mesh);
    TetraMesh operator+(const TetraMesh &mesh) const;

    /// \brief Function that removes duplicated verties, i.e., vertices that
    /// have identical coordinates.
    TetraMesh &RemoveDuplicatedVertices();

    /// \brief Function that removes duplicated tetrahedra, i.e., removes
    /// tetrahedra that reference the same four vertices, independent of their
    /// order.
    TetraMesh &RemoveDuplicatedTetras();

    /// \brief This function removes vertices from the tetra mesh that are not
    /// referenced in any tetrahedron of the mesh.
    TetraMesh &RemoveUnreferencedVertices();

    /// \brief Function that removes degenerate tetrahedra, i.e., tetrahedra
    /// that reference a single vertex multiple times in a single tetrahedron.
    /// They are usually the product of removing duplicated vertices.
    TetraMesh &RemoveDegenerateTetras();

    /// Returns `true` if the mesh contains tetras.
    bool HasTetras() const {
        return vertices_.size() > 0 && tetras_.size() > 0;
    }

    /// \brief Function to extract a triangle mesh of the specified iso-surface
    /// at a level This method applies primal contouring and generates triangles
    /// for each tetrahedron.
    ///
    /// \param level specifies the level.
    /// \param values specifies values per-vertex.
    std::shared_ptr<TriangleMesh> ExtractTriangleMesh(
            const std::vector<double> &values, double level);

    /// \brief Function that creates a tetrahedral mesh (TetraMeshFactory.cpp).
    /// from a point cloud.
    ///
    /// The method creates the Delaunay triangulation
    /// using the implementation from Qhull.
    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    CreateFromPointCloud(const PointCloud &point_cloud);

protected:
    // Forward child class type to avoid indirect nonvirtual base
    TetraMesh(Geometry::GeometryType type) : MeshBase(type) {}

public:
    /// List of tetras denoted by the index of points forming the tetra.
    std::vector<Eigen::Vector4i, utility::Vector4i_allocator> tetras_;
};

}  // namespace geometry
}  // namespace open3d
