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

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open3d/geometry/Geometry3D.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace geometry {

class PointCloud;
class TriangleMesh;

/// \class MeshBase
///
/// \brief MeshBash Class.
///
/// Triangle mesh contains vertices. Optionally, the mesh may also contain
/// vertex normals and vertex colors.
class MeshBase : public Geometry3D {
public:
    /// \brief Indicates the method that is used for mesh simplification if
    /// multiple vertices are combined to a single one.
    ///
    /// \param Average indicates that the average position is computed as
    /// output.
    /// \param Quadric indicates that the distance to the adjacent triangle
    /// planes is minimized. Cf. "Simplifying Surfaces with Color and Texture
    /// using Quadric Error Metrics" by Garland and Heckbert.
    enum class SimplificationContraction { Average, Quadric };

    /// \brief Indicates the scope of filter operations.
    ///
    /// \param All indicates that all properties (color, normal,
    /// vertex position) are filtered.
    /// \param Color indicates that only the colors are filtered.
    /// \param Normal indicates that only the normals are filtered.
    /// \param Vertex indicates that only the vertex positions are filtered.
    enum class FilterScope { All, Color, Normal, Vertex };

    /// Energy model that is minimized in the DeformAsRigidAsPossible method.
    /// \param Spokes is the original energy as formulated in
    /// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
    /// \param Smoothed adds a rotation smoothing term to the rotations.
    enum class DeformAsRigidAsPossibleEnergy { Spokes, Smoothed };

    /// \brief Default Constructor.
    MeshBase() : Geometry3D(Geometry::GeometryType::MeshBase) {}
    ~MeshBase() override {}

public:
    virtual MeshBase &Clear() override;
    virtual bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override;
    virtual MeshBase &Transform(const Eigen::Matrix4d &transformation) override;
    virtual MeshBase &Translate(const Eigen::Vector3d &translation,
                                bool relative = true) override;
    virtual MeshBase &Scale(const double scale,
                            const Eigen::Vector3d &center) override;
    virtual MeshBase &Rotate(const Eigen::Matrix3d &R,
                             const Eigen::Vector3d &center) override;

    MeshBase &operator+=(const MeshBase &mesh);
    MeshBase operator+(const MeshBase &mesh) const;

    /// Returns `True` if the mesh contains vertices.
    bool HasVertices() const { return vertices_.size() > 0; }

    /// Returns `True` if the mesh contains vertex normals.
    bool HasVertexNormals() const {
        return vertices_.size() > 0 &&
               vertex_normals_.size() == vertices_.size();
    }

    /// Returns `True` if the mesh contains vertex colors.
    bool HasVertexColors() const {
        return vertices_.size() > 0 &&
               vertex_colors_.size() == vertices_.size();
    }

    /// Normalize vertex normals to length 1.
    MeshBase &NormalizeNormals() {
        for (size_t i = 0; i < vertex_normals_.size(); i++) {
            vertex_normals_[i].normalize();
            if (std::isnan(vertex_normals_[i](0))) {
                vertex_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
        return *this;
    }

    /// \brief Assigns each vertex in the TriangleMesh the same color
    ///
    /// \param color RGB colors of vertices.
    MeshBase &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
        return *this;
    }

    /// Function that computes the convex hull of the triangle mesh using qhull
    std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
    ComputeConvexHull() const;

protected:
    // Forward child class type to avoid indirect nonvirtual base
    MeshBase(Geometry::GeometryType type) : Geometry3D(type) {}
    MeshBase(Geometry::GeometryType type,
             const std::vector<Eigen::Vector3d> &vertices)
        : Geometry3D(type), vertices_(vertices) {}

public:
    /// Vertex coordinates.
    std::vector<Eigen::Vector3d> vertices_;
    /// Vertex normals.
    std::vector<Eigen::Vector3d> vertex_normals_;
    /// RGB colors of vertices.
    std::vector<Eigen::Vector3d> vertex_colors_;
};

}  // namespace geometry
}  // namespace open3d
