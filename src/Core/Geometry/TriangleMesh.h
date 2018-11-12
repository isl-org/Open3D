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

#include <vector>
#include <memory>
#include <Eigen/Core>

#include <Core/Geometry/Geometry3D.h>

namespace open3d {

class TriangleMesh : public Geometry3D
{
public:
    TriangleMesh() : Geometry3D(Geometry::GeometryType::TriangleMesh) {};
    ~TriangleMesh() override {};

public:
    void Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    void Transform(const Eigen::Matrix4d &transformation) override;

public:
    TriangleMesh &operator+=(const TriangleMesh &mesh);
    TriangleMesh operator+(const TriangleMesh &mesh) const;

    /// Function to compute triangle normals, usually called before rendering
    void ComputeTriangleNormals(bool normalized = true);

    /// Function to compute vertex normals, usually called before rendering
    void ComputeVertexNormals(bool normalized = true);

    /// Function to remove duplicated and non-manifold vertices/triangles
    void Purge();

protected:
    void RemoveDuplicatedVertices();
    void RemoveDuplicatedTriangles();
    void RemoveNonManifoldVertices();
    void RemoveNonManifoldTriangles();

public:
    bool HasVertices() const {
        return vertices_.size() > 0;
    }

    bool HasTriangles() const {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    bool HasVertexNormals() const {
        return vertices_.size() > 0 &&
                vertex_normals_.size() == vertices_.size();
    }

    bool HasVertexColors() const {
        return vertices_.size() > 0 &&
                vertex_colors_.size() == vertices_.size();
    }

    bool HasTriangleNormals() const {
        return HasTriangles() &&
                triangles_.size() == triangle_normals_.size();
    }

    void NormalizeNormals() {
        for (size_t i = 0; i < vertex_normals_.size(); i++) {
            vertex_normals_[i].normalize();
            if (std::isnan(vertex_normals_[i](0))) {
                vertex_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
        for (size_t i = 0; i < triangle_normals_.size(); i++) {
            triangle_normals_[i].normalize();
            if (std::isnan(triangle_normals_[i](0))) {
                triangle_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
    }

    void PaintUniformColor(const Eigen::Vector3d &color) {
        vertex_colors_.resize(vertices_.size());
        for (size_t i = 0; i < vertices_.size(); i++) {
            vertex_colors_[i] = color;
        }
    }

public:
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> vertex_normals_;
    std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> triangle_normals_;
};

/// Function to select points from \param input TriangleMesh into
/// \return output TriangleMesh
/// Vertices with indices in \param indices are selected.
std::shared_ptr<TriangleMesh> SelectDownSample(const TriangleMesh &input,
        const std::vector<size_t> &indices);

/// Function to crop \param input tringlemesh into output tringlemesh
/// All points with coordinates less than \param min_bound or larger than
/// \param max_bound are clipped.
std::shared_ptr<TriangleMesh> CropTriangleMesh(const TriangleMesh &input,
        const Eigen::Vector3d &min_bound, const Eigen::Vector3d &max_bound);

/// Factory function to create a sphere mesh (TriangleMeshFactory.cpp)
/// The sphere with \param radius will be centered at (0, 0, 0).
/// Its axis is aligned with z-axis.
/// The longitudes will be split into \param resolution segments.
/// The latitudes will be split into \param resolution * 2 segments.
std::shared_ptr<TriangleMesh> CreateMeshSphere(double radius = 1.0,
        int resolution = 20);

/// Factory function to create a cylinder mesh (TriangleMeshFactory.cpp)
/// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0, height/2).
/// The circle with \param radius will be split into \param resolution segments.
/// The \param height will be split into \param split segments.
std::shared_ptr<TriangleMesh> CreateMeshCylinder(double radius = 1.0,
        double height = 2.0, int resolution = 20, int split = 4);

/// Factory function to create a cone mesh (TriangleMeshFactory.cpp)
/// The axis of the cone will be from (0, 0, 0) to (0, 0, \param height).
/// The circle with \param radius will be split into \param resolution segments.
/// The height will be split into \param split segments.
std::shared_ptr<TriangleMesh> CreateMeshCone(double radius = 1.0,
        double height = 2.0, int resolution = 20, int split = 1);

/// Factory function to create an arrow mesh (TriangleMeshFactory.cpp)
/// The axis of the cone with \param cone_radius will be along the z-axis.
/// The cylinder with \param cylinder_radius is from
/// (0, 0, 0) to (0, 0, cylinder_height), and
/// the cone is from (0, 0, cylinder_height)
/// to (0, 0, cylinder_height + cone_height).
/// The cone will be split into \param resolution segments.
/// The \param cylinder_height will be split into \param cylinder_split segments.
/// The \param cone_height will be split into \param cone_split segments.
std::shared_ptr<TriangleMesh> CreateMeshArrow(double cylinder_radius = 1.0,
        double cone_radius = 1.5, double cylinder_height = 5.0,
        double cone_height = 4.0, int resolution = 20, int cylinder_split = 4,
        int cone_split = 1);

/// Factory function to create a coordinate frame mesh (TriangleMeshFactory.cpp)
/// The coordinate frame will be centered at \param origin
/// The x, y, z axis will be rendered as red, green, and blue arrows respectively.
/// \param size is the length of the axes.
std::shared_ptr<TriangleMesh> CreateMeshCoordinateFrame(double size = 1.0,
        const Eigen::Vector3d &origin = Eigen::Vector3d(0.0, 0.0, 0.0));

}   // namespace open3d
