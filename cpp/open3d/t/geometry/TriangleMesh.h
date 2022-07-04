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

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/DrawableGeometry.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/TensorMap.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class TriangleMesh
/// \brief A triangle mesh contains vertices and triangles.
///
/// The triangle mesh class stores the attribute data in key-value maps. There
/// are two maps: the vertex attributes map, and the triangle attribute map.
///
/// - Default attribute: vertex_attr_["positions"], triangle_attr_["indices"]
///     - Vertex positions
///         - TriangleMesh::GetVertexPositions()
///         - TriangleMesh::SetVertexPositions(const Tensor& vertex_positions)
///         - TriangleMesh::HasVertexPositions()
///         - Value tensor must have shape {num_vertices, 3}.
///     - Triangle indices
///         - TriangleMesh::GetTriangleIndices()
///         - TriangleMesh::SetTriangleIndices(const Tensor& triangle_indices)
///         - TriangleMesh::HasTriangleIndices()
///         - Value tensor must have shape {num_triangles, 3}.
///     - Created by default, required for all triangle meshes.
///     - The device of vertex positions and triangle indices must be the same.
///       They determine the device of the trianglemesh.
///
/// - Common attributes: vertex_attr_["normals"], vertex_attr_["colors"]
///                      triangle_attr_["normals"], triangle_attr_["colors"]
///     - Vertex normals
///         - TriangleMesh::GetVertexNormals()
///         - TriangleMesh::SetVertexNormals(const Tensor& vertex_normals)
///         - TriangleMesh::HasVertexNormals()
///         - Value tensor must have shape {num_vertices, 3}.
///         - Value tensor can have any dtype.
///     - Vertex colors
///         - TriangleMesh::GetVertexColors()
///         - TriangleMesh::SetVertexColors(const Tensor& vertex_colors)
///         - TriangleMesh::HasVertexColors()
///         - Value tensor must have shape {num_vertices, 3}.
///         - Value tensor can have any dtype.
///     - Triangle normals
///         - TriangleMesh::GetTriangleNormals()
///         - TriangleMesh::SetTriangleNormals(const Tensor& triangle_normals)
///         - TriangleMesh::HasTriangleNormals()
///         - Value tensor must have shape {num_triangles, 3}.
///         - Value tensor can have any dtype.
///     - Triangle colors
///         - TriangleMesh::GetTriangleColors()
///         - TriangleMesh::SetTriangleColors(const Tensor& triangle_colors)
///         - TriangleMesh::HasTriangleColors()
///         - Value tensor must have shape {num_triangles, 3}.
///         - Value tensor can have any dtype.
///     - Not created by default.
///     - For all attributes above, the device must be consistent with the
///       device of the triangle mesh.
///
/// - Custom attributes: e.g. vetex_attr_["labels"], triangle_attr_["labels"]
///     - Use generalized helper functions, e.g.:
///         - TriangleMesh::GetVertexAttr(const std::string& key)
///         - TriangleMesh::SetVertexAttr(const std::string& key,
///                                       const Tensor& value)
///         - TriangleMesh::HasVertexAttr(const std::string& key)
///         - TriangleMesh::GetTriangleAttr(const std::string& key)
///         - TriangleMesh::SetTriangleAttr(const std::string& key,
///                                         const Tensor& value)
///         - TriangleMesh::HasTriangleAttr(const std::string& key)
///     - Not created by default. Users can add their own custom attributes.
///     - Value tensor must be on the same device as the triangle mesh.
///
/// Note that the we can also use the generalized helper functions for the
/// default and common attributes.
class TriangleMesh : public Geometry, public DrawableGeometry {
public:
    /// Construct an empty pointcloud on the provided device.
    /// \param device The device on which to initialize the trianglemesh
    /// (default: 'CPU:0').
    TriangleMesh(const core::Device &device = core::Device("CPU:0"));

    /// Construct a trianglemesh from vertices and triangles.
    ///
    /// The input tensors will be directly used as the underlying storage of
    /// the triangle mesh (no memory copy). The device for \p vertex_positions
    /// must be consistent with \p triangle_indices.
    ///
    /// \param vertex_positions A tensor with element shape {3}.
    /// \param triangle_indices A tensor with element shape {3}.
    TriangleMesh(const core::Tensor &vertex_positions,
                 const core::Tensor &triangle_indices);

    virtual ~TriangleMesh() override {}

public:
    /// \brief Text description.
    std::string ToString() const;

    /// Transfer the triangle mesh to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new triangle mesh is always created; if false,
    /// the copy is avoided when the original triangle mesh is already on the
    /// targeted device.
    TriangleMesh To(const core::Device &device, bool copy = false) const;

    /// Returns copy of the triangle mesh on the same device.
    TriangleMesh Clone() const { return To(GetDevice(), /*copy=*/true); }

    /// Getter for vertex_attr_ TensorMap. Used in Pybind.
    const TensorMap &GetVertexAttr() const { return vertex_attr_; }

    /// Get vertex attributes in vertex_attr_. Throws exception if the attribute
    /// does not exist.
    ///
    /// \param key Attribute name.
    core::Tensor &GetVertexAttr(const std::string &key) {
        return vertex_attr_.at(key);
    }

    /// Get the value of the "positions" attribute in vertex_attr_.
    /// Convenience function.
    core::Tensor &GetVertexPositions() { return GetVertexAttr("positions"); }

    /// Get the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    core::Tensor &GetVertexColors() { return GetVertexAttr("colors"); }

    /// Get the value of the "normals" attribute in vertex_attr_.
    /// Convenience function.
    core::Tensor &GetVertexNormals() { return GetVertexAttr("normals"); }

    /// Getter for triangle_attr_ TensorMap. Used in Pybind.
    const TensorMap &GetTriangleAttr() const { return triangle_attr_; }

    /// Get triangle attributes in triangle_attr_. Throws exception if the
    /// attribute does not exist.
    ///
    /// \param key Attribute name.
    core::Tensor &GetTriangleAttr(const std::string &key) {
        return triangle_attr_.at(key);
    }

    /// Get the value of the "indices" attribute in triangle_attr_.
    /// Convenience function.
    core::Tensor &GetTriangleIndices() { return GetTriangleAttr("indices"); }

    /// Get the value of the "normals" attribute in triangle_attr_.
    /// Convenience function.
    core::Tensor &GetTriangleNormals() { return GetTriangleAttr("normals"); }

    /// Get the value of the "colors" attribute in triangle_attr_.
    /// Convenience function.
    core::Tensor &GetTriangleColors() { return GetTriangleAttr("colors"); }

    /// Get vertex attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::Tensor &GetVertexAttr(const std::string &key) const {
        return vertex_attr_.at(key);
    }

    /// Removes vertex attribute by key value. Primary attribute "positions"
    /// cannot be removed. Throws warning if attribute key does not exists.
    ///
    /// \param key Attribute name.
    void RemoveVertexAttr(const std::string &key) { vertex_attr_.Erase(key); }

    /// Get the value of the "positions" attribute in vertex_attr_.
    /// Convenience function.
    const core::Tensor &GetVertexPositions() const {
        return GetVertexAttr("positions");
    }

    /// Get the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    const core::Tensor &GetVertexColors() const {
        return GetVertexAttr("colors");
    }

    /// Get the value of the "normals" attribute in vertex_attr_.
    /// Convenience function.
    const core::Tensor &GetVertexNormals() const {
        return GetVertexAttr("normals");
    }

    /// Get triangle attributes in triangle_attr_. Throws exception if the
    /// attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::Tensor &GetTriangleAttr(const std::string &key) const {
        return triangle_attr_.at(key);
    }

    /// Removes triangle attribute by key value. Primary attribute "indices"
    /// cannot be removed. Throws warning if attribute key does not exists.
    ///
    /// \param key Attribute name.
    void RemoveTriangleAttr(const std::string &key) {
        triangle_attr_.Erase(key);
    }

    /// Get the value of the "indices" attribute in triangle_attr_.
    /// Convenience function.
    const core::Tensor &GetTriangleIndices() const {
        return GetTriangleAttr("indices");
    }

    /// Get the value of the "normals" attribute in triangle_attr_.
    /// Convenience function.
    const core::Tensor &GetTriangleNormals() const {
        return GetTriangleAttr("normals");
    }

    /// Get the value of the "colors" attribute in triangle_attr_.
    /// Convenience function.
    const core::Tensor &GetTriangleColors() const {
        return GetTriangleAttr("colors");
    }

    /// Set vertex attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensor.
    void SetVertexAttr(const std::string &key, const core::Tensor &value) {
        core::AssertTensorDevice(value, device_);
        vertex_attr_[key] = value;
    }

    /// Set the value of the "positions" attribute in vertex_attr_.
    /// Convenience function.
    void SetVertexPositions(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetVertexAttr("positions", value);
    }

    /// Set the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    void SetVertexColors(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetVertexAttr("colors", value);
    }

    /// Set the value of the "normals" attribute in vertex_attr_.
    /// This is a convenience function.
    void SetVertexNormals(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetVertexAttr("normals", value);
    }

    /// Set triangle attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensor.
    void SetTriangleAttr(const std::string &key, const core::Tensor &value) {
        core::AssertTensorDevice(value, device_);
        triangle_attr_[key] = value;
    }

    /// Set the value of the "indices" attribute in triangle_attr_.
    void SetTriangleIndices(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetTriangleAttr("indices", value);
    }

    /// Set the value of the "normals" attribute in triangle_attr_.
    /// This is a convenience function.
    void SetTriangleNormals(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetTriangleAttr("normals", value);
    }

    /// Set the value of the "colors" attribute in triangle_attr_.
    /// This is a convenience function.
    void SetTriangleColors(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetTriangleAttr("colors", value);
    }

    /// Returns true if all of the following are true in vertex_attr_:
    /// 1) attribute key exist
    /// 2) attribute's length as vertices' length
    /// 3) attribute's length > 0
    bool HasVertexAttr(const std::string &key) const {
        return vertex_attr_.Contains(key) &&
               GetVertexAttr(key).GetLength() > 0 &&
               GetVertexAttr(key).GetLength() ==
                       GetVertexPositions().GetLength();
    }

    /// Check if the "positions" attribute's value in vertex_attr_ has length >
    /// 0. Convenience function.
    bool HasVertexPositions() const { return HasVertexAttr("positions"); }

    /// Returns true if all of the following are true in vertex_attr_:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as vertices' length
    /// 3) attribute "colors"'s length > 0
    /// Convenience function.
    bool HasVertexColors() const { return HasVertexAttr("colors"); }

    /// Returns true if all of the following are true in vertex_attr_:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as vertices' length
    /// 3) attribute "normals"'s length > 0
    /// Convenience function.
    bool HasVertexNormals() const { return HasVertexAttr("normals"); }

    /// Returns true if all of the following are true in triangle_attr_:
    /// 1) attribute key exist
    /// 2) attribute's length as triangles' length
    /// 3) attribute's length > 0
    bool HasTriangleAttr(const std::string &key) const {
        return triangle_attr_.Contains(key) &&
               GetTriangleAttr(key).GetLength() > 0 &&
               GetTriangleAttr(key).GetLength() ==
                       GetTriangleIndices().GetLength();
    }

    /// Check if the "indices" attribute's value in triangle_attr_ has length
    /// > 0.
    /// Convenience function.
    bool HasTriangleIndices() const { return HasTriangleAttr("indices"); }

    /// Returns true if all of the following are true in triangle_attr_:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as vertices' length
    /// 3) attribute "normals"'s length > 0
    /// Convenience function.
    bool HasTriangleNormals() const { return HasTriangleAttr("normals"); }

    /// Returns true if all of the following are true in triangle_attr_:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as vertices' length
    /// 3) attribute "colors"'s length > 0
    /// Convenience function.
    bool HasTriangleColors() const { return HasTriangleAttr("colors"); }

    /// Create a box triangle mesh. One vertex of the box will be placed at
    /// the origin and the box aligns with the positive x, y, and z axes.
    /// \param width is x-directional length.
    /// \param height is y-directional length.
    /// \param depth is z-directional length.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateBox(
            double width = 1.0,
            double height = 1.0,
            double depth = 1.0,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

public:
    /// Clear all data in the trianglemesh.
    TriangleMesh &Clear() override {
        vertex_attr_.clear();
        triangle_attr_.clear();
        return *this;
    }

    /// Returns !HasVertexPositions(), triangles are ignored.
    bool IsEmpty() const override { return !HasVertexPositions(); }

    core::Tensor GetMinBound() const { return GetVertexPositions().Min({0}); }

    core::Tensor GetMaxBound() const { return GetVertexPositions().Max({0}); }

    core::Tensor GetCenter() const { return GetVertexPositions().Mean({0}); }

    /// \brief Transforms the VertexPositions, VertexNormals and TriangleNormals
    /// (if exist) of the TriangleMesh.
    ///
    /// Transformation matrix is a 4x4 matrix.
    ///  T (4x4) =   [[ R(3x3)  t(3x1) ],
    ///               [ O(1x3)  s(1x1) ]]
    ///  (s = 1 for Transformation without scaling)
    ///
    ///  It applies the following general transform to each `positions` and
    ///  `normals`.
    ///   |x'|   | R(0,0) R(0,1) R(0,2) t(0)|   |x|
    ///   |y'| = | R(1,0) R(1,1) R(1,2) t(1)| @ |y|
    ///   |z'|   | R(2,0) R(2,1) R(2,2) t(2)|   |z|
    ///   |w'|   | O(0,0) O(0,1) O(0,2)  s  |   |1|
    ///
    ///   [x, y, z] = [x', y', z'] / w'
    ///
    /// \param transformation Transformation [Tensor of dim {4,4}].
    /// \return Transformed TriangleMesh
    TriangleMesh &Transform(const core::Tensor &transformation);

    /// \brief Translates the VertexPositions of the TriangleMesh.
    /// \param translation translation tensor of dimension {3}
    /// \param relative if true (default): translates relative to Center
    /// \return Translated TriangleMesh
    TriangleMesh &Translate(const core::Tensor &translation,
                            bool relative = true);

    /// \brief Scales the VertexPositions of the TriangleMesh.
    /// \param scale Scale [double] of dimension
    /// \param center Center [Tensor of dim {3}] about which the TriangleMesh is
    /// to be scaled.
    /// \return Scaled TriangleMesh
    TriangleMesh &Scale(double scale, const core::Tensor &center);

    /// \brief Rotates the VertexPositions, VertexNormals and TriangleNormals
    /// (if exists).
    /// \param R Rotation [Tensor of dim {3,3}].
    /// \param center Center [Tensor of dim {3}] about which the TriangleMesh is
    /// to be scaled.
    /// \return Rotated TriangleMesh
    TriangleMesh &Rotate(const core::Tensor &R, const core::Tensor &center);

    /// \brief Clip mesh with a plane.
    /// This method clips the triangle mesh with the specified plane.
    /// Parts of the mesh on the positive side of the plane will be kept and
    /// triangles intersected by the plane will be cut.
    /// \param point A point on the plane as [Tensor of dim {3}].
    /// \param normal The normal of the plane as [Tensor of dim {3}]. The normal
    /// points to the positive side of the plane for which the geometry will be
    /// kept.
    /// \return New triangle mesh clipped with the plane.
    TriangleMesh ClipPlane(const core::Tensor &point,
                           const core::Tensor &normal) const;

    core::Device GetDevice() const override { return device_; }

    /// Create a TriangleMesh from a legacy Open3D TriangleMesh.
    /// \param mesh_legacy Legacy Open3D TriangleMesh.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static geometry::TriangleMesh FromLegacy(
            const open3d::geometry::TriangleMesh &mesh_legacy,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Convert to a legacy Open3D TriangleMesh.
    open3d::geometry::TriangleMesh ToLegacy() const;

    /// Compute the convex hull of the triangle mesh using qhull.
    ///
    /// This runs on the CPU.
    ///
    /// \param joggle_inputs (default False). Handle precision problems by
    /// randomly perturbing the input data. Set to True if perturbing the input
    /// iis acceptable but you need convex simplicial output. If False,
    /// neighboring facets may be merged in case of precision problems. See
    /// [QHull docs](http://www.qhull.org/html/qh-impre.htm#joggle) for more
    /// details.
    ///
    /// \return TriangleMesh representing the convexh hull. This contains an
    /// extra vertex property "point_map" that contains the index of the
    /// corresponding vertex in the original mesh.
    TriangleMesh ComputeConvexHull(bool joggle_inputs = false) const;

    /// Function to simplify mesh using Quadric Error Metric Decimation by
    /// Garland and Heckbert.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param target_reduction The factor of triangles to delete, i.e.,
    /// setting this to 0.9 will return a mesh with about 10% of the original
    /// triangle count.
    /// It is not guaranteed that the target reduction factor will be reached.
    /// \param preserve_volume If set to true this enables volume preservation
    /// which reduces the error in triangle normal direction.
    ///
    /// \return Simplified TriangleMesh.
    TriangleMesh SimplifyQuadricDecimation(double target_reduction,
                                           bool preserve_volume = true) const;

    /// Computes the mesh that encompasses the union of the volumes of two
    /// meshes.
    /// Both meshes should be manifold.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param mesh This is the second operand for the boolean operation.
    /// \param tolerance Threshold which determines when point distances are
    /// considered to be 0.
    ///
    /// \return The mesh describing the union volume.
    TriangleMesh BooleanUnion(const TriangleMesh &mesh,
                              double tolerance = 1e-6) const;

    /// Computes the mesh that encompasses the intersection of the volumes of
    /// two meshes. Both meshes should be manifold.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param mesh This is the second operand for the boolean operation.
    /// \param tolerance Threshold which determines when point distances are
    /// considered to be 0.
    ///
    /// \return The mesh describing the intersection volume.
    TriangleMesh BooleanIntersection(const TriangleMesh &mesh,
                                     double tolerance = 1e-6) const;

    /// Computes the mesh that encompasses the volume after subtracting the
    /// volume of the second operand. Both meshes should be manifold.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param mesh This is the second operand for the boolean operation.
    /// \param tolerance Threshold which determines when point distances are
    /// considered to be 0.
    ///
    /// \return The mesh describing the difference volume.
    TriangleMesh BooleanDifference(const TriangleMesh &mesh,
                                   double tolerance = 1e-6) const;

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorMap vertex_attr_;
    TensorMap triangle_attr_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
