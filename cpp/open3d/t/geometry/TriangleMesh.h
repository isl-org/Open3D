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

#include <list>

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/DrawableGeometry.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/TensorMap.h"

namespace open3d {
namespace t {
namespace geometry {

class LineSet;

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

    /// Getter for vertex_attr_ TensorMap.
    TensorMap &GetVertexAttr() { return vertex_attr_; }

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

    /// Getter for triangle_attr_ TensorMap.
    TensorMap &GetTriangleAttr() { return triangle_attr_; }

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

    /// Create a sphere triangle mesh. The sphere with radius will be centered
    /// at (0, 0, 0). Its axis is aligned with z-axis.
    /// \param radius defines the radius of the sphere.
    /// \param resolution defines the resolution of the sphere. The longitudes
    /// will be split into resolution segments (i.e. there are resolution + 1
    /// latitude lines including the north and south pole). The latitudes will
    /// be split into `2 * resolution segments (i.e. there are 2 * resolution
    /// longitude lines.)
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateSphere(
            double radius = 1.0,
            int resolution = 20,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a tetrahedron triangle mesh. The centroid of the mesh will be
    /// placed at (0, 0, 0) and the vertices have a distance of radius to the
    /// center.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateTetrahedron(
            double radius = 1.0,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a octahedron triangle mesh. The centroid of the mesh will be
    /// placed at (0, 0, 0) and the vertices have a distance of radius to the
    /// center.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateOctahedron(
            double radius = 1.0,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a icosahedron triangle mesh. The centroid of the mesh will be
    /// placed at (0, 0, 0) and the vertices have a distance of radius to the
    /// center.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateIcosahedron(
            double radius = 1.0,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a cylinder triangle mesh.
    /// \param radius defines the radius of the cylinder.
    /// \param height defines the height of the cylinder. The axis of the
    /// cylinder will be from (0, 0, -height/2) to (0, 0, height/2).
    /// \param resolution defines the resolution of the cylinder. The circle
    /// will be split into resolution segments
    /// \param split defines the number of segments along the height direction.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateCylinder(
            double radius = 1.0,
            double height = 2.0,
            int resolution = 20,
            int split = 4,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a cone triangle mesh.
    /// \param radius defines the radius of the cone.
    /// \param height defines the height of the cone. The axis of the
    /// cone will be from (0, 0, 0) to (0, 0, height).
    /// \param resolution defines the resolution of the cone. The circle
    /// will be split into resolution segments.
    /// \param split defines the number of segments along the height direction.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateCone(
            double radius = 1.0,
            double height = 2.0,
            int resolution = 20,
            int split = 1,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a torus triangle mesh.
    /// \param torus_radius defines the radius from the center of the
    /// torus to the center of the tube.
    /// \param tube_radius defines the radius of the torus tube.
    /// \param radial_resolution defines the number of segments along the
    /// radial direction.
    /// \param tubular_resolution defines the number of segments along
    /// the tubular direction.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateTorus(
            double torus_radius = 1.0,
            double tube_radius = 0.5,
            int radial_resolution = 30,
            int tubular_resolution = 20,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a arrow triangle mesh.
    /// \param cylinder_radius defines the radius of the cylinder.
    /// \param cone_radius defines the radius of the cone.
    /// \param cylinder_height defines the height of the cylinder. The axis of
    /// cylinder is from (0, 0, 0) to (0, 0, cylinder_height).
    /// \param cone_height defines the height of the cone. The axis of the
    /// cone will be from (0, 0, cylinder_height) to (0, 0, cylinder_height +
    /// cone_height). \param resolution defines the resolution of the cone. The
    /// circle will be split into resolution segments. \param cylinder_split
    /// defines the number of segments along the cylinder_height direction.
    /// \param cone_split defines the number of segments along
    /// the cone_height direction.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateArrow(
            double cylinder_radius = 1.0,
            double cone_radius = 1.5,
            double cylinder_height = 5.0,
            double cone_height = 4.0,
            int resolution = 20,
            int cylinder_split = 4,
            int cone_split = 1,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a coordinate frame mesh.
    /// \param size defines the size of the coordinate frame.
    /// \param origin defines the origin of the coordinate frame.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateCoordinateFrame(
            double size = 1.0,
            const Eigen::Vector3d &origin = Eigen::Vector3d(0.0, 0.0, 0.0),
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a Mobius strip.
    /// \param length_split defines the number of segments along the Mobius
    /// strip.
    /// \param width_split defines the number of segments along the width
    /// of the Mobius strip.
    /// \param twists defines the number of twists of the strip.
    /// \param radius defines the radius of the Mobius strip.
    /// \param flatness controls the height of the strip.
    /// \param width controls the width of the Mobius strip.
    /// \param scale is used to scale the entire Mobius strip.
    /// \param float_dtype Float32 or Float64, used to store floating point
    /// values, e.g. vertices, normals, colors.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateMobius(
            int length_split = 70,
            int width_split = 15,
            int twists = 1,
            double radius = 1,
            double flatness = 1,
            double width = 1,
            double scale = 1,
            core::Dtype float_dtype = core::Float32,
            core::Dtype int_dtype = core::Int64,
            const core::Device &device = core::Device("CPU:0"));

    /// Create a text triangle mesh.
    /// \param text The text for generating the mesh. ASCII characters 32-126
    /// are supported (includes alphanumeric characters and punctuation). In
    /// addition the line feed '\n' is supported to start a new line.
    /// \param depth The depth of the generated mesh. If depth is 0 then a flat
    /// mesh will be generated.
    /// \param int_dtype Int32 or Int64, used to store index values, e.g.
    /// triangles.
    /// \param device The device where the resulting TriangleMesh resides in.
    static TriangleMesh CreateText(
            const std::string &text,
            double depth = 0.0,
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

    /// Normalize both triangle normals and vertex normals to length 1.
    TriangleMesh &NormalizeNormals();

    /// \brief Function to compute triangle normals, usually called before
    /// rendering.
    TriangleMesh &ComputeTriangleNormals(bool normalized = true);

    /// \brief Function to compute vertex normals, usually called before
    /// rendering.
    TriangleMesh &ComputeVertexNormals(bool normalized = true);

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

    /// \brief Extract contour slices given a plane.
    /// This method extracts slices as LineSet from the mesh at specific
    /// contour values defined by the specified plane.
    /// \param point A point on the plane as [Tensor of dim {3}].
    /// \param normal The normal of the plane as [Tensor of dim {3}].
    /// \param contour_values Contour values at which slices will be generated.
    /// The value describes the signed distance to the plane.
    /// \return LineSet with the extracted contours.
    LineSet SlicePlane(const core::Tensor &point,
                       const core::Tensor &normal,
                       const std::vector<double> contour_values = {0.0}) const;

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

    /// Create an axis-aligned bounding box from vertex attribute "positions".
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const;

    /// Fill holes by triangulating boundary edges.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param hole_size This is the approximate threshold for filling holes.
    /// The value describes the maximum radius of holes to be filled.
    ///
    /// \return New mesh after filling holes.
    TriangleMesh FillHoles(double hole_size = 1e6) const;

    /// Creates an UV atlas and adds it as triangle attr 'texture_uvs' to the
    /// mesh.
    ///
    /// Input meshes must be manifold for this method to work.
    ///
    /// The algorithm is based on:
    /// - Zhou et al, "Iso-charts: Stretch-driven Mesh Parameterization using
    /// Spectral Analysis", Eurographics Symposium on Geometry Processing (2004)
    /// - Sander et al. "Signal-Specialized Parametrization" Europgraphics 2002
    ///
    /// This function always uses the CPU device.
    ///
    /// \param size The target size of the texture (size x size). The uv
    /// coordinates will still be in the range [0..1] but parameters like gutter
    /// use pixels as units.
    /// \param gutter This is the space around the uv islands in pixels.
    /// \param max_stretch The maximum amount of stretching allowed. The
    /// parameter range is [0..1] with 0 meaning no stretch allowed.
    void ComputeUVAtlas(size_t size = 512,
                        float gutter = 1.0f,
                        float max_stretch = 1.f / 6);

    /// Bake vertex attributes into textures.
    ///
    /// This function assumes a triangle attribute with name 'texture_uvs'.
    /// Only float type attributes can be baked to textures.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param size The width and height of the texture in pixels. Only square
    /// textures are supported.
    ///
    /// \param vertex_attr The vertex attributes for which textures should be
    /// generated.
    ///
    /// \param margin The margin in pixels. The recommended value is 2. The
    /// margin are additional pixels around the UV islands to avoid
    /// discontinuities.
    ///
    /// \param fill The value used for filling texels outside the UV islands.
    ///
    /// \param update_material If true updates the material of the mesh.
    /// Baking a vertex attribute with the name 'albedo' will become the albedo
    /// texture in the material. Existing textures in the material will be
    /// overwritten.
    ///
    /// \return A dictionary of textures.
    std::unordered_map<std::string, core::Tensor> BakeVertexAttrTextures(
            int size,
            const std::unordered_set<std::string> &vertex_attr = {},
            double margin = 2.,
            double fill = 0.,
            bool update_material = true);

    /// Bake triangle attributes into textures.
    ///
    /// This function assumes a triangle attribute with name 'texture_uvs'.
    ///
    /// This function always uses the CPU device.
    ///
    /// \param size The width and height of the texture in pixels. Only square
    /// textures are supported.
    ///
    /// \param vertex_attr The vertex attributes for which textures should be
    /// generated.
    ///
    /// \param margin The margin in pixels. The recommended value is 2. The
    /// margin are additional pixels around the UV islands to avoid
    /// discontinuities.
    ///
    /// \param fill The value used for filling texels outside the UV islands.
    ///
    /// \param update_material If true updates the material of the mesh.
    /// Baking a vertex attribute with the name 'albedo' will become the albedo
    /// texture in the material. Existing textures in the material will be
    /// overwritten.
    ///
    /// \return A dictionary of textures.
    std::unordered_map<std::string, core::Tensor> BakeTriangleAttrTextures(
            int size,
            const std::unordered_set<std::string> &triangle_attr = {},
            double margin = 2.,
            double fill = 0.,
            bool update_material = true);

    /// Sweeps the triangle mesh rotationally about an axis.
    /// \param angle The rotation angle in degree.
    /// \param axis The rotation axis.
    /// \param resolution The resolution defines the number of intermediate
    /// sweeps about the rotation axis.
    /// \param translation The translation along the rotation axis.
    /// \param capping If true adds caps to the mesh.
    /// \return A triangle mesh with the result of the sweep operation.
    TriangleMesh ExtrudeRotation(double angle,
                                 const core::Tensor &axis,
                                 int resolution = 16,
                                 double translation = 0.0,
                                 bool capping = true) const;

    /// Sweeps the triangle mesh along a direction vector.
    /// \param vector The direction vector.
    /// \param scale Scalar factor which essentially scales the direction
    /// vector. \param capping If true adds caps to the mesh. \return A triangle
    /// mesh with the result of the sweep operation.
    TriangleMesh ExtrudeLinear(const core::Tensor &vector,
                               double scale = 1.0,
                               bool capping = true) const;

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorMap vertex_attr_;
    TensorMap triangle_attr_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
