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

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/TensorListMap.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class TriangleMesh
/// \brief A TriangleMesh contains vertices and triangles.
///
/// The TriangleMesh class stores the attribute data in key-value pairs for
/// flexibility, where the key is a string representing the attribute name and
/// value is a TensorList containing the attribute data.
///
/// By default, there are two sets of dictionaries, i.e. `vertex_attr_` and
/// `triangle_attr_`. In most cases, the length of an attribute should be
/// equal to the length of the the data corresponding to the master key. For
/// instance `vertex_attr_["normals"]` should have the same length as
/// `vertex_attr_["vertices"]`.
///
/// Although the attributes are all stored in a key-value pair dictionary, the
/// attributes have different levels:
///
/// - Level 0: Default attribute {"vertices", "triangles"}.
///     - Created by default, required for all trianglemeshes.
///     - The tensorlist must be of shape N x {3,}.
///     - Convenience functions:
///         - TriangleMesh::GetVertices()
///         - TriangleMesh::SetVertices(vertices_tensorlist)
///         - TriangleMesh::HasVertices()
///         - TriangleMesh::GetTriangles()
///         - TriangleMesh::SetTriangles(triangles_tensorlist)
///         - TriangleMesh::HasTriangles()
///     - The device of "vertices" and "triangles" must be consistent and they
///       determine the device of the trianglemesh.
/// - Level 1: Commonly-used attributes: {"normals", "colors"} for vertices and
///            {"normals"} for triangles.
///     - Not created by default.
///     - The tensorlist must be of shape N x {3,}.
///     - Convenience functions:
///         - Vertex normals (stored at vertex_attr_["normals"])
///             - TriangleMesh::GetVertexNormals()
///             - TriangleMesh::SetVertexNormals(vertex_normals_tensorlist)
///             - TriangleMesh::HasVertexNormals()
///         - Vertex colors (stored at vertex_attr_["colors"])
///             - TriangleMesh::GetVertexColors()
///             - TriangleMesh::SetVertexColors(vertex_colors_tensorlist)
///             - TriangleMesh::HasVertexColors()
///         - Triangle normals (stored at triangle_attr_["normals"])
///             - TriangleMesh::GetTriangleNormals()
///             - TriangleMesh::SetTriangleNormals(triangle_normals_tensorlist)
///             - TriangleMesh::HasTriangleNormals()
///         - Triangle colors (stored at triangle_attr_["colors"])
///             - TriangleMesh::GetTriangleColors()
///             - TriangleMesh::SetTriangleColors(triangle_colors_tensorlist)
///             - TriangleMesh::HasTriangleColors()
///     - For all attributes, the device must be consistent with the device of
///       the trianglemesh. Dtype can be different.
/// - Level 2: Custom attributes, e.g. {"labels"}.
///     - Not created by default. Created by users.
///     - No convenience functions.
///     - Use generalized helper functions. Examples:
///         - TriangleMesh::GetVertexAttr("labels")
///         - TriangleMesh::SetVertexAttr("labels",
///                                       vertex_labels_tensorlist)
///         - TriangleMesh::HasVertexAttr("labels")
///         - TriangleMesh::GetTriangleAttr("labels")
///         - TriangleMesh::SetTriangleAttr("labels",
///                                         triangle_labels_tensorlist)
///         - TriangleMesh::HasTriangleAttr("labels")
///     - For all attributes, the device must be consistent with the device of
///       the trianglemesh. Dtype can be different.
///
/// Note that the level 0 and level 1 convenience functions can also be achieved
/// via the generalized helper functions.
class TriangleMesh : public Geometry {
public:
    /// Construct an empty trianglemesh.
    TriangleMesh(core::Dtype vertex_dtype = core::Dtype::Float32,
                 core::Dtype triangle_dtype = core::Dtype::Int64,
                 const core::Device &device = core::Device("CPU:0"));

    /// Construct a trianglemesh from vertices and triangles.
    ///
    /// The input tensorlists will be directly used as the underlying storage of
    /// the trianglemsh (no memory copy). If the tensorlist is created in-place
    /// from a pre-allocated buffer, the tensorlist has a fixed size and thus
    /// the resulting trianglemesh will have a fixed size and calling to
    /// functions like `SynchronizedPushBack` will raise an exception.
    ///
    /// The resulting trianglemesh will have the same dtype and device as the
    /// tensorlist. The device for \p vertices must be consistent with
    /// \p triangles.
    ///
    /// \param vertices A tensorlist with element shape (3,).
    /// \param triangles A tensorlist with element shape (3,).
    TriangleMesh(const core::TensorList &vertices,
                 const core::TensorList &triangles);

    virtual ~TriangleMesh() override {}

public:
    /// Get vertex attributes in vertex_attr_. Throws exception if the attribute
    /// does not exist.
    ///
    /// \param key Attribute name.
    core::TensorList &GetVertexAttr(const std::string &key) {
        return vertex_attr_.at(key);
    }

    /// Get the value of the "vertices" attribute in vertex_attr_.
    /// Convenience function.
    core::TensorList &GetVertices() { return GetVertexAttr("vertices"); }

    /// Get the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    core::TensorList &GetVertexColors() { return GetVertexAttr("colors"); }

    /// Get the value of the "normals" attribute in vertex_attr_.
    /// Convenience function.
    core::TensorList &GetVertexNormals() { return GetVertexAttr("normals"); }

    /// Get triangle attributes in triangle_attr_. Throws exception if the
    /// attribute does not exist.
    ///
    /// \param key Attribute name.
    core::TensorList &GetTriangleAttr(const std::string &key) {
        return triangle_attr_.at(key);
    }

    /// Get the value of the "triangles" attribute in triangle_attr_.
    /// Convenience function.
    core::TensorList &GetTriangles() { return GetTriangleAttr("triangles"); }

    /// Get the value of the "normals" attribute in triangle_attr_.
    /// Convenience function.
    core::TensorList &GetTriangleNormals() {
        return GetTriangleAttr("normals");
    }

    /// Get the value of the "colors" attribute in triangle_attr_.
    /// Convenience function.
    core::TensorList &GetTriangleColors() { return GetTriangleAttr("colors"); }

    /// Get vertex attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::TensorList &GetVertexAttr(const std::string &key) const {
        return vertex_attr_.at(key);
    }

    /// Get the value of the "vertices" attribute in vertex_attr_.
    /// Convenience function.
    const core::TensorList &GetVertices() const {
        return GetVertexAttr("vertices");
    }

    /// Get the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    const core::TensorList &GetVertexColors() const {
        return GetVertexAttr("colors");
    }

    /// Get the value of the "normals" attribute in vertex_attr_.
    /// Convenience function.
    const core::TensorList &GetVertexNormals() const {
        return GetVertexAttr("normals");
    }

    /// Get triangle attributes in triangle_attr_. Throws exception if the
    /// attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::TensorList &GetTriangleAttr(const std::string &key) const {
        return triangle_attr_.at(key);
    }

    /// Get the value of the "triangles" attribute in triangle_attr_.
    /// Convenience function.
    const core::TensorList &GetTriangles() const {
        return GetTriangleAttr("triangles");
    }

    /// Get the value of the "normals" attribute in triangle_attr_.
    /// Convenience function.
    const core::TensorList &GetTriangleNormals() const {
        return GetTriangleAttr("normals");
    }

    /// Get the value of the "colors" attribute in triangle_attr_.
    /// Convenience function.
    const core::TensorList &GetTriangleColors() const {
        return GetTriangleAttr("colors");
    }

    /// Set vertex attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensorlist.
    void SetVertexAttr(const std::string &key, const core::TensorList &value) {
        value.AssertDevice(device_);
        vertex_attr_[key] = value;
    }

    /// Set the value of the "vertices" attribute in vertex_attr_.
    /// Convenience function.
    void SetVertices(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetVertexAttr("vertices", value);
    }

    /// Set the value of the "colors" attribute in vertex_attr_.
    /// Convenience function.
    void SetVertexColors(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetVertexAttr("colors", value);
    }

    /// Set the value of the "normals" attribute in vertex_attr_.
    /// This is a convenience function.
    void SetVertexNormals(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetVertexAttr("normals", value);
    }

    /// Set triangle attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensorlist.
    void SetTriangleAttr(const std::string &key,
                         const core::TensorList &value) {
        value.AssertDevice(device_);
        triangle_attr_[key] = value;
    }

    /// Set the vlaue of the "triangles" attribute in triangle_attr_.
    void SetTriangles(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetTriangleAttr("triangles", value);
    }

    /// Set the value of the "normals" attribute in triangle_attr_.
    /// This is a convenience function.
    void SetTriangleNormals(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetTriangleAttr("normals", value);
    }

    /// Set the value of the "colors" attribute in triangle_attr_.
    /// This is a convenience function.
    void SetTriangleColors(const core::TensorList &value) {
        value.AssertElementShape({3});
        SetTriangleAttr("colors", value);
    }

    /// Returns true if all of the followings are true in vertex_attr_:
    /// 1) attribute key exist
    /// 2) attribute's length as vertices' length
    /// 3) attribute's length > 0
    bool HasVertexAttr(const std::string &key) const {
        return vertex_attr_.Contains(key) && GetVertexAttr(key).GetSize() > 0 &&
               GetVertexAttr(key).GetSize() == GetVertices().GetSize();
    }

    /// Check if the "vertices" attribute's value in vertex_attr_ has length >
    /// 0. Convenience function.
    bool HasVertices() const { return HasVertexAttr("vertices"); }

    /// Returns true if all of the followings are true in vertex_attr_:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as vertices' length
    /// 3) attribute "colors"'s length > 0
    /// Convenience function.
    bool HasVertexColors() const { return HasVertexAttr("colors"); }

    /// Returns true if all of the followings are true in vertex_attr_:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as vertices' length
    /// 3) attribute "normals"'s length > 0
    /// Convenience function.
    bool HasVertexNormals() const { return HasVertexAttr("normals"); }

    /// Returns true if all of the followings are true in triangle_attr_:
    /// 1) attribute key exist
    /// 2) attribute's length as triangles' length
    /// 3) attribute's length > 0
    bool HasTriangleAttr(const std::string &key) const {
        return triangle_attr_.Contains(key) &&
               GetTriangleAttr(key).GetSize() > 0 &&
               GetTriangleAttr(key).GetSize() == GetTriangles().GetSize();
    }

    /// Check if the "triangles" attribute's value in triangle_attr_ has length
    /// > 0.
    /// Convenience function.
    bool HasTriangles() const { return HasTriangleAttr("triangles"); }

    /// Returns true if all of the followings are true in triangle_attr_:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as vertices' length
    /// 3) attribute "normals"'s length > 0
    /// Convenience function.
    bool HasTriangleNormals() const { return HasTriangleAttr("normals"); }

    /// Returns true if all of the followings are true in triangle_attr_:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as vertices' length
    /// 3) attribute "colors"'s length > 0
    /// Convenience function.
    bool HasTriangleColors() const { return HasTriangleAttr("colors"); }

    /// Synchronized push back to the vertex_attr_, data will be
    /// copied. Before push back, all existing tensorlists must have the same
    /// length.
    ///
    /// \param map_keys_to_tensors The keys and values to be pushed back. It
    /// must contain the same keys and each corresponding tensor must have the
    /// same dtype and device.
    void VertexSynchronizedPushBack(
            const std::unordered_map<std::string, core::Tensor>
                    &map_keys_to_tensors) {
        vertex_attr_.SynchronizedPushBack(map_keys_to_tensors);
    }

    /// Synchronized push back to the triangle_attr_, data will be
    /// copied. Before push back, all existing tensorlists must have the same
    /// length.
    ///
    /// \param map_keys_to_tensors The keys and values to be pushed back. It
    /// must contain the same keys and each corresponding tensor must have the
    /// same dtype and device.
    void TriangleSynchronizedPushBack(
            const std::unordered_map<std::string, core::Tensor>
                    &map_keys_to_tensors) {
        triangle_attr_.SynchronizedPushBack(map_keys_to_tensors);
    }

public:
    /// Clear all data in the trianglemesh.
    TriangleMesh &Clear() override {
        vertex_attr_.clear();
        triangle_attr_.clear();
        return *this;
    }

    /// Returns !HasVertices(), triangles are ignored.
    bool IsEmpty() const override { return !HasVertices(); }

    core::Tensor GetMinBound() const { utility::LogError("Unimplemented"); }

    core::Tensor GetMaxBound() const { utility::LogError("Unimplemented"); }

    core::Tensor GetCenter() const { utility::LogError("Unimplemented"); }

    TriangleMesh &Transform(const core::Tensor &transformation) {
        utility::LogError("Unimplemented");
    }

    TriangleMesh &Translate(const core::Tensor &translation,
                            bool relative = true) {
        utility::LogError("Unimplemented");
    }

    TriangleMesh &Scale(double scale, const core::Tensor &center) {
        utility::LogError("Unimplemented");
    }

    TriangleMesh &Rotate(const core::Tensor &R, const core::Tensor &center) {
        utility::LogError("Unimplemented");
    }

    core::Device GetDevice() const { return device_; }

    /// Create a TriangleMesh from a legacy Open3D TriangleMesh.
    static geometry::TriangleMesh FromLegacyTrangleMesh(
            const geometry::TriangleMesh &mesh_legacy,
            core::Dtype dtype = core::Dtype::Float32,
            const core::Device &device = core::Device("CPU:0")) {
        utility::LogError("Unimplemented");
    }

    /// Convert to a legacy Open3D TriangleMesh.
    geometry::TriangleMesh ToLegacyTriangleMesh() const {
        utility::LogError("Unimplemented");
    }

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorListMap vertex_attr_;
    TensorListMap triangle_attr_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
