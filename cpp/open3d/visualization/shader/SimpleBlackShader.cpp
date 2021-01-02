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

#include "open3d/visualization/shader/SimpleBlackShader.h"

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/geometry/PlanarPatch.h"
#include "open3d/visualization/shader/Shader.h"
#include "open3d/visualization/utility/ColorMap.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool SimpleBlackShader::Compile() {
    if (!CompileShaders(SimpleBlackVertexShader, NULL,
                        SimpleBlackFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleBlackShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleBlackShader::BindGeometry(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    std::vector<Eigen::Vector3f> points;
    if (!PrepareBinding(geometry, option, view, points)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool SimpleBlackShader::RenderGeometry(const geometry::Geometry &geometry,
                                       const RenderOption &option,
                                       const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    return true;
}

void SimpleBlackShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        bound_ = false;
    }
}

bool SimpleBlackShaderForPointCloudNormal::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleBlackShaderForPointCloudNormal::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    const geometry::PointCloud &pointcloud =
            (const geometry::PointCloud &)geometry;
    if (!pointcloud.HasPoints()) {
        PrintShaderWarning("Binding failed with empty pointcloud.");
        return false;
    }
    points.resize(pointcloud.points_.size() * 2);
    double line_length =
            option.point_size_ * 0.01 * view.GetBoundingBox().GetMaxExtent();
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const auto &point = pointcloud.points_[i];
        const auto &normal = pointcloud.normals_[i];
        points[i * 2] = point.cast<float>();
        points[i * 2 + 1] = (point + normal * line_length).cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_POLYGON_OFFSET_FILL);
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    const geometry::TriangleMesh &mesh =
            (const geometry::TriangleMesh &)geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty geometry::TriangleMesh.");
        return false;
    }
    points.resize(mesh.triangles_.size() * 3);
    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t vi = triangle(j);
            const auto &vertex = mesh.vertices_[vi];
            points[idx] = vertex.cast<float>();
        }
    }
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleBlackShaderForPlanarPatchNormal::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::PlanarPatch) {
        PrintShaderWarning("Rendering type is not geometry::PlanarPatch.");
        return false;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleBlackShaderForPlanarPatchNormal::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::PlanarPatch) {
        PrintShaderWarning("Rendering type is not geometry::PlanarPatch.");
        return false;
    }
    const geometry::PlanarPatch &patch =
            (const geometry::PlanarPatch &)geometry;
    if (patch.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty PlanarPatch.");
        return false;
    }
    points.resize(2);
    double line_length = option.point_size_ * 0.02 * view.GetBoundingBox().GetMaxExtent();
    points[0] = patch.center_.cast<float>();
    points[1] = (patch.center_ + patch.normal_ * line_length).cast<float>();

    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
