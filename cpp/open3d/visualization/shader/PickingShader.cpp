// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/shader/PickingShader.h"

#include "open3d/geometry/PointCloud.h"
#include "open3d/visualization/shader/Shader.h"
#include "open3d/visualization/utility/ColorMap.h"
#include "open3d/visualization/utility/GLHelper.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool PickingShader::Compile() {
    if (!CompileShaders(PickingVertexShader, NULL, PickingFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_index_ = glGetAttribLocation(program_, "vertex_index");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void PickingShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool PickingShader::BindGeometry(const geometry::Geometry &geometry,
                                 const RenderOption &option,
                                 const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    std::vector<Eigen::Vector3f> points;
    std::vector<float> indices;
    if (!PrepareBinding(geometry, option, view, points, indices)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_index_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_index_buffer_);
    glBufferData(GL_ARRAY_BUFFER, indices.size() * sizeof(float),
                 indices.data(), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool PickingShader::RenderGeometry(const geometry::Geometry &geometry,
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
    glEnableVertexAttribArray(vertex_index_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_index_buffer_);
    glVertexAttribPointer(vertex_index_, 1, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_index_);
    return true;
}

void PickingShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_index_buffer_);
        bound_ = false;
    }
}

bool PickingShaderForPointCloud::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool PickingShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<float> &indices) {
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
    points.resize(pointcloud.points_.size());
    indices.resize(pointcloud.points_.size());
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const auto &point = pointcloud.points_[i];
        points[i] = point.cast<float>();
        indices[i] = (float)i;
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
