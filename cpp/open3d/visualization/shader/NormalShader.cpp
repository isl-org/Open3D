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

#include "open3d/visualization/shader/NormalShader.h"

#include "open3d/geometry/PlanarPatch.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/shader/Shader.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool NormalShader::Compile() {
    if (!CompileShaders(NormalVertexShader, NULL, NormalFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    return true;
}

void NormalShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool NormalShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector3f> normals;
    if (!PrepareBinding(geometry, option, view, points, normals)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_normal_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(Eigen::Vector3f),
                 normals.data(), GL_STATIC_DRAW);
    bound_ = true;
    return true;
}

bool NormalShader::RenderGeometry(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    return true;
}

void NormalShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        bound_ = false;
    }
}

bool NormalShaderForPointCloud::PrepareRendering(
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
    glPointSize(GLfloat(option.point_size_));
    return true;
}

bool NormalShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &normals) {
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
    if (!pointcloud.HasNormals()) {
        PrintShaderWarning("Binding failed with pointcloud with no normals.");
        return false;
    }
    points.resize(pointcloud.points_.size());
    normals.resize(pointcloud.points_.size());
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const auto &point = pointcloud.points_[i];
        const auto &normal = pointcloud.normals_[i];
        points[i] = point.cast<float>();
        normals[i] = normal.cast<float>();
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool NormalShaderForTriangleMesh::PrepareRendering(
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
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool NormalShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &normals) {
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
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    if (!mesh.HasTriangleNormals() || !mesh.HasVertexNormals()) {
        PrintShaderWarning("Binding failed because mesh has no normals.");
        PrintShaderWarning("Call ComputeVertexNormals() before binding.");
        return false;
    }
    points.resize(mesh.triangles_.size() * 3);
    normals.resize(mesh.triangles_.size() * 3);
    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t vi = triangle(j);
            const auto &vertex = mesh.vertices_[vi];
            points[idx] = vertex.cast<float>();
            if (option.mesh_shade_option_ ==
                RenderOption::MeshShadeOption::FlatShade) {
                normals[idx] = mesh.triangle_normals_[i].cast<float>();
            } else {
                normals[idx] = mesh.vertex_normals_[vi].cast<float>();
            }
        }
    }
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool NormalShaderForPlanarPatch::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PlanarPatch) {
        PrintShaderWarning("Rendering type is not geometry::PlanarPatch.");
        return false;
    }
    // if (option.mesh_show_back_face_) {
    //     glDisable(GL_CULL_FACE);
    // } else {
    // glEnable(GL_CULL_FACE);
    // }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // if (option.mesh_show_wireframe_) {
    //     glEnable(GL_POLYGON_OFFSET_FILL);
    //     glPolygonOffset(1.0, 1.0);
    // } else {
    glDisable(GL_POLYGON_OFFSET_FILL);
    // }
    return true;
}

bool NormalShaderForPlanarPatch::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &normals) {
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
    points.resize(1);
    normals.resize(1);
    normals[0].z() = 1;
    // for (size_t i = 0; i < mesh.triangles_.size(); i++) {
    //     const auto &triangle = mesh.triangles_[i];
    //     for (size_t j = 0; j < 3; j++) {
    //         size_t idx = i * 3 + j;
    //         size_t vi = triangle(j);
    //         const auto &vertex = mesh.vertices_[vi];
    //         points[idx] = vertex.cast<float>();
    //         if (option.mesh_shade_option_ ==
    //             RenderOption::MeshShadeOption::FlatShade) {
    //             normals[idx] = mesh.triangle_normals_[i].cast<float>();
    //         } else {
    //             normals[idx] = mesh.vertex_normals_[vi].cast<float>();
    //         }
    //     }
    // }
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
