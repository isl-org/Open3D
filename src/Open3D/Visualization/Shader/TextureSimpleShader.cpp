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

#include "Open3D/Visualization/Shader/TextureSimpleShader.h"

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Shader/Shader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"

namespace open3d {
namespace visualization {
namespace glsl {

bool TextureSimpleShader::Compile() {
    if (CompileShaders(TextureSimpleVertexShader, NULL,
                       TextureSimpleFragmentShader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_uv_ = glGetAttribLocation(program_, "vertex_uv");
    texture_ = glGetUniformLocation(program_, "diffuse_texture");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void TextureSimpleShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool TextureSimpleShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector2f> uvs;
    if (PrepareBinding(geometry, option, view, points, uvs) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_uv_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(Eigen::Vector2f),
                 uvs.data(), GL_STATIC_DRAW);
    bound_ = true;
    return true;
}

bool TextureSimpleShader::RenderGeometry(const geometry::Geometry &geometry,
                                         const RenderOption &option,
                                         const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());

    glUniform1i(texture_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_buffer_);

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_uv_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_uv_);
    return true;
}

void TextureSimpleShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteTextures(1, &texture_buffer_);
        bound_ = false;
    }
}

bool TextureSimpleShaderForTriangleMesh::PrepareRendering(
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
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool TextureSimpleShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector2f> &uvs) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    const geometry::TriangleMesh &mesh =
            (const geometry::TriangleMesh &)geometry;
    if (mesh.HasTriangles() == false) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    points.resize(mesh.triangles_.size() * 3);
    uvs.resize(mesh.triangles_.size() * 3);

    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t vi = triangle(j);
            points[idx] = mesh.vertices_[vi].cast<float>();
            uvs[idx] = mesh.triangle_uvs_[idx].cast<float>();
        }
    }

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_buffer_);

    GLenum format;
    switch (mesh.texture_.num_of_channels_) {
        case 1: {
            format = GL_RED;
            break;
        }
        case 3: {
            format = GL_RGB;
            break;
        }
        case 4: {
            format = GL_RGBA;
            break;
        }
        default: {
            utility::LogError("Unknown format, abort!\n");
            return false;
        }
    }

    GLenum type;
    switch (mesh.texture_.bytes_per_channel_) {
        case 1: {
            type = GL_UNSIGNED_BYTE;
            break;
        }
        case 2: {
            type = GL_UNSIGNED_SHORT;
            break;
        }
        case 4: {
            type = GL_FLOAT;
            break;
        }
        default: {
            utility::LogError("Unknown format, abort!\n");
            return false;
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, format, mesh.texture_.width_,
                 mesh.texture_.height_, 0, format, type,
                 mesh.texture_.data_.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl
}  // namespace visualization
}  // namespace open3d
