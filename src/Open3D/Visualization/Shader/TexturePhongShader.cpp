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

#include "Open3D/Visualization/Shader/TexturePhongShader.h"

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Shader/Shader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool TexturePhongShader::Compile() {
    if (CompileShaders(TexturePhongVertexShader, NULL,
                       TexturePhongFragmentShader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    vertex_uv_ = glGetAttribLocation(program_, "vertex_uv");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    light_position_world_ =
            glGetUniformLocation(program_, "light_position_world_4");
    light_color_ = glGetUniformLocation(program_, "light_color_4");
    light_diffuse_power_ =
            glGetUniformLocation(program_, "light_diffuse_power_4");
    light_specular_power_ =
            glGetUniformLocation(program_, "light_specular_power_4");
    light_specular_shininess_ =
            glGetUniformLocation(program_, "light_specular_shininess_4");
    light_ambient_ = glGetUniformLocation(program_, "light_ambient");

    diffuse_texture_ = glGetUniformLocation(program_, "diffuse_texture");
    return true;
}

void TexturePhongShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool TexturePhongShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector2f> uvs;

    if (PrepareBinding(geometry, option, view, points, normals, uvs) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    for (int mi = 0; mi < num_materials_; ++mi) {
        glGenBuffers(1, &vertex_position_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector3f),
                     points.data() + array_offsets_[mi], GL_STATIC_DRAW);

        glGenBuffers(1, &vertex_normal_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector3f),
                     normals.data() + array_offsets_[mi], GL_STATIC_DRAW);

        glGenBuffers(1, &vertex_uv_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector2f),
                     uvs.data() + array_offsets_[mi], GL_STATIC_DRAW);
    }
    bound_ = true;
    return true;
}

bool TexturePhongShader::RenderGeometry(const geometry::Geometry &geometry,
                                        const RenderOption &option,
                                        const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    for (int mi = 0; mi < num_materials_; ++mi) {
        glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
        glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
        glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
        glUniformMatrix4fv(light_position_world_, 1, GL_FALSE,
                           light_position_world_data_.data());
        glUniformMatrix4fv(light_color_, 1, GL_FALSE, light_color_data_.data());
        glUniform4fv(light_diffuse_power_, 1, light_diffuse_power_data_.data());
        glUniform4fv(light_specular_power_, 1,
                     light_specular_power_data_.data());
        glUniform4fv(light_specular_shininess_, 1,
                     light_specular_shininess_data_.data());
        glUniform4fv(light_ambient_, 1, light_ambient_data_.data());

        glUniform1i(diffuse_texture_, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuse_texture_buffers_[mi]);

        glEnableVertexAttribArray(vertex_position_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
        glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glEnableVertexAttribArray(vertex_normal_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffers_[mi]);
        glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glEnableVertexAttribArray(vertex_uv_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
        glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glDrawArrays(draw_arrays_mode_, 0, draw_array_sizes_[mi]);

        glDisableVertexAttribArray(vertex_position_);
        glDisableVertexAttribArray(vertex_normal_);
        glDisableVertexAttribArray(vertex_uv_);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    return true;
}

void TexturePhongShader::UnbindGeometry() {
    if (bound_) {
        for (auto buf : vertex_position_buffers_) {
            glDeleteBuffers(1, &buf);
        }
        for (auto buf : vertex_normal_buffers_) {
            glDeleteBuffers(1, &buf);
        }
        for (auto buf : vertex_uv_buffers_) {
            glDeleteBuffers(1, &buf);
        }
        for (auto buf : diffuse_texture_buffers_) {
            glDeleteTextures(1, &buf);
        }

        vertex_position_buffers_.clear();
        vertex_normal_buffers_.clear();
        vertex_uv_buffers_.clear();
        diffuse_texture_buffers_.clear();
        draw_array_sizes_.clear();
        array_offsets_.clear();
        num_materials_ = 0;
        bound_ = false;
    }
}

void TexturePhongShader::SetLighting(const ViewControl &view,
                                     const RenderOption &option) {
    const auto &box = view.GetBoundingBox();
    light_position_world_data_.setOnes();
    light_color_data_.setOnes();
    for (int i = 0; i < 4; i++) {
        light_position_world_data_.block<3, 1>(0, i) =
                box.GetCenter().cast<GLfloat>() +
                (float)box.GetMaxExtent() *
                        ((float)option.light_position_relative_[i](0) *
                                 view.GetRight() +
                         (float)option.light_position_relative_[i](1) *
                                 view.GetUp() +
                         (float)option.light_position_relative_[i](2) *
                                 view.GetFront());
        light_color_data_.block<3, 1>(0, i) =
                option.light_color_[i].cast<GLfloat>();
    }
    if (option.light_on_) {
        light_diffuse_power_data_ =
                Eigen::Vector4d(option.light_diffuse_power_).cast<GLfloat>();
        light_specular_power_data_ =
                Eigen::Vector4d(option.light_specular_power_).cast<GLfloat>();
        light_specular_shininess_data_ =
                Eigen::Vector4d(option.light_specular_shininess_)
                        .cast<GLfloat>();
        light_ambient_data_.block<3, 1>(0, 0) =
                option.light_ambient_color_.cast<GLfloat>();
        light_ambient_data_(3) = 1.0f;
    } else {
        light_diffuse_power_data_ = GLHelper::GLVector4f::Zero();
        light_specular_power_data_ = GLHelper::GLVector4f::Zero();
        light_specular_shininess_data_ = GLHelper::GLVector4f::Ones();
        light_ambient_data_ = GLHelper::GLVector4f(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

bool TexturePhongShaderForTriangleMesh::PrepareRendering(
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
    SetLighting(view, option);
    return true;
}

bool TexturePhongShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &normals,
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
    if (mesh.HasTriangleNormals() == false ||
        mesh.HasVertexNormals() == false) {
        PrintShaderWarning("Binding failed because mesh has no normals.");
        PrintShaderWarning("Call ComputeVertexNormals() before binding.");
        return false;
    }
    std::vector<std::vector<Eigen::Vector3f>> tmp_points;
    std::vector<std::vector<Eigen::Vector3f>> tmp_normals;
    std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;

    num_materials_ = (int)mesh.textures_.size();
    array_offsets_.resize(num_materials_);
    draw_array_sizes_.resize(num_materials_);
    vertex_position_buffers_.resize(num_materials_);
    vertex_normal_buffers_.resize(num_materials_);
    vertex_uv_buffers_.resize(num_materials_);
    diffuse_texture_buffers_.resize(num_materials_);

    tmp_points.resize(num_materials_);
    tmp_normals.resize(num_materials_);
    tmp_uvs.resize(num_materials_);

    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        int mi = mesh.triangle_material_ids_[i];

        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t vi = triangle(j);

            tmp_points[mi].push_back(mesh.vertices_[vi].cast<float>());
            tmp_uvs[mi].push_back(mesh.triangle_uvs_[idx].cast<float>());

            if (option.mesh_shade_option_ ==
                RenderOption::MeshShadeOption::FlatShade) {
                tmp_normals[mi].push_back(
                        mesh.triangle_normals_[i].cast<float>());
            } else {
                tmp_normals[mi].push_back(
                        mesh.vertex_normals_[vi].cast<float>());
            }
        }
    }

    // Bind textures
    for (int mi = 0; mi < num_materials_; ++mi) {
        glGenTextures(1, &diffuse_texture_buffers_[mi]);
        glBindTexture(GL_TEXTURE_2D, diffuse_texture_buffers_[mi]);

        GLenum format, type;
        auto it = GLHelper::texture_format_map_.find(
                mesh.textures_[mi].num_of_channels_);
        if (it == GLHelper::texture_format_map_.end()) {
            utility::LogWarning("Unknown texture format, abort!");
            return false;
        }
        format = it->second;

        it = GLHelper::texture_type_map_.find(
                mesh.textures_[mi].bytes_per_channel_);
        if (it == GLHelper::texture_type_map_.end()) {
            utility::LogWarning("Unknown texture type, abort!");
            return false;
        }
        type = it->second;

        glTexImage2D(GL_TEXTURE_2D, 0, format, mesh.textures_[mi].width_,
                     mesh.textures_[mi].height_, 0, format, type,
                     mesh.textures_[mi].data_.data());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Point seperations
    array_offsets_[0] = 0;
    draw_array_sizes_[0] = tmp_points[0].size();
    for (int mi = 1; mi < num_materials_; ++mi) {
        draw_array_sizes_[mi] = tmp_points[mi].size();
        array_offsets_[mi] = array_offsets_[mi - 1] + draw_array_sizes_[mi - 1];
    }

    // Prepared chunk of points and uvs
    points.clear();
    uvs.clear();
    normals.clear();
    for (int mi = 0; mi < num_materials_; ++mi) {
        points.insert(points.end(), tmp_points[mi].begin(),
                      tmp_points[mi].end());
        uvs.insert(uvs.end(), tmp_uvs[mi].begin(), tmp_uvs[mi].end());
        normals.insert(normals.end(), tmp_normals[mi].begin(),
                       tmp_normals[mi].end());
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
