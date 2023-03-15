// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "open3d/visualization/shader/ShaderWrapper.h"

namespace open3d {
namespace visualization {

namespace glsl {

class TexturePhongShader : public ShaderWrapper {
public:
    ~TexturePhongShader() override { Release(); }

protected:
    TexturePhongShader(const std::string &name) : ShaderWrapper(name) {
        Compile();
    }

protected:
    bool Compile() final;
    void Release() final;
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
    void UnbindGeometry() final;

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                std::vector<Eigen::Vector3f> &points,
                                std::vector<Eigen::Vector3f> &normals,
                                std::vector<Eigen::Vector2f> &uvs) = 0;

protected:
    void SetLighting(const ViewControl &view, const RenderOption &option);

protected:
    GLuint vertex_position_;
    GLuint vertex_uv_;
    GLuint vertex_normal_;
    GLuint MVP_;
    GLuint V_;
    GLuint M_;
    GLuint light_position_world_;
    GLuint light_color_;
    GLuint light_diffuse_power_;
    GLuint light_specular_power_;
    GLuint light_specular_shininess_;
    GLuint light_ambient_;

    GLuint diffuse_texture_;

    int num_materials_;
    std::vector<int> array_offsets_;
    std::vector<GLsizei> draw_array_sizes_;

    std::vector<GLuint> vertex_position_buffers_;
    std::vector<GLuint> vertex_uv_buffers_;
    std::vector<GLuint> vertex_normal_buffers_;
    std::vector<GLuint> diffuse_texture_buffers_;

    // At most support 4 lights
    gl_util::GLMatrix4f light_position_world_data_;
    gl_util::GLMatrix4f light_color_data_;
    gl_util::GLVector4f light_diffuse_power_data_;
    gl_util::GLVector4f light_specular_power_data_;
    gl_util::GLVector4f light_specular_shininess_data_;
    gl_util::GLVector4f light_ambient_data_;
};

class TexturePhongShaderForTriangleMesh : public TexturePhongShader {
public:
    TexturePhongShaderForTriangleMesh()
        : TexturePhongShader("TexturePhongShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector2f> &uvs) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
