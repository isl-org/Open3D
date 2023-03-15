// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <GL/glew.h>

#include "open3d/geometry/Geometry.h"
#include "open3d/visualization/visualizer/RenderOption.h"
#include "open3d/visualization/visualizer/ViewControl.h"

namespace open3d {
namespace visualization {

namespace glsl {

class ShaderWrapper {
public:
    virtual ~ShaderWrapper() {}
    ShaderWrapper(const ShaderWrapper &) = delete;
    ShaderWrapper &operator=(const ShaderWrapper &) = delete;

protected:
    ShaderWrapper(const std::string &name) : shader_name_(name) {}

public:
    /// Function to render geometry under condition of mode and view
    /// The geometry is updated in a passive manner (bind only when needed).
    /// Thus this function compiles shaders if not yet, binds geometry if not
    /// yet, then do the rendering.
    bool Render(const geometry::Geometry &geometry,
                const RenderOption &option,
                const ViewControl &view);

    /// Function to invalidate the geometry (set the dirty flag and release
    /// geometry resource)
    void InvalidateGeometry();

    const std::string &GetShaderName() const { return shader_name_; }

    void PrintShaderWarning(const std::string &message) const;

protected:
    /// Function to compile shader
    /// In a derived class, this must be declared as final, and called from
    /// the constructor.
    virtual bool Compile() = 0;

    /// Function to release resource
    /// In a derived class, this must be declared as final, and called from
    /// the destructor.
    virtual void Release() = 0;

    virtual bool BindGeometry(const geometry::Geometry &geometry,
                              const RenderOption &option,
                              const ViewControl &view) = 0;
    virtual bool RenderGeometry(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view) = 0;
    virtual void UnbindGeometry() = 0;

protected:
    bool ValidateShader(GLuint shader_index);
    bool ValidateProgram(GLuint program_index);
    bool CompileShaders(const char *const vertex_shader_code,
                        const char *const geometry_shader_code,
                        const char *const fragment_shader_code);
    void ReleaseProgram();

protected:
    GLuint vertex_shader_ = 0;
    GLuint geometry_shader_ = 0;
    GLuint fragment_shader_ = 0;
    GLuint program_ = 0;
    GLenum draw_arrays_mode_ = GL_POINTS;
    GLsizei draw_arrays_size_ = 0;
    bool compiled_ = false;
    bool bound_ = false;

    void SetShaderName(const std::string &shader_name) {
        shader_name_ = shader_name;
    }

private:
    std::string shader_name_ = "ShaderWrapper";
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
