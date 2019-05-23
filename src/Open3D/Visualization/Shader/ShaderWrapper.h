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

#include <GL/glew.h>

#include "Open3D/Geometry/Geometry.h"
#include "Open3D/Visualization/Visualizer/RenderOption.h"
#include "Open3D/Visualization/Visualizer/ViewControl.h"

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
    GLuint vertex_shader_;
    GLuint geometry_shader_;
    GLuint fragment_shader_;
    GLuint program_;
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
