// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "open3d/visualization/shader/ShaderWrapper.h"

namespace open3d {
namespace visualization {

namespace glsl {

class SimpleBlackShader : public ShaderWrapper {
public:
    ~SimpleBlackShader() override { Release(); }

protected:
    SimpleBlackShader(const std::string &name) : ShaderWrapper(name) {
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
                                std::vector<Eigen::Vector3f> &points) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint MVP_;
};

class SimpleBlackShaderForPointCloudNormal : public SimpleBlackShader {
public:
    SimpleBlackShaderForPointCloudNormal()
        : SimpleBlackShader("SimpleBlackShaderForPointCloudNormal") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points) final;
};

class SimpleBlackShaderForTriangleMeshWireFrame : public SimpleBlackShader {
public:
    SimpleBlackShaderForTriangleMeshWireFrame()
        : SimpleBlackShader("SimpleBlackShaderForTriangleMeshWireFrame") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
