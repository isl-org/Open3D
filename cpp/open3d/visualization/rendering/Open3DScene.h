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

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/visualization/rendering/Renderer.h"

namespace open3d {

namespace geometry {
class Geometry3D;
}  // namespace geometry

namespace visualization {

class Camera;

class Open3DScene {
public:
    Open3DScene(Renderer& renderer);
    ~Open3DScene();

    ViewHandle CreateView();
    void DestroyView(ViewHandle view);
    View* GetView(ViewHandle view) const;

    void SetIndirectLight(IndirectLightHandle ibl);
    void SetSkybox(SkyboxHandle skybox);

    void ClearGeometry();
    GeometryHandle AddGeometry(
            std::shared_ptr<const geometry::Geometry3D> geom,
            const MaterialInstanceHandle& material_id,
            bool add_downsampled_copy_for_fast_rendering = true);

    enum class LOD {
        HIGH_DETAIL,  // used when rendering time is not as important
        FAST,         // used when rendering time is important, like rotating
    };
    void SetLOD(LOD lod);
    LOD GetLOD() const;

    Scene* GetScene() const;
    Camera* GetCamera() const;
    const std::vector<GeometryHandle>& GetModel(
            LOD lod = LOD::HIGH_DETAIL) const;
    GeometryHandle GetAxis() const;
    SkyboxHandle GetSkybox() const;
    IndirectLightHandle GetIndirectLight() const;
    LightHandle GetSun() const;

private:
    Renderer& renderer_;
    SceneHandle scene_;
    ViewHandle view_;
    mutable GeometryHandle axis_;
    IndirectLightHandle ibl_;
    SkyboxHandle skybox_;
    LightHandle sun_;

    LOD lod_ = LOD::HIGH_DETAIL;
    std::vector<GeometryHandle> model_;
    std::vector<GeometryHandle> fast_model_;
    geometry::AxisAlignedBoundingBox bounds_;
};

}  // namespace visualization
}  // namespace open3d
