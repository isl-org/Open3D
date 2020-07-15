// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <utils/Entity.h>
#include <utils/EntityInstance.h>

#include <memory>
#include <unordered_map>

#include "open3d/visualization/rendering/Scene.h"

/// @cond
namespace filament {
class Engine;
class IndirectLight;
class Renderer;
class Scene;
class Skybox;
class TransformManager;
class VertexBuffer;
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;
class FilamentView;

class FilamentScene : public Scene {
public:
    FilamentScene(filament::Engine& engine,
                  FilamentResourceManager& resource_mgr);
    ~FilamentScene() override;

    // All views above first will discard
    // only depth and stencil buffers by default
    ViewHandle AddView(std::int32_t x,
                       std::int32_t y,
                       std::uint32_t w,
                       std::uint32_t h) override;

    View* GetView(const ViewHandle& view_id) const override;
    void SetViewActive(const ViewHandle& view_id, bool is_active) override;
    void RemoveView(const ViewHandle& view_id) override;

    GeometryHandle AddGeometry(const geometry::Geometry3D& geometry) override;
    GeometryHandle AddGeometry(
            const geometry::Geometry3D& geometry,
            const MaterialInstanceHandle& material_id) override;
    GeometryHandle AddGeometry(const geometry::Geometry3D& geometry,
                               const MaterialInstanceHandle& material_id,
                               const std::string& name) override;
    std::vector<GeometryHandle> FindGeometryByName(
            const std::string& name) override;
    void AssignMaterial(const GeometryHandle& geometry_id,
                        const MaterialInstanceHandle& material_id) override;
    MaterialInstanceHandle GetMaterial(
            const GeometryHandle& geometry_id) const override;
    void SetGeometryShadows(const GeometryHandle& geometry_id,
                            bool casts_shadows,
                            bool receives_shadows) override;
    void RemoveGeometry(const GeometryHandle& geometry_id) override;

    LightHandle AddLight(const LightDescription& descr) override;
    void SetLightIntensity(const LightHandle& id, float intensity) override;
    void SetLightColor(const LightHandle& id,
                       const Eigen::Vector3f& color) override;
    Eigen::Vector3f GetLightDirection(const LightHandle& id) const override;
    void SetLightDirection(const LightHandle& id,
                           const Eigen::Vector3f& dir) override;
    void SetLightPosition(const LightHandle& id,
                          const Eigen::Vector3f& pos) override;
    void SetLightFalloff(const LightHandle& id, float falloff) override;
    void RemoveLight(const LightHandle& id) override;

    void SetIndirectLight(const IndirectLightHandle& id) override;
    void SetIndirectLightIntensity(float intensity) override;
    float GetIndirectLightIntensity() const override;
    void SetIndirectLightRotation(const Transform& rotation) override;
    Transform GetIndirectLightRotation() const override;

    void SetSkybox(const SkyboxHandle& id) override;

    void SetEntityEnabled(const REHandle_abstract& entity_id,
                          bool enabled) override;
    bool GetEntityEnabled(const REHandle_abstract& entity_id) override;
    void SetEntityTransform(const REHandle_abstract& entity_id,
                            const Transform& transform) override;
    Transform GetEntityTransform(const REHandle_abstract& entity_id) override;

    geometry::AxisAlignedBoundingBox GetEntityBoundingBox(
            const REHandle_abstract& entity_id) override;

    void Draw(filament::Renderer& renderer);

    filament::Scene* GetNativeScene() const { return scene_; }

private:
    friend class FilamentView;

    struct SceneEntity {
        struct Details {
            utils::Entity self;
            EntityType type;
            VertexBufferHandle vb;
            IndexBufferHandle ib;

            bool IsValid() const { return !self.isNull(); }
            void ReleaseResources(filament::Engine& engine,
                                  FilamentResourceManager& manager);
        } info;

        // We can disable entities removing them from scene, but not
        // deallocating
        bool enabled = true;
        MaterialInstanceHandle material;
        TextureHandle texture;  // if none, default is used
        // Used for relocating transform to center of mass
        utils::Entity parent;
        std::string name;

        bool IsValid() const { return info.IsValid(); }
        void ReleaseResources(filament::Engine& engine,
                              FilamentResourceManager& manager);
    };

    struct ViewContainer {
        std::unique_ptr<FilamentView> view;
        bool is_active = true;
    };

    utils::EntityInstance<filament::TransformManager>
    GetEntityTransformInstance(const REHandle_abstract& id);
    void RemoveEntity(REHandle_abstract id);

    filament::Scene* scene_ = nullptr;

    filament::Engine& engine_;
    FilamentResourceManager& resource_mgr_;

    std::unordered_map<REHandle_abstract, ViewContainer> views_;
    std::unordered_map<REHandle_abstract, SceneEntity> entities_;
    std::weak_ptr<filament::IndirectLight> indirect_light_;
    std::weak_ptr<filament::Skybox> skybox_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
