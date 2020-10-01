// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
// 4305: LightManager.h needs to specify some constants as floats
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293 4305)
#endif  // _MSC_VER

#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <utils/Entity.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <Eigen/Geometry>
#include <unordered_map>
#include <vector>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

/// @cond
namespace filament {
class Box;
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

class FilamentView;
class GeometryBuffersBuilder;
class Renderer;
class View;

// Contains renderable objects like geometry and lights
// Can have multiple views
class FilamentScene : public Scene {
public:
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;

    FilamentScene(filament::Engine& engine,
                  FilamentResourceManager& resource_mgr,
                  Renderer& renderer);
    ~FilamentScene();

    // NOTE: Temporarily needed to support old View interface for ImGUI
    ViewHandle AddView(std::int32_t x,
                       std::int32_t y,
                       std::uint32_t w,
                       std::uint32_t h) override;

    View* GetView(const ViewHandle& view_id) const override;
    void SetViewActive(const ViewHandle& view_id, bool is_active) override;
    void SetRenderOnce(const ViewHandle& view_id) override;
    void RemoveView(const ViewHandle& view_id) override;

    // Camera
    void AddCamera(const std::string& camera_name,
                   std::shared_ptr<Camera> cam) override;
    void RemoveCamera(const std::string& camera_name) override;
    void SetActiveCamera(const std::string& camera_name) override;

    // Scene geometry
    bool AddGeometry(const std::string& object_name,
                     const geometry::Geometry3D& geometry,
                     const Material& material,
                     const std::string& downsampled_name = "",
                     size_t downsample_threshold = SIZE_MAX) override;
    bool AddGeometry(const std::string& object_name,
                     const t::geometry::PointCloud& point_cloud,
                     const Material& material,
                     const std::string& downsampled_name = "",
                     size_t downsample_threshold = SIZE_MAX) override;
    bool AddGeometry(const std::string& object_name,
                     const TriangleMeshModel& model) override;
    bool HasGeometry(const std::string& object_name) const override;
    void UpdateGeometry(const std::string& object_name,
                        const t::geometry::PointCloud& point_cloud,
                        uint32_t update_flags) override;
    void RemoveGeometry(const std::string& object_name) override;
    void ShowGeometry(const std::string& object_name, bool show) override;
    bool GeometryIsVisible(const std::string& object_name) override;
    void SetGeometryTransform(const std::string& object_name,
                              const Transform& transform) override;
    Transform GetGeometryTransform(const std::string& object_name) override;
    geometry::AxisAlignedBoundingBox GetGeometryBoundingBox(
            const std::string& object_name) override;
    void GeometryShadows(const std::string& object_name,
                         bool cast_shadows,
                         bool receive_shadows) override;
    void OverrideMaterial(const std::string& object_name,
                          const Material& material) override;
    void QueryGeometry(std::vector<std::string>& geometry) override;
    void OverrideMaterialAll(const Material& material,
                             bool shader_only = true) override;

    // Lighting Environment
    bool AddPointLight(const std::string& light_name,
                       const Eigen::Vector3f& color,
                       const Eigen::Vector3f& position,
                       float intensity,
                       float falloff,
                       bool cast_shadows) override;
    bool AddSpotLight(const std::string& light_name,
                      const Eigen::Vector3f& color,
                      const Eigen::Vector3f& position,
                      const Eigen::Vector3f& direction,
                      float intensity,
                      float falloff,
                      float inner_cone_angle,
                      float outer_cone_angle,
                      bool cast_shadows) override;
    Light& GetLight(const std::string& light_name) override;
    void RemoveLight(const std::string& light_name) override;
    void UpdateLight(const std::string& light_name,
                     const Light& light) override;
    void UpdateLightColor(const std::string& light_name,
                          const Eigen::Vector3f& color) override;
    void UpdateLightPosition(const std::string& light_name,
                             const Eigen::Vector3f& position) override;
    void UpdateLightDirection(const std::string& light_name,
                              const Eigen::Vector3f& direction) override;
    void UpdateLightIntensity(const std::string& light_name,
                              float intensity) override;
    void UpdateLightFalloff(const std::string& light_name,
                            float falloff) override;
    void UpdateLightConeAngles(const std::string& light_name,
                               float inner_cone_angle,
                               float outer_cone_angle) override;
    void EnableLightShadow(const std::string& light_name,
                           bool cast_shadows) override;

    void SetDirectionalLight(const Eigen::Vector3f& direction,
                             const Eigen::Vector3f& color,
                             float intensity) override;
    void EnableDirectionalLight(bool enable) override;
    void EnableDirectionalLightShadows(bool enable) override;
    void SetDirectionalLightDirection(
            const Eigen::Vector3f& direction) override;
    Eigen::Vector3f GetDirectionalLightDirection() override;

    bool SetIndirectLight(const std::string& ibl_name) override;
    const std::string& GetIndirectLight() override;
    void EnableIndirectLight(bool enable) override;
    void SetIndirectLightIntensity(float intensity) override;
    float GetIndirectLightIntensity() override;
    void SetIndirectLightRotation(const Transform& rotation) override;
    Transform GetIndirectLightRotation() override;
    void ShowSkybox(bool show) override;

    void RenderToImage(int width,
                       int height,
                       std::function<void(std::shared_ptr<geometry::Image>)>
                               callback) override;

    void Draw(filament::Renderer& renderer);
    // NOTE: Can GetNativeScene be removed?
    filament::Scene* GetNativeScene() const { return scene_; }

private:
    MaterialInstanceHandle AssignMaterialToFilamentGeometry(
            filament::RenderableManager::Builder& builder,
            const Material& material);
    enum BufferReuse { kNo, kYes };
    bool CreateAndAddFilamentEntity(
            const std::string& object_name,
            GeometryBuffersBuilder& buffer_builder,
            filament::Box& aabb,
            VertexBufferHandle vb,
            IndexBufferHandle ib,
            const Material& material,
            BufferReuse reusing_vertex_buffer = BufferReuse::kNo);

    filament::Engine& engine_;
    FilamentResourceManager& resource_mgr_;
    filament::Scene* scene_ = nullptr;

    struct TextureMaps {
        rendering::TextureHandle albedo_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle normal_map =
                rendering::FilamentResourceManager::kDefaultNormalMap;
        rendering::TextureHandle ao_rough_metal_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle reflectance_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle clear_coat_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle clear_coat_roughness_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle anisotropy_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle gradient_texture =
                rendering::FilamentResourceManager::kDefaultTexture;
    };

    struct GeometryMaterialInstance {
        TextureMaps maps;
        Material properties;
        MaterialInstanceHandle mat_instance;
    };

    struct RenderableGeometry {
        std::string name;
        bool visible = true;
        bool cast_shadows = true;
        bool receive_shadow = true;

        GeometryMaterialInstance mat;

        // Filament resources
        utils::Entity filament_entity;
        VertexBufferHandle vb;
        IndexBufferHandle ib;
        void ReleaseResources(filament::Engine& engine,
                              FilamentResourceManager& manager);
    };

    struct LightEntity {
        bool enabled = true;
        utils::Entity filament_entity;
    };

    // NOTE: ViewContainer and views_ are temporary
    struct ViewContainer {
        std::unique_ptr<FilamentView> view;
        bool is_active = true;
        int render_count = -1;
    };
    std::unordered_map<REHandle_abstract, ViewContainer> views_;

    std::vector<RenderableGeometry*> GetGeometry(const std::string& object_name,
                                                 bool warn_if_not_found = true);
    bool GeometryIsModel(const std::string& object_name) const;
    LightEntity* GetLightInternal(const std::string& light_name,
                                  bool warn_if_not_found = true);
    void OverrideMaterialInternal(RenderableGeometry* geom,
                                  const Material& material,
                                  bool shader_only = false);
    void UpdateMaterialProperties(RenderableGeometry& geom);
    void UpdateDefaultLit(GeometryMaterialInstance& geom_mi);
    void UpdateDefaultUnlit(GeometryMaterialInstance& geom_mi);
    void UpdateNormalShader(GeometryMaterialInstance& geom_mi);
    void UpdateDepthShader(GeometryMaterialInstance& geom_mi);
    void UpdateGradientShader(GeometryMaterialInstance& geom_mi);
    void UpdateSolidColorShader(GeometryMaterialInstance& geom_mi);
    utils::EntityInstance<filament::TransformManager>
    GetGeometryTransformInstance(RenderableGeometry* geom);
    void CreateSunDirectionalLight();

    std::unordered_map<std::string, RenderableGeometry> geometries_;
    std::unordered_map<std::string, LightEntity> lights_;
    std::unordered_map<std::string, std::vector<std::string>> model_geometries_;

    std::string ibl_name_;
    bool ibl_enabled_ = false;
    bool skybox_enabled_ = false;
    std::weak_ptr<filament::IndirectLight> indirect_light_;
    std::weak_ptr<filament::Skybox> skybox_;
    LightEntity sun_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
