// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace geometry {
class Geometry3D;
class AxisAlignedBoundingBox;
class Image;
}  // namespace geometry

namespace t {
namespace geometry {
class Geometry;
class PointCloud;
}  // namespace geometry
}  // namespace t

namespace visualization {
namespace rendering {

class Renderer;
class View;
struct TriangleMeshModel;
struct MaterialRecord;
struct Light;

// Contains renderable objects like geometry and lights
// Can have multiple views
class Scene {
public:
    static const uint32_t kUpdatePointsFlag = (1 << 0);
    static const uint32_t kUpdateNormalsFlag = (1 << 1);
    static const uint32_t kUpdateColorsFlag = (1 << 2);
    static const uint32_t kUpdateUv0Flag = (1 << 3);

    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;

    Scene(Renderer& renderer) : renderer_(renderer) {}
    virtual ~Scene() = default;

    virtual Scene* Copy() = 0;

    // NOTE: Temporarily need to support old View interface for ImGUI
    virtual ViewHandle AddView(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) = 0;

    virtual View* GetView(const ViewHandle& view_id) const = 0;
    virtual void SetViewActive(const ViewHandle& view_id, bool is_active) = 0;
    virtual void SetRenderOnce(const ViewHandle& view_id) = 0;
    virtual void RemoveView(const ViewHandle& view_id) = 0;

    // Camera
    virtual void AddCamera(const std::string& camera_name,
                           std::shared_ptr<Camera> cam) = 0;
    virtual void RemoveCamera(const std::string& camera_name) = 0;
    virtual void SetActiveCamera(const std::string& camera_name) = 0;

    // Scene geometry
    virtual bool AddGeometry(const std::string& object_name,
                             const geometry::Geometry3D& geometry,
                             const MaterialRecord& material,
                             const std::string& downsampled_name = "",
                             size_t downsample_threshold = SIZE_MAX) = 0;
    virtual bool AddGeometry(const std::string& object_name,
                             const t::geometry::Geometry& geometry,
                             const MaterialRecord& material,
                             const std::string& downsampled_name = "",
                             size_t downsample_threshold = SIZE_MAX) = 0;
    virtual bool AddGeometry(const std::string& object_name,
                             const TriangleMeshModel& model) = 0;
    virtual bool HasGeometry(const std::string& object_name) const = 0;
    virtual void UpdateGeometry(const std::string& object_name,
                                const t::geometry::PointCloud& point_cloud,
                                uint32_t update_flags) = 0;
    virtual void RemoveGeometry(const std::string& object_name) = 0;
    virtual void ShowGeometry(const std::string& object_name, bool show) = 0;
    virtual bool GeometryIsVisible(const std::string& object_name) = 0;
    virtual void OverrideMaterial(const std::string& object_name,
                                  const MaterialRecord& material) = 0;
    virtual void GeometryShadows(const std::string& object_name,
                                 bool cast_shadows,
                                 bool receive_shadows) = 0;
    virtual void SetGeometryCulling(const std::string& object_name,
                                    bool enable) = 0;
    virtual void SetGeometryPriority(const std::string& object_name,
                                     uint8_t priority) = 0;
    virtual void QueryGeometry(std::vector<std::string>& geometry) = 0;
    virtual void SetGeometryTransform(const std::string& object_name,
                                      const Transform& transform) = 0;
    virtual Transform GetGeometryTransform(const std::string& object_name) = 0;
    virtual geometry::AxisAlignedBoundingBox GetGeometryBoundingBox(
            const std::string& object_name) = 0;
    virtual void OverrideMaterialAll(const MaterialRecord& material,
                                     bool shader_only = true) = 0;

    // Lighting Environment
    virtual bool AddPointLight(const std::string& light_name,
                               const Eigen::Vector3f& color,
                               const Eigen::Vector3f& position,
                               float intensity,
                               float falloff,
                               bool cast_shadows) = 0;
    virtual bool AddSpotLight(const std::string& light_name,
                              const Eigen::Vector3f& color,
                              const Eigen::Vector3f& position,
                              const Eigen::Vector3f& direction,
                              float intensity,
                              float falloff,
                              float inner_cone_angle,
                              float outer_cone_angle,
                              bool cast_shadows) = 0;
    virtual bool AddDirectionalLight(const std::string& light_name,
                                     const Eigen::Vector3f& color,
                                     const Eigen::Vector3f& direction,
                                     float intensity,
                                     bool cast_shadows) = 0;
    virtual Light& GetLight(const std::string& light_name) = 0;
    virtual void RemoveLight(const std::string& light_name) = 0;
    virtual void UpdateLight(const std::string& light_name,
                             const Light& light) = 0;
    virtual void UpdateLightColor(const std::string& light_name,
                                  const Eigen::Vector3f& color) = 0;
    virtual void UpdateLightPosition(const std::string& light_name,
                                     const Eigen::Vector3f& position) = 0;
    virtual void UpdateLightDirection(const std::string& light_name,
                                      const Eigen::Vector3f& direction) = 0;
    virtual void UpdateLightIntensity(const std::string& light_name,
                                      float intensity) = 0;
    virtual void UpdateLightFalloff(const std::string& light_name,
                                    float falloff) = 0;
    virtual void UpdateLightConeAngles(const std::string& light_name,
                                       float inner_cone_angle,
                                       float outer_cone_angle) = 0;
    virtual void EnableLightShadow(const std::string& light_name,
                                   bool cast_shadows) = 0;

    virtual void SetSunLight(const Eigen::Vector3f& direction,
                             const Eigen::Vector3f& color,
                             float intensity) = 0;
    virtual void EnableSunLight(bool enable) = 0;
    virtual void EnableSunLightShadows(bool enable) = 0;
    virtual void SetSunLightColor(const Eigen::Vector3f& color) = 0;
    virtual Eigen::Vector3f GetSunLightColor() = 0;
    virtual void SetSunLightIntensity(float intensity) = 0;
    virtual float GetSunLightIntensity() = 0;
    virtual void SetSunLightDirection(const Eigen::Vector3f& direction) = 0;
    virtual Eigen::Vector3f GetSunLightDirection() = 0;
    virtual void SetSunAngularRadius(float radius) = 0;
    virtual void SetSunHaloSize(float size) = 0;
    virtual void SetSunHaloFalloff(float falloff) = 0;

    virtual bool SetIndirectLight(const std::string& ibl_name) = 0;
    virtual const std::string& GetIndirectLight() = 0;
    virtual void EnableIndirectLight(bool enable) = 0;
    virtual void SetIndirectLightIntensity(float intensity) = 0;
    virtual float GetIndirectLightIntensity() = 0;
    virtual void SetIndirectLightRotation(const Transform& rotation) = 0;
    virtual Transform GetIndirectLightRotation() = 0;
    virtual void ShowSkybox(bool show) = 0;
    virtual bool GetSkyboxVisible() const = 0;
    virtual void SetBackground(
            const Eigen::Vector4f& color,
            const std::shared_ptr<geometry::Image> image = nullptr) = 0;
    virtual void SetBackground(TextureHandle image) = 0;

    enum class GroundPlane { XZ, XY, YZ };
    virtual void EnableGroundPlane(bool enable, GroundPlane plane) = 0;
    virtual void SetGroundPlaneColor(const Eigen::Vector4f& color) = 0;

    /// Size of image is the size of the window.
    virtual void RenderToImage(
            std::function<void(std::shared_ptr<geometry::Image>)> callback) = 0;

    /// Size of image is the size of the window.
    virtual void RenderToDepthImage(
            std::function<void(std::shared_ptr<geometry::Image>)> callback) = 0;

protected:
    Renderer& renderer_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
