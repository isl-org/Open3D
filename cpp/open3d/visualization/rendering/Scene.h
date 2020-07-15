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

#include <Eigen/Geometry>

#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/RendererStructs.h"

namespace open3d {

namespace geometry {
class Geometry3D;
class AxisAlignedBoundingBox;
}  // namespace geometry

namespace visualization {
namespace rendering {

class View;

// Contains renderable objects like geometry and lights
// Can have multiple views
class Scene {
public:
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;

    virtual ~Scene() = default;

    virtual ViewHandle AddView(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) = 0;
    virtual View* GetView(const ViewHandle& view_id) const = 0;
    virtual void SetViewActive(const ViewHandle& view_id, bool is_active) = 0;
    virtual void RemoveView(const ViewHandle& view_id) = 0;

    // 'All defaults' way:
    // * Will use geometry name as entity name
    // * Will use apropriate default material for rendering
    // * For geometries with textures, will try load and assign a texture
    virtual GeometryHandle AddGeometry(
            const geometry::Geometry3D& geometry) = 0;
    // Will use geometry name as entity name
    virtual GeometryHandle AddGeometry(
            const geometry::Geometry3D& geometry,
            const MaterialInstanceHandle& material_id) = 0;
    virtual GeometryHandle AddGeometry(
            const geometry::Geometry3D& geometry,
            const MaterialInstanceHandle& material_id,
            const std::string& name) = 0;
    virtual void AssignMaterial(const GeometryHandle& geometry_id,
                                const MaterialInstanceHandle& material_id) = 0;
    virtual MaterialInstanceHandle GetMaterial(
            const GeometryHandle& geometry_id) const = 0;
    virtual void SetGeometryShadows(const GeometryHandle& geometry_id,
                                    bool casts_shadows,
                                    bool receives_shadows) = 0;
    virtual std::vector<GeometryHandle> FindGeometryByName(
            const std::string& name) = 0;
    virtual void RemoveGeometry(const GeometryHandle& geometry_id) = 0;

    virtual LightHandle AddLight(const LightDescription& descr) = 0;
    // TODO: If possible, add getters
    virtual void SetLightIntensity(const LightHandle& id, float intensity) = 0;
    virtual void SetLightColor(const LightHandle& id,
                               const Eigen::Vector3f& color) = 0;
    virtual Eigen::Vector3f GetLightDirection(const LightHandle& id) const = 0;
    virtual void SetLightDirection(const LightHandle& id,
                                   const Eigen::Vector3f& dir) = 0;
    virtual void SetLightPosition(const LightHandle& id,
                                  const Eigen::Vector3f& pos) = 0;
    virtual void SetLightFalloff(const LightHandle& id, float falloff) = 0;
    virtual void RemoveLight(const LightHandle& id) = 0;

    // Passing empty id disables indirect lightning
    virtual void SetIndirectLight(const IndirectLightHandle& id) = 0;
    virtual void SetIndirectLightIntensity(float intensity) = 0;
    virtual float GetIndirectLightIntensity() const = 0;
    virtual void SetIndirectLightRotation(const Transform& rotation) = 0;
    virtual Transform GetIndirectLightRotation() const = 0;

    // Passing empty id removes skybox
    virtual void SetSkybox(const SkyboxHandle& id) = 0;

    virtual void SetEntityEnabled(const REHandle_abstract& entity_id,
                                  bool enabled) = 0;
    virtual bool GetEntityEnabled(const REHandle_abstract& entity_id) = 0;
    virtual void SetEntityTransform(const REHandle_abstract& entity_id,
                                    const Transform& transform) = 0;
    virtual Transform GetEntityTransform(
            const REHandle_abstract& entity_id) = 0;

    // Returns world space AABB
    virtual geometry::AxisAlignedBoundingBox GetEntityBoundingBox(
            const REHandle_abstract& entity_id) = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
