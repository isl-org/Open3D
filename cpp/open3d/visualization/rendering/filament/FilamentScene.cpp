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

#include "open3d/visualization/rendering/filament/FilamentScene.h"

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <utils/EntityManager.h>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

namespace {  // avoid polluting global namespace, since only used here
/// @cond

namespace defaults_mapping {

using GeometryType = open3d::geometry::Geometry::GeometryType;
using MaterialHandle = open3d::visualization::rendering::MaterialHandle;
using ResourceManager =
        open3d::visualization::rendering::FilamentResourceManager;

MaterialHandle kColorOnlyMesh = ResourceManager::kDefaultUnlit;
MaterialHandle kPlainMesh = ResourceManager::kDefaultLit;
MaterialHandle kMesh = ResourceManager::kDefaultLit;

MaterialHandle kColoredPointcloud = ResourceManager::kDefaultUnlit;
MaterialHandle kPointcloud = ResourceManager::kDefaultLit;

MaterialHandle kLineset = ResourceManager::kDefaultUnlit;

}  // namespace defaults_mapping

namespace converters {
using EigenMatrix =
        open3d::visualization::rendering::FilamentScene::Transform::MatrixType;
using FilamentMatrix = filament::math::mat4f;
EigenMatrix EigenMatrixFromFilamentMatrix(const filament::math::mat4f& fm) {
    EigenMatrix em;

    em << fm(0, 0), fm(0, 1), fm(0, 2), fm(0, 3), fm(1, 0), fm(1, 1), fm(1, 2),
            fm(1, 3), fm(2, 0), fm(2, 1), fm(2, 2), fm(2, 3), fm(3, 0),
            fm(3, 1), fm(3, 2), fm(3, 3);

    return em;
}

FilamentMatrix FilamentMatrixFromEigenMatrix(const EigenMatrix& em) {
    // Filament matrices is column major and Eigen's - row major
    return FilamentMatrix(FilamentMatrix::row_major_init{
            em(0, 0), em(0, 1), em(0, 2), em(0, 3), em(1, 0), em(1, 1),
            em(1, 2), em(1, 3), em(2, 0), em(2, 1), em(2, 2), em(2, 3),
            em(3, 0), em(3, 1), em(3, 2), em(3, 3)});
}
}  // namespace converters

/// @endcond
}  // namespace

namespace open3d {
namespace visualization {
namespace rendering {

FilamentScene::FilamentScene(filament::Engine& engine,
                             FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    scene_ = engine_.createScene();
}

FilamentScene::~FilamentScene() {
    for (const auto& pair : entities_) {
        const auto& allocated_entity = pair.second;

        if (allocated_entity.info.ib) {
            resource_mgr_.Destroy(allocated_entity.info.ib);
        }
        if (allocated_entity.info.vb) {
            resource_mgr_.Destroy(allocated_entity.info.vb);
        }

        engine_.destroy(allocated_entity.info.self);
    }

    views_.clear();

    engine_.destroy(scene_);
}

ViewHandle FilamentScene::AddView(std::int32_t x,
                                  std::int32_t y,
                                  std::uint32_t w,
                                  std::uint32_t h) {
    auto handle = ViewHandle::Next();
    auto view = std::make_unique<FilamentView>(engine_, *this, resource_mgr_);

    view->SetViewport(x, y, w, h);
    if (!views_.empty()) {
        view->SetDiscardBuffers(View::TargetBuffers::DepthAndStencil);
    }

    ViewContainer c;
    c.view = std::move(view);
    views_.emplace(handle, std::move(c));

    return handle;
}

View* FilamentScene::GetView(const ViewHandle& view_id) const {
    auto found = views_.find(view_id);
    if (found != views_.end()) {
        return found->second.view.get();
    }

    return nullptr;
}

void FilamentScene::SetViewActive(const ViewHandle& view_id, bool is_active) {
    auto found = views_.find(view_id);
    if (found != views_.end()) {
        found->second.is_active = is_active;
    }
}

void FilamentScene::RemoveView(const ViewHandle& view_id) {
    views_.erase(view_id);
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry) {
    GeometryHandle handle;

    const auto geometry_type = geometry.GetGeometryType();

    MaterialInstanceHandle material_instance;
    handle = AddGeometry(geometry, material_instance);

    if (geometry_type == geometry::Geometry::GeometryType::TriangleMesh) {
        const auto& mesh = static_cast<const geometry::TriangleMesh&>(geometry);

        if (mesh.HasMaterials()) {  // Mesh with materials
            material_instance = resource_mgr_.CreateFromDescriptor(
                    mesh.materials_.begin()->second);
        } else if (mesh.HasVertexColors()) {  // Mesh with vertex color
                                              // attribute set
            material_instance = resource_mgr_.CreateMaterialInstance(
                    defaults_mapping::kColorOnlyMesh);

        } else if (mesh.HasTextures()) {  // Mesh with textures
            material_instance = resource_mgr_.CreateMaterialInstance(
                    defaults_mapping::kMesh);

            auto wmaterial =
                    resource_mgr_.GetMaterialInstance(material_instance);
            auto mat = wmaterial.lock();

            auto htex = resource_mgr_.CreateTexture(
                    mesh.textures_[0].FlipVertical());

            if (htex) {
                auto& entity = entities_[handle];
                entity.texture = htex;

                auto wtex = resource_mgr_.GetTexture(htex);
                auto tex = wtex.lock();
                if (tex) {
                    static const auto kDefaultSampler =
                            FilamentMaterialModifier::
                                    SamplerFromSamplerParameters(
                                            TextureSamplerParameters::Pretty());
                    mat->setParameter("texture", tex.get(), kDefaultSampler);
                }
            }
        } else {  // Mesh without any attributes set, only tangents are needed
            material_instance = resource_mgr_.CreateMaterialInstance(
                    defaults_mapping::kPlainMesh);

            auto wmaterial =
                    resource_mgr_.GetMaterialInstance(material_instance);
            auto mat = wmaterial.lock();

            if (mat) {
                mat->setParameter("baseColor", filament::RgbType::LINEAR,
                                  {0.75f, 0.75f, 0.75f});
            }
        }
    } else if (geometry_type == geometry::Geometry::GeometryType::PointCloud) {
        const auto& pcd = static_cast<const geometry::PointCloud&>(geometry);
        if (pcd.HasColors()) {
            material_instance = resource_mgr_.CreateMaterialInstance(
                    defaults_mapping::kColoredPointcloud);
        } else {
            material_instance = resource_mgr_.CreateMaterialInstance(
                    defaults_mapping::kPointcloud);
        }
    } else if (geometry_type == geometry::Geometry::GeometryType::LineSet) {
        material_instance = resource_mgr_.CreateMaterialInstance(
                defaults_mapping::kLineset);
    } else {
        utility::LogWarning(
                "Geometry type {} is not yet supported for easy-init!",
                static_cast<size_t>(geometry.GetGeometryType()));
    }

    AssignMaterial(handle, material_instance);

    return handle;
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry,
        const MaterialInstanceHandle& material_id) {
    return AddGeometry(geometry, material_id, geometry.GetName());
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry,
        const MaterialInstanceHandle& material_id,
        const std::string& name) {
    using namespace geometry;
    using namespace filament;

    SceneEntity entity_entry;
    entity_entry.info.type = EntityType::Geometry;
    entity_entry.name = name;

    auto geometry_buffer_builder = GeometryBuffersBuilder::GetBuilder(geometry);
    if (!geometry_buffer_builder) {
        utility::LogWarning("Geometry type {} is not supported yet!",
                            static_cast<size_t>(geometry.GetGeometryType()));
        return {};
    }

    auto buffers = geometry_buffer_builder->ConstructBuffers();
    entity_entry.info.vb = std::get<0>(buffers);
    entity_entry.info.ib = std::get<1>(buffers);

    Box aabb = geometry_buffer_builder->ComputeAABB();

    auto vbuf = resource_mgr_.GetVertexBuffer(entity_entry.info.vb).lock();
    auto ibuf = resource_mgr_.GetIndexBuffer(entity_entry.info.ib).lock();

    entity_entry.info.self = utils::EntityManager::get().create();
    RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .layerMask(FilamentView::kAllLayersMask, FilamentView::kMainLayer)
            .castShadows(true)
            .receiveShadows(true)
            .geometry(0, geometry_buffer_builder->GetPrimitiveType(),
                      vbuf.get(), ibuf.get());

    auto wmat_instance = resource_mgr_.GetMaterialInstance(material_id);
    if (!wmat_instance.expired()) {
        builder.material(0, wmat_instance.lock().get());
        entity_entry.material = material_id;
    }

    auto result = builder.build(engine_, entity_entry.info.self);

    GeometryHandle handle;
    if (result == RenderableManager::Builder::Success) {
        scene_->addEntity(entity_entry.info.self);

        handle = GeometryHandle::Next();
        entities_[handle] = entity_entry;

        SetEntityTransform(handle, Transform::Identity());
    }

    return handle;
}

std::vector<GeometryHandle> FilamentScene::FindGeometryByName(
        const std::string& name) {
    std::vector<GeometryHandle> found;
    for (const auto& e : entities_) {
        if (e.first.type == EntityType::Geometry && e.second.name == name) {
            found.push_back(GeometryHandle::Concretize(e.first));
        }
    }

    return found;
}

void FilamentScene::AssignMaterial(const GeometryHandle& geometry_id,
                                   const MaterialInstanceHandle& material_id) {
    auto wmat_instance = resource_mgr_.GetMaterialInstance(material_id);
    auto found = entities_.find(geometry_id);
    if (found != entities_.end() && !wmat_instance.expired()) {
        found->second.material = material_id;

        auto& renderable_mgr = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderable_mgr.getInstance(found->second.info.self);
        renderable_mgr.setMaterialInstanceAt(inst, 0,
                                             wmat_instance.lock().get());
    } else {
        utility::LogWarning(
                "Failed to assign material ({}) to geometry ({}): material or "
                "entity not found",
                material_id, geometry_id);
    }
}

MaterialInstanceHandle FilamentScene::GetMaterial(
        const GeometryHandle& geometry_id) const {
    const auto found = entities_.find(geometry_id);
    if (found != entities_.end()) {
        return found->second.material;
    }

    utility::LogWarning("Geometry {} is not registered in scene", geometry_id);
    return {};
}

void FilamentScene::SetGeometryShadows(const GeometryHandle& geometry_id,
                                       bool casts_shadows,
                                       bool receives_shadows) {
    const auto found = entities_.find(geometry_id);
    if (found != entities_.end()) {
        auto& renderable_mgr = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderable_mgr.getInstance(found->second.info.self);
        renderable_mgr.setCastShadows(inst, casts_shadows);
        renderable_mgr.setReceiveShadows(inst, casts_shadows);
    }
}

void FilamentScene::RemoveGeometry(const GeometryHandle& geometry_id) {
    RemoveEntity(geometry_id);
}

LightHandle FilamentScene::AddLight(const LightDescription& descr) {
    filament::LightManager::Type light_type =
            filament::LightManager::Type::POINT;
    if (descr.custom_attributes["custom_type"].isString()) {
        auto custom_type = descr.custom_attributes["custom_type"];
        if (custom_type == "SUN") {
            light_type = filament::LightManager::Type::SUN;
        }
    } else {
        switch (descr.type) {
            case LightDescription::POINT:
                light_type = filament::LightManager::Type::POINT;
                break;
            case LightDescription::SPOT:
                light_type = filament::LightManager::Type::SPOT;
                break;
            case LightDescription::DIRECTIONAL:
                light_type = filament::LightManager::Type::DIRECTIONAL;
                break;
        }
    }

    auto light = utils::EntityManager::get().create();
    auto result =
            filament::LightManager::Builder(light_type)
                    .direction({descr.direction.x(), descr.direction.y(),
                                descr.direction.z()})
                    .position({descr.position.x(), descr.position.y(),
                               descr.position.z()})
                    .intensity(descr.intensity)
                    .falloff(descr.falloff)
                    .castShadows(descr.cast_shadows)
                    .color({descr.color.x(), descr.color.y(), descr.color.z()})
                    .spotLightCone(descr.light_cone_inner,
                                   descr.light_cone_outer)
                    .build(engine_, light);

    LightHandle handle;
    if (result == filament::LightManager::Builder::Success) {
        handle = LightHandle::Next();

        SceneEntity entity;
        entity.info.self = light;
        entity.info.type = EntityType::Light;
        entities_[handle] = entity;

        scene_->addEntity(light);
    }

    return handle;
}

void FilamentScene::SetLightIntensity(const LightHandle& id,
                                      const float intensity) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        light_mgr.setIntensity(inst, intensity);
    }
}

void FilamentScene::SetLightColor(const LightHandle& id,
                                  const Eigen::Vector3f& color) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        light_mgr.setColor(inst, {color(0), color(1), color(2)});
    }
}

Eigen::Vector3f FilamentScene::GetLightDirection(const LightHandle& id) const {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        auto dir = light_mgr.getDirection(inst);
        return {dir[0], dir[1], dir[2]};
    }
    return {0.0f, 0.0f, 0.0f};
}

void FilamentScene::SetLightDirection(const LightHandle& id,
                                      const Eigen::Vector3f& dir) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        light_mgr.setDirection(inst, {dir.x(), dir.y(), dir.z()});
    }
}

void FilamentScene::SetLightPosition(const LightHandle& id,
                                     const Eigen::Vector3f& pos) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        if (!light_mgr.isDirectional(inst)) {
            light_mgr.setPosition(inst, {pos.x(), pos.y(), pos.z()});
        }
    }
}

void FilamentScene::SetLightFalloff(const LightHandle& id,
                                    const float falloff) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(found->second.info.self);
        light_mgr.setFalloff(inst, falloff);
    }
}

void FilamentScene::RemoveLight(const LightHandle& id) { RemoveEntity(id); }

void FilamentScene::SetIndirectLight(const IndirectLightHandle& id) {
    if (!id) {
        indirect_light_.reset();
        scene_->setIndirectLight(nullptr);
        return;
    }

    auto wlight = resource_mgr_.GetIndirectLight(id);
    if (auto light = wlight.lock()) {
        indirect_light_ = wlight;
        scene_->setIndirectLight(light.get());
    }
}

void FilamentScene::SetIndirectLightIntensity(float intensity) {
    if (auto light = indirect_light_.lock()) {
        light->setIntensity(intensity);
    }
}

float FilamentScene::GetIndirectLightIntensity() const {
    if (auto light = indirect_light_.lock()) {
        return light->getIntensity();
    }

    return 0.f;
}

void FilamentScene::SetIndirectLightRotation(const Transform& rotation) {
    if (auto light = indirect_light_.lock()) {
        auto ft = converters::FilamentMatrixFromEigenMatrix(rotation.matrix());
        light->setRotation(ft.upperLeft());
    }
}

FilamentScene::Transform FilamentScene::GetIndirectLightRotation() const {
    if (auto light = indirect_light_.lock()) {
        converters::FilamentMatrix ft(light->getRotation());
        auto et = converters::EigenMatrixFromFilamentMatrix(ft);

        return Transform(et);
    }

    return {};
}

void FilamentScene::SetSkybox(const SkyboxHandle& id) {
    if (!id) {
        skybox_.reset();
        scene_->setSkybox(nullptr);
        return;
    }

    auto wskybox = resource_mgr_.GetSkybox(id);
    if (auto skybox = wskybox.lock()) {
        skybox_ = wskybox;
        scene_->setSkybox(skybox.get());
    }
}

void FilamentScene::SetEntityEnabled(const REHandle_abstract& entity_id,
                                     const bool enabled) {
    auto found = entities_.find(entity_id);
    if (found != entities_.end()) {
        auto& entity = found->second;
        if (entity.enabled != enabled) {
            entity.enabled = enabled;

            if (enabled) {
                scene_->addEntity(entity.info.self);
            } else {
                scene_->remove(entity.info.self);
            }
        }
    }
}

bool FilamentScene::GetEntityEnabled(const REHandle_abstract& entity_id) {
    auto found = entities_.find(entity_id);
    if (found != entities_.end()) {
        return found->second.enabled;
    } else {
        return false;
    }
}

void FilamentScene::SetEntityTransform(const REHandle_abstract& entity_id,
                                       const Transform& transform) {
    auto itransform = GetEntityTransformInstance(entity_id);
    if (itransform.isValid()) {
        const auto& ematrix = transform.matrix();
        auto& transform_mgr = engine_.getTransformManager();
        transform_mgr.setTransform(
                itransform, converters::FilamentMatrixFromEigenMatrix(ematrix));
    }
}

FilamentScene::Transform FilamentScene::GetEntityTransform(
        const REHandle_abstract& entity_id) {
    auto itransform = GetEntityTransformInstance(entity_id);

    Transform etransform;
    if (itransform.isValid()) {
        auto& transform_mgr = engine_.getTransformManager();
        auto ftransform = transform_mgr.getTransform(itransform);
        etransform = converters::EigenMatrixFromFilamentMatrix(ftransform);
    }

    return etransform;
}

geometry::AxisAlignedBoundingBox FilamentScene::GetEntityBoundingBox(
        const REHandle_abstract& entity_id) {
    geometry::AxisAlignedBoundingBox result;

    auto found = entities_.find(entity_id);
    if (found != entities_.end()) {
        auto& renderable_mgr = engine_.getRenderableManager();
        auto inst = renderable_mgr.getInstance(found->second.info.self);
        auto box = renderable_mgr.getAxisAlignedBoundingBox(inst);

        auto& transform_mgr = engine_.getTransformManager();
        auto itransform = transform_mgr.getInstance(found->second.info.self);
        auto transform = transform_mgr.getWorldTransform(itransform);

        box = rigidTransform(box, transform);

        auto min = box.center - box.halfExtent;
        auto max = box.center + box.halfExtent;
        result = {{min.x, min.y, min.z}, {max.x, max.y, max.z}};
    }

    return result;
}

void FilamentScene::Draw(filament::Renderer& renderer) {
    for (const auto& pair : views_) {
        auto& container = pair.second;
        if (container.is_active) {
            container.view->PreRender();
            renderer.render(container.view->GetNativeView());
            container.view->PostRender();
        }
    }
}

utils::EntityInstance<filament::TransformManager>
FilamentScene::GetEntityTransformInstance(const REHandle_abstract& id) {
    auto found = entities_.find(id);

    filament::TransformManager::Instance itransform;
    if (found != entities_.end()) {
        auto& transform_mgr = engine_.getTransformManager();
        itransform = transform_mgr.getInstance(found->second.parent);
        if (!itransform.isValid()) {
            using namespace filament::math;

            auto parent = utils::EntityManager::get().create();
            found->second.parent = parent;

            transform_mgr.create(found->second.parent);
            transform_mgr.create(found->second.info.self);

            itransform = transform_mgr.getInstance(found->second.info.self);
            itransform = transform_mgr.getInstance(found->second.parent);

            transform_mgr.create(found->second.info.self, itransform,
                                 mat4f::translation(float3{0.0f, 0.0f, 0.0f}));
        }
    }

    return itransform;
}

void FilamentScene::RemoveEntity(REHandle_abstract id) {
    auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& data = found->second;
        scene_->remove(data.info.self);

        data.ReleaseResources(engine_, resource_mgr_);

        entities_.erase(found);
    }
}

void FilamentScene::SceneEntity::Details::ReleaseResources(
        filament::Engine& engine, FilamentResourceManager& manager) {
    if (vb) {
        manager.Destroy(vb);
    }
    if (ib) {
        manager.Destroy(ib);
    }

    engine.destroy(self);
    self.clear();
}

void FilamentScene::SceneEntity::ReleaseResources(
        filament::Engine& engine, FilamentResourceManager& manager) {
    info.ReleaseResources(engine, manager);

    if (texture) {
        manager.Destroy(texture);
    }

    engine.destroy(parent);
    parent.clear();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
