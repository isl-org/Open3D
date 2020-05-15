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

#include "FilamentScene.h"

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
#include <filament/utils/EntityManager.h>

#include "FilamentEntitiesMods.h"
#include "FilamentGeometryBuffersBuilder.h"
#include "FilamentResourceManager.h"
#include "FilamentView.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

namespace {  // avoid polluting global namespace, since only used here

namespace defaults_mapping {

using GeometryType = open3d::geometry::Geometry::GeometryType;
using MaterialHandle = open3d::visualization::MaterialHandle;
using ResourceManager = open3d::visualization::FilamentResourceManager;

MaterialHandle ColorOnlyMesh = ResourceManager::kDefaultUnlit;
MaterialHandle PlainMesh = ResourceManager::kDefaultLit;
MaterialHandle Mesh = ResourceManager::kDefaultLit;

MaterialHandle ColoredPointcloud = ResourceManager::kDefaultUnlit;
MaterialHandle Pointcloud = ResourceManager::kDefaultLit;

MaterialHandle Lineset = ResourceManager::kDefaultUnlit;

}  // namespace defaults_mapping

namespace converters {
using EigenMatrix = open3d::visualization::FilamentScene::Transform::MatrixType;
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

}  // namespace

namespace open3d {
namespace visualization {

FilamentScene::FilamentScene(filament::Engine& aEngine,
                             FilamentResourceManager& aResourceManager)
    : engine_(aEngine), resourceManager_(aResourceManager) {
    scene_ = engine_.createScene();
}

FilamentScene::~FilamentScene() {
    for (const auto& pair : entities_) {
        const auto& allocatedEntity = pair.second;

        if (allocatedEntity.info.ib) {
            resourceManager_.Destroy(allocatedEntity.info.ib);
        }
        if (allocatedEntity.info.vb) {
            resourceManager_.Destroy(allocatedEntity.info.vb);
        }

        engine_.destroy(allocatedEntity.info.self);
    }

    views_.clear();

    engine_.destroy(scene_);
}

ViewHandle FilamentScene::AddView(std::int32_t x,
                                  std::int32_t y,
                                  std::uint32_t w,
                                  std::uint32_t h) {
    auto handle = ViewHandle::Next();
    auto view =
            std::make_unique<FilamentView>(engine_, *this, resourceManager_);

    view->SetViewport(x, y, w, h);
    if (!views_.empty()) {
        view->SetDiscardBuffers(View::TargetBuffers::DepthAndStencil);
    }

    ViewContainer c;
    c.view = std::move(view);
    views_.emplace(handle, std::move(c));

    return handle;
}

View* FilamentScene::GetView(const ViewHandle& viewId) const {
    auto found = views_.find(viewId);
    if (found != views_.end()) {
        return found->second.view.get();
    }

    return nullptr;
}

void FilamentScene::SetViewActive(const ViewHandle& viewId, bool isActive) {
    auto found = views_.find(viewId);
    if (found != views_.end()) {
        found->second.isActive = isActive;
    }
}

void FilamentScene::RemoveView(const ViewHandle& viewId) {
    views_.erase(viewId);
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry) {
    GeometryHandle handle;

    const auto geometryType = geometry.GetGeometryType();

    MaterialInstanceHandle materialInstance;
    handle = AddGeometry(geometry, materialInstance);

    if (geometryType == geometry::Geometry::GeometryType::TriangleMesh) {
        const auto& mesh = static_cast<const geometry::TriangleMesh&>(geometry);

        if (mesh.HasMaterials()) {  // Mesh with materials
            materialInstance = resourceManager_.CreateFromDescriptor(
                    mesh.materials_.begin()->second);
        } else if (mesh.HasVertexColors()) {  // Mesh with vertex color
                                              // attribute set
            materialInstance = resourceManager_.CreateMaterialInstance(
                    defaults_mapping::ColorOnlyMesh);

        } else if (mesh.HasTextures()) {  // Mesh with textures
            materialInstance = resourceManager_.CreateMaterialInstance(
                    defaults_mapping::Mesh);

            auto wMaterial =
                    resourceManager_.GetMaterialInstance(materialInstance);
            auto mat = wMaterial.lock();

            auto hTexture = resourceManager_.CreateTexture(
                    mesh.textures_[0].FlipVertical());

            if (hTexture) {
                auto& entity = entities_[handle];
                entity.texture = hTexture;

                auto wTexture = resourceManager_.GetTexture(hTexture);
                auto tex = wTexture.lock();
                if (tex) {
                    static const auto kDefaultSampler =
                            FilamentMaterialModifier::
                                    SamplerFromSamplerParameters(
                                            TextureSamplerParameters::Pretty());
                    mat->setParameter("texture", tex.get(), kDefaultSampler);
                }
            }
        } else {  // Mesh without any attributes set, only tangents are needed
            materialInstance = resourceManager_.CreateMaterialInstance(
                    defaults_mapping::PlainMesh);

            auto wMaterial =
                    resourceManager_.GetMaterialInstance(materialInstance);
            auto mat = wMaterial.lock();

            if (mat) {
                mat->setParameter("baseColor", filament::RgbType::LINEAR,
                                  {0.75f, 0.75f, 0.75f});
            }
        }
    } else if (geometryType == geometry::Geometry::GeometryType::PointCloud) {
        const auto& pcd = static_cast<const geometry::PointCloud&>(geometry);
        if (pcd.HasColors()) {
            materialInstance = resourceManager_.CreateMaterialInstance(
                    defaults_mapping::ColoredPointcloud);
        } else {
            materialInstance = resourceManager_.CreateMaterialInstance(
                    defaults_mapping::Pointcloud);
        }
    } else if (geometryType == geometry::Geometry::GeometryType::LineSet) {
        materialInstance = resourceManager_.CreateMaterialInstance(
                defaults_mapping::Lineset);
    } else {
        utility::LogWarning(
                "Geometry type {} is not yet supported for easy-init!",
                static_cast<size_t>(geometry.GetGeometryType()));
    }

    AssignMaterial(handle, materialInstance);

    return handle;
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry,
        const MaterialInstanceHandle& materialId) {
    return AddGeometry(geometry, materialId, geometry.GetName());
}

GeometryHandle FilamentScene::AddGeometry(
        const geometry::Geometry3D& geometry,
        const MaterialInstanceHandle& materialId,
        const std::string& name) {
    using namespace geometry;
    using namespace filament;

    SceneEntity entityEntry;
    entityEntry.info.type = EntityType::Geometry;
    entityEntry.name = name;

    auto geometryBuffersBuilder = GeometryBuffersBuilder::GetBuilder(geometry);
    if (!geometryBuffersBuilder) {
        utility::LogWarning("Geometry type {} is not supported yet!",
                            static_cast<size_t>(geometry.GetGeometryType()));
        return {};
    }

    auto buffers = geometryBuffersBuilder->ConstructBuffers();
    entityEntry.info.vb = std::get<0>(buffers);
    entityEntry.info.ib = std::get<1>(buffers);

    Box aabb = geometryBuffersBuilder->ComputeAABB();

    auto vbuf = resourceManager_.GetVertexBuffer(entityEntry.info.vb).lock();
    auto ibuf = resourceManager_.GetIndexBuffer(entityEntry.info.ib).lock();

    entityEntry.info.self = utils::EntityManager::get().create();
    RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .layerMask(FilamentView::kAllLayersMask, FilamentView::kMainLayer)
            .castShadows(true)
            .receiveShadows(true)
            .geometry(0, geometryBuffersBuilder->GetPrimitiveType(), vbuf.get(),
                      ibuf.get());

    auto wMatInstance = resourceManager_.GetMaterialInstance(materialId);
    if (!wMatInstance.expired()) {
        builder.material(0, wMatInstance.lock().get());
        entityEntry.material = materialId;
    }

    auto result = builder.build(engine_, entityEntry.info.self);

    GeometryHandle handle;
    if (result == RenderableManager::Builder::Success) {
        scene_->addEntity(entityEntry.info.self);

        handle = GeometryHandle::Next();
        entities_[handle] = entityEntry;

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

void FilamentScene::AssignMaterial(const GeometryHandle& geometryId,
                                   const MaterialInstanceHandle& materialId) {
    auto wMaterialInstance = resourceManager_.GetMaterialInstance(materialId);
    auto found = entities_.find(geometryId);
    if (found != entities_.end() && false == wMaterialInstance.expired()) {
        found->second.material = materialId;

        auto& renderableManger = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderableManger.getInstance(found->second.info.self);
        renderableManger.setMaterialInstanceAt(inst, 0,
                                               wMaterialInstance.lock().get());
    } else {
        utility::LogWarning(
                "Failed to assign material ({}) to geometry ({}): material or "
                "entity not found",
                materialId, geometryId);
    }
}

MaterialInstanceHandle FilamentScene::GetMaterial(
        const GeometryHandle& geometryId) const {
    const auto found = entities_.find(geometryId);
    if (found != entities_.end()) {
        return found->second.material;
    }

    utility::LogWarning("Geometry {} is not registered in scene", geometryId);
    return {};
}

void FilamentScene::SetGeometryShadows(const GeometryHandle& geometryId,
                                       bool castsShadows,
                                       bool receivesShadows) {
    const auto found = entities_.find(geometryId);
    if (found != entities_.end()) {
        auto& renderableManger = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderableManger.getInstance(found->second.info.self);
        renderableManger.setCastShadows(inst, castsShadows);
        renderableManger.setReceiveShadows(inst, castsShadows);
    }
}

void FilamentScene::RemoveGeometry(const GeometryHandle& geometryId) {
    RemoveEntity(geometryId);
}

LightHandle FilamentScene::AddLight(const LightDescription& descr) {
    filament::LightManager::Type lightType =
            filament::LightManager::Type::POINT;
    if (descr.customAttributes["custom_type"].isString()) {
        auto customType = descr.customAttributes["custom_type"];
        if (customType == "SUN") {
            lightType = filament::LightManager::Type::SUN;
        }
    } else {
        switch (descr.type) {
            case LightDescription::POINT:
                lightType = filament::LightManager::Type::POINT;
                break;
            case LightDescription::SPOT:
                lightType = filament::LightManager::Type::SPOT;
                break;
            case LightDescription::DIRECTIONAL:
                lightType = filament::LightManager::Type::DIRECTIONAL;
                break;
        }
    }

    auto light = utils::EntityManager::get().create();
    auto result =
            filament::LightManager::Builder(lightType)
                    .direction({descr.direction.x(), descr.direction.y(),
                                descr.direction.z()})
                    .position({descr.position.x(), descr.position.y(),
                               descr.position.z()})
                    .intensity(descr.intensity)
                    .falloff(descr.falloff)
                    .castShadows(descr.castShadows)
                    .color({descr.color.x(), descr.color.y(), descr.color.z()})
                    .spotLightCone(descr.lightConeInner, descr.lightConeOuter)
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
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        lightManager.setIntensity(inst, intensity);
    }
}

void FilamentScene::SetLightColor(const LightHandle& id,
                                  const Eigen::Vector3f& color) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        lightManager.setColor(inst, {color(0), color(1), color(2)});
    }
}

Eigen::Vector3f FilamentScene::GetLightDirection(const LightHandle& id) const {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        auto dir = lightManager.getDirection(inst);
        return {dir[0], dir[1], dir[2]};
    }
    return {0.0f, 0.0f, 0.0f};
}

void FilamentScene::SetLightDirection(const LightHandle& id,
                                      const Eigen::Vector3f& dir) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        lightManager.setDirection(inst, {dir.x(), dir.y(), dir.z()});
    }
}

void FilamentScene::SetLightPosition(const LightHandle& id,
                                     const Eigen::Vector3f& pos) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        if (!lightManager.isDirectional(inst)) {
            lightManager.setPosition(inst, {pos.x(), pos.y(), pos.z()});
        }
    }
}

void FilamentScene::SetLightFalloff(const LightHandle& id,
                                    const float falloff) {
    const auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& lightManager = engine_.getLightManager();
        filament::LightManager::Instance inst =
                lightManager.getInstance(found->second.info.self);
        lightManager.setFalloff(inst, falloff);
    }
}

void FilamentScene::RemoveLight(const LightHandle& id) { RemoveEntity(id); }

void FilamentScene::SetIndirectLight(const IndirectLightHandle& id) {
    if (!id) {
        wIndirectLight_.reset();
        scene_->setIndirectLight(nullptr);
        return;
    }

    auto wLight = resourceManager_.GetIndirectLight(id);
    if (auto light = wLight.lock()) {
        wIndirectLight_ = wLight;
        scene_->setIndirectLight(light.get());
    }
}

void FilamentScene::SetIndirectLightIntensity(float intensity) {
    if (auto light = wIndirectLight_.lock()) {
        light->setIntensity(intensity);
    }
}

float FilamentScene::GetIndirectLightIntensity() const {
    if (auto light = wIndirectLight_.lock()) {
        return light->getIntensity();
    }

    return 0.f;
}

void FilamentScene::SetIndirectLightRotation(const Transform& rotation) {
    if (auto light = wIndirectLight_.lock()) {
        auto ft = converters::FilamentMatrixFromEigenMatrix(rotation.matrix());
        light->setRotation(ft.upperLeft());
    }
}

FilamentScene::Transform FilamentScene::GetIndirectLightRotation() const {
    if (auto light = wIndirectLight_.lock()) {
        converters::FilamentMatrix ft(light->getRotation());
        auto et = converters::EigenMatrixFromFilamentMatrix(ft);

        return Transform(et);
    }

    return {};
}

void FilamentScene::SetSkybox(const SkyboxHandle& id) {
    if (!id) {
        wSkybox_.reset();
        scene_->setSkybox(nullptr);
        return;
    }

    auto wSkybox = resourceManager_.GetSkybox(id);
    if (auto skybox = wSkybox.lock()) {
        wSkybox_ = wSkybox;
        scene_->setSkybox(skybox.get());
    }
}

void FilamentScene::SetEntityEnabled(const REHandle_abstract& entityId,
                                     const bool enabled) {
    auto found = entities_.find(entityId);
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

bool FilamentScene::GetEntityEnabled(const REHandle_abstract& entityId) {
    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        return found->second.enabled;
    } else {
        return false;
    }
}

void FilamentScene::SetEntityTransform(const REHandle_abstract& entityId,
                                       const Transform& transform) {
    auto iTransform = GetEntityTransformInstance(entityId);
    if (iTransform.isValid()) {
        const auto& eMatrix = transform.matrix();
        auto& transformMgr = engine_.getTransformManager();
        transformMgr.setTransform(
                iTransform, converters::FilamentMatrixFromEigenMatrix(eMatrix));
    }
}

FilamentScene::Transform FilamentScene::GetEntityTransform(
        const REHandle_abstract& entityId) {
    auto iTransform = GetEntityTransformInstance(entityId);

    Transform eTransform;
    if (iTransform.isValid()) {
        auto& transformMgr = engine_.getTransformManager();
        auto fTransform = transformMgr.getTransform(iTransform);
        eTransform = converters::EigenMatrixFromFilamentMatrix(fTransform);
    }

    return eTransform;
}

geometry::AxisAlignedBoundingBox FilamentScene::GetEntityBoundingBox(
        const REHandle_abstract& entityId) {
    geometry::AxisAlignedBoundingBox result;

    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        auto& renderableManager = engine_.getRenderableManager();
        auto inst = renderableManager.getInstance(found->second.info.self);
        auto box = renderableManager.getAxisAlignedBoundingBox(inst);

        auto& transformMgr = engine_.getTransformManager();
        auto iTransform = transformMgr.getInstance(found->second.info.self);
        auto transform = transformMgr.getWorldTransform(iTransform);

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
        if (container.isActive) {
            container.view->PreRender();
            renderer.render(container.view->GetNativeView());
            container.view->PostRender();
        }
    }
}

utils::EntityInstance<filament::TransformManager>
FilamentScene::GetEntityTransformInstance(const REHandle_abstract& id) {
    auto found = entities_.find(id);

    filament::TransformManager::Instance iTransform;
    if (found != entities_.end()) {
        auto& transformMgr = engine_.getTransformManager();
        iTransform = transformMgr.getInstance(found->second.parent);
        if (!iTransform.isValid()) {
            using namespace filament::math;

            auto parent = utils::EntityManager::get().create();
            found->second.parent = parent;

            transformMgr.create(found->second.parent);
            transformMgr.create(found->second.info.self);

            iTransform = transformMgr.getInstance(found->second.info.self);
            iTransform = transformMgr.getInstance(found->second.parent);

            transformMgr.create(found->second.info.self, iTransform,
                                mat4f::translation(float3{0.0f, 0.0f, 0.0f}));
        }
    }

    return iTransform;
}

void FilamentScene::RemoveEntity(REHandle_abstract id) {
    auto found = entities_.find(id);
    if (found != entities_.end()) {
        auto& data = found->second;
        scene_->remove(data.info.self);

        data.ReleaseResources(engine_, resourceManager_);

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

}  // namespace visualization
}  // namespace open3d
