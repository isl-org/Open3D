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
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/View.h>

#include "FilamentEntitiesMods.h"
#include "FilamentGeometryBuffersBuilder.h"
#include "FilamentResourceManager.h"
#include "FilamentView.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

namespace defaults_mapping {

using GeometryType = open3d::geometry::Geometry::GeometryType;
using MaterialHandle = open3d::visualization::MaterialHandle;
using ResourceManager = open3d::visualization::FilamentResourceManager;

const std::unordered_map<GeometryType, MaterialHandle> kDefaultMaterials = {
        {GeometryType::TriangleMesh, ResourceManager::kDefaultLit},
        {GeometryType::LineSet, ResourceManager::kDefaultUnlit},
        {GeometryType::PointCloud, ResourceManager::kDefaultUnlit}};
}  // namespace defaults_mapping

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
    auto defaults = defaults_mapping::kDefaultMaterials.find(geometryType);
    if (defaults != defaults_mapping::kDefaultMaterials.end()) {
        MaterialInstanceHandle materialInstance =
                resourceManager_.CreateMaterialInstance(defaults->second);
        handle = AddGeometry(geometry, materialInstance);

        if (geometryType == geometry::Geometry::GeometryType::TriangleMesh) {
            const auto& mesh =
                    static_cast<const geometry::TriangleMesh&>(geometry);
            if (mesh.texture_.HasData()) {
                auto hTexture = resourceManager_.CreateTexture(
                        mesh.texture_.FlipVertical());

                if (hTexture) {
                    auto& entity = entities_[handle];
                    entity.texture = hTexture;

                    auto wMaterial = resourceManager_.GetMaterialInstance(
                            entity.material);
                    auto mat = wMaterial.lock();

                    auto wTexture = resourceManager_.GetTexture(hTexture);
                    auto tex = wTexture.lock();
                    if (mat && tex) {
                        static const auto kDefaultSampler =
                                FilamentMaterialModifier::
                                        SamplerFromSamplerParameters(
                                                TextureSamplerParameters::
                                                        Pretty());
                        mat->setParameter("texture", tex.get(),
                                          kDefaultSampler);
                    }
                }
            }
        }
    }

    utility::LogWarning("Geometry type {} is not supported yet!",
                        static_cast<size_t>(geometry.GetGeometryType()));

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

void FilamentScene::RemoveLight(const LightHandle& id) { RemoveEntity(id); }

void FilamentScene::SetEntityTransform(const REHandle_abstract& entityId,
                                       const Transform& transform) {
    auto iTransform = GetEntityTransformInstance(entityId);
    if (iTransform.isValid()) {
        using namespace filament::math;

        // Filament matrices is column major and Eigen's - row major
        auto eMatrix = transform.matrix();
        mat4f fTransform(mat4f::row_major_init{
                eMatrix(0, 0), eMatrix(0, 1), eMatrix(0, 2), eMatrix(0, 3),
                eMatrix(1, 0), eMatrix(1, 1), eMatrix(1, 2), eMatrix(1, 3),
                eMatrix(2, 0), eMatrix(2, 1), eMatrix(2, 2), eMatrix(2, 3),
                eMatrix(3, 0), eMatrix(3, 1), eMatrix(3, 2), eMatrix(3, 3)});

        auto& transformMgr = engine_.getTransformManager();
        transformMgr.setTransform(iTransform, fTransform);
    }
}

FilamentScene::Transform FilamentScene::GetEntityTransform(
        const REHandle_abstract& entityId) {
    auto iTransform = GetEntityTransformInstance(entityId);

    Transform eTransform;
    if (iTransform.isValid()) {
        auto& transformMgr = engine_.getTransformManager();
        auto fTransform = transformMgr.getTransform(iTransform);

        Transform::MatrixType matrix;

        matrix << fTransform(0, 0), fTransform(0, 1), fTransform(0, 2),
                fTransform(0, 3), fTransform(1, 0), fTransform(1, 1),
                fTransform(1, 2), fTransform(1, 3), fTransform(2, 0),
                fTransform(2, 1), fTransform(2, 2), fTransform(2, 3),
                fTransform(3, 0), fTransform(3, 1), fTransform(3, 2),
                fTransform(3, 3);

        eTransform = matrix;
    }

    return eTransform;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> FilamentScene::GetEntityBoundingBox(
        const REHandle_abstract& entityId) {
    std::pair<Eigen::Vector3f, Eigen::Vector3f> result;

    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        auto& renderableManager = engine_.getRenderableManager();
        auto inst = renderableManager.getInstance(found->second.info.self);
        auto box = renderableManager.getAxisAlignedBoundingBox(inst);

        result.first = {box.center.x, box.center.y, box.center.z};
        result.second = {box.halfExtent.x, box.halfExtent.y, box.halfExtent.z};
    }

    return result;
}

std::pair<Eigen::Vector3f, float> FilamentScene::GetEntityBoundingSphere(
        const REHandle_abstract& entityId) {
    std::pair<Eigen::Vector3f, float> result;

    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        auto& renderableManager = engine_.getRenderableManager();
        auto inst = renderableManager.getInstance(found->second.info.self);
        auto sphere = renderableManager.getAxisAlignedBoundingBox(inst)
                              .getBoundingSphere();

        result.first = {sphere.x, sphere.y, sphere.z};
        result.second = sphere.w;
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
            // Forcing user to manipulate transform located in center of entity
            auto parent = utils::EntityManager::get().create();
            found->second.parent = parent;

            transformMgr.create(found->second.parent);
            transformMgr.create(found->second.info.self);

            iTransform = transformMgr.getInstance(found->second.info.self);
            iTransform = transformMgr.getInstance(found->second.parent);

            auto center = GetEntityBoundingSphere(id).first;
            transformMgr.create(
                    found->second.info.self, iTransform,
                    mat4f::translation(
                            float3{-center.x(), -center.y(), -center.z()}));
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
    manager.Destroy(texture);

    engine.destroy(parent);
    parent.clear();
}

}  // namespace visualization
}  // namespace open3d
