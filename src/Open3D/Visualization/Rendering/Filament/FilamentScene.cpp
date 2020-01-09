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

#include "FilamentGeometryBuffersBuilder.h"
#include "FilamentResourceManager.h"
#include "FilamentView.h"
#include "Open3D/Geometry/Geometry3D.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/View.h>

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

        if (allocatedEntity.ib) {
            resourceManager_.Destroy(allocatedEntity.ib);
        }
        if (allocatedEntity.vb) {
            resourceManager_.Destroy(allocatedEntity.vb);
        }

        engine_.destroy(allocatedEntity.self);
    }

    views_.clear();

    engine_.destroy(scene_);
}

ViewHandle FilamentScene::AddView(std::int32_t x,
                                  std::int32_t y,
                                  std::uint32_t w,
                                  std::uint32_t h) {
    auto handle = ViewHandle::Next();
    auto view = std::make_unique<FilamentView>(engine_, *scene_);

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
        const geometry::Geometry3D& geometry,
        const MaterialInstanceHandle& materialId) {
    using namespace geometry;
    using namespace filament;

    AllocatedEntity entityEntry;

    auto geometryBuffersBuilder = GeometryBuffersBuilder::GetBuilder(geometry);
    if (!geometryBuffersBuilder) {
        // FIXME: Log unsupported geometry
        return {};
    }

    entityEntry.vb = geometryBuffersBuilder->ConstructVertexBuffer();
    entityEntry.ib = geometryBuffersBuilder->ConstructIndexBuffer();

    Box aabb = geometryBuffersBuilder->ComputeAABB();

    auto vbuf = resourceManager_.GetVertexBuffer(entityEntry.vb).lock();
    auto ibuf = resourceManager_.GetIndexBuffer(entityEntry.ib).lock();

    entityEntry.self = utils::EntityManager::get().create();
    RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .geometry(0, geometryBuffersBuilder->GetPrimitiveType(), vbuf.get(),
                      ibuf.get())
            .culling(false);

    auto wMatInstance = resourceManager_.GetMaterialInstance(materialId);
    if (!wMatInstance.expired()) {
        builder.material(0, wMatInstance.lock().get());
    }

    auto result = builder.build(engine_, entityEntry.self);

    GeometryHandle handle;
    if (result == RenderableManager::Builder::Success) {
        handle = GeometryHandle::Next();
        entities_[handle] = entityEntry;
        scene_->addEntity(entityEntry.self);
    }

    return handle;
}

void FilamentScene::AssignMaterial(const GeometryHandle& geometryId,
                                   const MaterialInstanceHandle& materialId) {
    auto wMaterialInstance = resourceManager_.GetMaterialInstance(materialId);
    auto found = entities_.find(geometryId);
    if (found != entities_.end() && false == wMaterialInstance.expired()) {
        auto& renderableManger = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderableManger.getInstance(found->second.self);
        renderableManger.setMaterialInstanceAt(inst, 0,
                                               wMaterialInstance.lock().get());
    }
    // TODO: Log if failed
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
        entities_[handle] = {light};
        scene_->addEntity(light);
    }

    return handle;
}

void FilamentScene::RemoveLight(const LightHandle& id) { RemoveEntity(id); }

void FilamentScene::SetEntityTransform(const REHandle_abstract& entityId, const Transform& transform) {
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

FilamentScene::Transform FilamentScene::GetEntityTransform(const REHandle_abstract& entityId) {
    auto iTransform = GetEntityTransformInstance(entityId);

    Transform eTransform;
    if (iTransform.isValid()) {
        auto& transformMgr = engine_.getTransformManager();
        auto fTransform = transformMgr.getTransform(iTransform);

        Transform::MatrixType matrix;

        matrix << fTransform(0,0), fTransform(0,1), fTransform(0,2), fTransform(0,3),
                fTransform(1,0), fTransform(1,1), fTransform(1,2), fTransform(1,3),
                fTransform(2,0), fTransform(2,1), fTransform(2,2), fTransform(2,3),
                fTransform(3,0), fTransform(3,1), fTransform(3,2), fTransform(3,3);

        eTransform = matrix;
    }

    return eTransform;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> FilamentScene::GetEntityBoundingBox(const REHandle_abstract& entityId)
{
    std::pair<Eigen::Vector3f, Eigen::Vector3f> result;

    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        auto& renderableManager = engine_.getRenderableManager();
        auto inst = renderableManager.getInstance(found->second.self);
        auto box = renderableManager.getAxisAlignedBoundingBox(inst);

        result.first = {box.center.x, box.center.y, box.center.z};
        result.second = {box.halfExtent.x, box.halfExtent.y, box.halfExtent.z};
    }

    return result;
}

std::pair<Eigen::Vector3f, float> FilamentScene::GetEntityBoundingSphere(const REHandle_abstract& entityId) {
    std::pair<Eigen::Vector3f, float> result;

    auto found = entities_.find(entityId);
    if (found != entities_.end()) {
        auto& renderableManager = engine_.getRenderableManager();
        auto inst = renderableManager.getInstance(found->second.self);
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
            renderer.render(container.view->GetNativeView());
        }
    }
}

utils::EntityInstance<filament::TransformManager> FilamentScene::GetEntityTransformInstance(const REHandle_abstract& id) {
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
            transformMgr.create(found->second.self);

            iTransform = transformMgr.getInstance(found->second.self);
            iTransform = transformMgr.getInstance(found->second.parent);

            auto center = GetEntityBoundingSphere(id).first;
            transformMgr.create(found->second.self, iTransform, mat4f::translation(float3 {-center.x(),-center.y(),-center.z()}));
        }
    }

    return iTransform;
}

void FilamentScene::RemoveEntity(REHandle_abstract id) {
    auto found = entities_.find(id);
    if (found != entities_.end()) {
        const auto& data = found->second;
        scene_->remove(data.self);

        if (data.vb) {
            resourceManager_.Destroy(data.vb);
        }
        if (data.ib) {
            resourceManager_.Destroy(data.ib);
        }
        engine_.destroy(data.self);
        engine_.destroy(data.parent);

        entities_.erase(found);
    }
}

}  // namespace visualization
}  // namespace open3d
