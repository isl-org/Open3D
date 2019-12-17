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

#include "FilamentResourceManager.h"
#include "FilamentView.h"
#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/TriangleMesh.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <filament/geometry/SurfaceOrientation.h>

namespace open3d {
namespace visualization {

static void freeBufferDescriptor(void* buffer, size_t, void*) { free(buffer); }

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

    if (geometry.GetGeometryType() != Geometry::GeometryType::TriangleMesh) {
        return GeometryHandle::kBad;
    }

    AllocatedEntity entityEntry;

    auto triangleMesh = static_cast<const TriangleMesh&>(geometry);
    const size_t nVertices = triangleMesh.vertices_.size();

    VertexBuffer* vbuf = AllocateVertexBuffer(entityEntry, nVertices);

    // Copying vertex coordinates
    const size_t coordsBytesCount = nVertices * 3 * sizeof(float);
    auto* float3VCoord = (Eigen::Vector3f*)malloc(coordsBytesCount);
    for (size_t i = 0; i < nVertices; ++i) {
        float3VCoord[i] = triangleMesh.vertices_[i].cast<float>();
    }

    // Moving copied vertex coordinates to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor coordsDescriptor(float3VCoord,
                                                    coordsBytesCount);
    coordsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine_, 0, std::move(coordsDescriptor));

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = triangleMesh.vertex_normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    const size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4VTangents = (math::quatf*)malloc(tangentsBytesCount);
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
                               .vertexCount(nVertices)
                               .normals((math::float3*)normals.data())
                               .build();
    orientation.getQuats(float4VTangents, nVertices);

    // Moving allocated tangents to VertexBuffer
    // they will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor tangentsDescriptor(float4VTangents,
                                                      tangentsBytesCount);
    tangentsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine_, 1, std::move(tangentsDescriptor));

    auto indexStride = sizeof(triangleMesh.triangles_[0][0]);
    auto ibHandle = resourceManager_.CreateIndexBuffer(
            triangleMesh.triangles_.size() * 3, indexStride);
    entityEntry.ib = ibHandle;

    auto ibuf = resourceManager_.GetIndexBuffer(ibHandle).lock();

    // Copying indices data
    const size_t indicesCount = triangleMesh.triangles_.size() * 3 * indexStride;
    auto* uint3Indices = (Eigen::Vector3i*)malloc(indicesCount);
    for (size_t i = 0; i < triangleMesh.triangles_.size(); ++i) {
        uint3Indices[i] = triangleMesh.triangles_[i];
    }

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(uint3Indices, indicesCount);
    indicesDescriptor.setCallback(freeBufferDescriptor);
    ibuf->setBuffer(engine_, std::move(indicesDescriptor));

    Box aabb;
    if (indexStride == sizeof(std::uint16_t)) {
        aabb = RenderableManager::computeAABB((math::float3*)float3VCoord,
                                              (std::uint16_t*)uint3Indices,
                                              nVertices);
    } else {
        aabb = RenderableManager::computeAABB((math::float3*)float3VCoord,
                                              (std::uint32_t*)uint3Indices,
                                              nVertices);
    }

    entityEntry.self = utils::EntityManager::get().create();
    RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vbuf,
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

//void FilamentScene::SetGeometryTransform(const GeometryHandle& geometryId, const Transform& transform) {
//    auto found = entities_.find(geometryId);
//    if (found != entities_.end()) {
//        auto& transformMgr = engine_.getTransformManager();
//        auto iTransform = transformMgr.getInstance(found->second.self);
//        if (!iTransform.isValid()) {
//            transformMgr.create(found->second.self);
//            iTransform = transformMgr.getInstance(found->second.self);
//        }
//
//        auto eMatrix = transform.matrix();
//        filament::math::mat4f fTransform(eMatrix(0,0), eMatrix(0,1), eMatrix(0,2), eMatrix(0,3),
//                                         eMatrix(1,0), eMatrix(1,1), eMatrix(1,2), eMatrix(1,3),
//                                         eMatrix(2,0), eMatrix(2,1), eMatrix(2,2), eMatrix(2,3),
//                                         eMatrix(3,0), eMatrix(3,1), eMatrix(3,2), eMatrix(3,3)
//                );
//        transformMgr.setTransform(iTransform, fTransform);
//    }
//}
//
//FilamentScene::Transform FilamentScene::GetGeometryTransform(const GeometryHandle& geometryId) const {
//    auto found = entities_.find(geometryId);
//    if (found != entities_.end()) {
//        auto& transformMgr = engine_.getTransformManager();
//        auto iTransform = transformMgr.getInstance(found->second.self);
//        if (!iTransform.isValid()) {
//            transformMgr.create(found->second.self);
//            iTransform = transformMgr.getInstance(found->second.self);
//        }
//    }
//    return FilamentScene::Transform();
//}

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
        auto eMatrix = transform.matrix();
        // FIXME: Need to find proper handling for different matrix storage approaches
        filament::math::mat4f fTransform(eMatrix(0,0), eMatrix(1,0), eMatrix(2,0), eMatrix(3,0),
                                         eMatrix(0,1), eMatrix(1,1), eMatrix(2,1), eMatrix(3,1),
                                         eMatrix(0,2), eMatrix(1,2), eMatrix(2,2), eMatrix(3,2),
                                         eMatrix(0,3), eMatrix(1,3), eMatrix(2,3), eMatrix(3,3)
                );

        auto& transformMgr = engine_.getTransformManager();
        transformMgr.setTransform(iTransform, fTransform);
    }
}

FilamentScene::Transform FilamentScene::GetEntityTransform(const REHandle_abstract& entityId) const {
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

void FilamentScene::Draw(filament::Renderer& renderer) {
    for (const auto& pair : views_) {
        auto& container = pair.second;
        if (container.isActive) {
            renderer.render(container.view->GetNativeView());
        }
    }
}

filament::VertexBuffer* FilamentScene::AllocateVertexBuffer(
        FilamentScene::AllocatedEntity& owner, const size_t verticesCount) {
    using namespace filament;

    VertexBuffer* vbuf =
            VertexBuffer::Builder()
                    .bufferCount(2)
                    .vertexCount(verticesCount)
                    .normalized(VertexAttribute::TANGENTS)
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT3, 0)
                    .attribute(VertexAttribute::TANGENTS, 1,
                               VertexBuffer::AttributeType::FLOAT4, 0)
                    .build(engine_);

    if (vbuf) {
        owner.vb = resourceManager_.AddVertexBuffer(vbuf);
    }

    return vbuf;
}

utils::EntityInstance<filament::TransformManager> FilamentScene::GetEntityTransformInstance(const REHandle_abstract& id) const {
    auto found = entities_.find(id);

    filament::TransformManager::Instance iTransform;
    if (found != entities_.end()) {
        auto& transformMgr = engine_.getTransformManager();
        iTransform = transformMgr.getInstance(found->second.self);
        if (!iTransform.isValid()) {
            transformMgr.create(found->second.self);
            iTransform = transformMgr.getInstance(found->second.self);
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

        entities_.erase(found);
    }
}

}
}
