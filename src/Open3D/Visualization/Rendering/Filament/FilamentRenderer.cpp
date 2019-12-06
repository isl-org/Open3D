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

#include "FilamentRenderer.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/geometry/SurfaceOrientation.h>

#include "FilamentCamera.h"
#include "FilamentEntitiesMods.h"
#include "FilamentResourceManager.h"
#include "FilamentView.h"
#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d
{
namespace visualization
{

AbstractRenderInterface* TheRenderer;

static void freeBufferDescriptor(void* buffer, size_t, void*) {
    free(buffer);
}

void FilamentRenderer::InitGlobal(void* nativeDrawable)
{
    TheRenderer = new FilamentRenderer(nativeDrawable);
}

void FilamentRenderer::ShutdownGlobal()
{
    delete TheRenderer;
    TheRenderer = nullptr;
}

FilamentRenderer::FilamentRenderer(void* nativeDrawable)
{
    engine = filament::Engine::create(filament::Engine::Backend::OPENGL);
    swapChain = engine->createSwapChain(nativeDrawable);
    renderer = engine->createRenderer();

    scene = engine->createScene();

    view = std::make_unique<FilamentView>(*engine, *scene);

    materialsModifier = std::make_unique<FilamentMaterialModifier>();
    resourcesManager = std::make_unique<FilamentResourceManager>(*engine);
}

FilamentRenderer::~FilamentRenderer()
{
    resourcesManager.reset();

#define BATCH_DESTROY(batch) for (const auto& pair : (batch)) {engine->destroy(pair.second.self);}
    BATCH_DESTROY(entities)
#undef BATCH_DESTROY

    view.reset();

    engine->destroy(scene);
    engine->destroy(renderer);
    engine->destroy(swapChain);

    filament::Engine::destroy(engine);
}

void FilamentRenderer::SetViewport(const std::int32_t x, const std::int32_t y, const std::uint32_t w, const std::uint32_t h)
{
    view->SetViewport(x, y, w, h);
}

void FilamentRenderer::SetClearColor(const Eigen::Vector3f& color)
{
    view->SetClearColor({ color.x(), color.y(), color.z() });
}

void FilamentRenderer::Draw()
{
    if (renderer->beginFrame(swapChain)) {
        renderer->render(view->GetNativeView());
        renderer->endFrame();
    }
}

Camera* FilamentRenderer::GetCamera() const
{
    return view->GetCamera();
}

GeometryHandle FilamentRenderer::AddGeometry(const geometry::Geometry3D& geometry, const MaterialInstanceHandle& materialId)
{
    using namespace geometry;
    using namespace filament;

    if (geometry.GetGeometryType() != Geometry::GeometryType::TriangleMesh)
    {
        return GeometryHandle::kBad;
    }

    AllocatedEntity entityEntry;

    auto triangleMesh = static_cast<const TriangleMesh&>(geometry);
    const size_t nVertices = triangleMesh.vertices_.size();

    VertexBuffer* vbuf = AllocateVertexBuffer(entityEntry, nVertices);

    // Copying vertex coordinates
    size_t coordsBytesCount = nVertices*3*sizeof(float);
    auto *float3VCoord = (Eigen::Vector3f*)malloc(coordsBytesCount);
    for (size_t i = 0; i < nVertices; ++i) {
        float3VCoord[i] = triangleMesh.vertices_[i].cast<float>();
    }

    // Moving copied vertex coordinates to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor coordsDescriptor(float3VCoord, coordsBytesCount);
    coordsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(*engine, 0, std::move(coordsDescriptor));

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = triangleMesh.vertex_normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    size_t tangentsBytesCount = nVertices*4*sizeof(float);
    auto *float4VTangents = (math::quatf*)malloc(tangentsBytesCount);
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
            .vertexCount(nVertices)
            .normals((math::float3*)normals.data())
            .build();
    orientation.getQuats(float4VTangents, nVertices);

    // Moving allocated tangents to VertexBuffer
    // they will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor tangentsDescriptor(float4VTangents, tangentsBytesCount);
    tangentsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(*engine, 1, std::move(tangentsDescriptor));

    auto indexStride = sizeof(triangleMesh.triangles_[0][0]);
    IndexBuffer* ibuf = AllocateIndexBuffer(entityEntry, triangleMesh.triangles_.size() * 3, indexStride);

    // Copying indices data
    size_t indicesCount = triangleMesh.triangles_.size()*3*indexStride;
    auto *uint3Indices = (Eigen::Vector3i*)malloc(indicesCount);
    for (size_t i = 0; i < triangleMesh.triangles_.size(); ++i) {
        uint3Indices[i] = triangleMesh.triangles_[i];
    }

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(uint3Indices, indicesCount);
    indicesDescriptor.setCallback(freeBufferDescriptor);
    ibuf->setBuffer(*engine, std::move(indicesDescriptor));

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
    builder
        .boundingBox(aabb)
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vbuf, ibuf)
        .culling(false);

    auto wMatInstance = resourcesManager->GetMaterialInstance(materialId);
    if (!wMatInstance.expired()) {
        builder.material(0, wMatInstance.lock().get());
    }

    auto result = builder.build(*engine, entityEntry.self);

    GeometryHandle handle;
    if (result == RenderableManager::Builder::Success) {
        handle = GeometryHandle::Next();
        entities[handle] = entityEntry;
        scene->addEntity(entityEntry.self);
    }

    return handle;
}

void FilamentRenderer::RemoveGeometry(const GeometryHandle& geometryId)
{
    RemoveEntity(geometryId);
}

LightHandle FilamentRenderer::AddLight(const LightDescription& descr)
{
    filament::LightManager::Type lightType = filament::LightManager::Type::POINT;
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
    auto result = filament::LightManager::Builder(lightType)
            .direction({ descr.direction.x(), descr.direction.y(), descr.direction.z() })
            .intensity(descr.intensity)
            .falloff(descr.falloff)
            .castShadows(descr.castShadows)
            .color({descr.color.x(), descr.color.y(), descr.color.z()})
            .spotLightCone(descr.lightConeInner, descr.lightConeOuter)
            .build(*engine, light);

    LightHandle handle;
    if (result == filament::LightManager::Builder::Success) {
        handle = LightHandle::Next();
        entities[handle] = {light};
        scene->addEntity(light);
    }

    return handle;
}

void FilamentRenderer::RemoveLight(const LightHandle& id)
{
    RemoveEntity(id);
}

MaterialHandle FilamentRenderer::AddMaterial(const void* materialData, const size_t dataSize)
{
    using namespace filament;

    Material* mat = Material::Builder()
            .package(materialData, dataSize)
            .build(*engine);

    MaterialHandle handle;
    if (mat) {
        handle = resourcesManager->AddMaterial(mat);
    }

    return handle;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id)
{
    materialsModifier->Reset();

    auto wMaterial = resourcesManager->GetMaterial(id);
    if (!wMaterial.expired()) {
        auto matInstance = wMaterial.lock()->createInstance();

        auto instanceId = resourcesManager->AddMaterialInstance(matInstance);
        auto wMaterialInstance = resourcesManager->GetMaterialInstance(instanceId);

        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), instanceId);
    }

    return *materialsModifier;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialInstanceHandle& id)
{
    materialsModifier->Reset();

    auto wMaterialInstance = resourcesManager->GetMaterialInstance(id);
    if (!wMaterialInstance.expired()) {
        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), id);
    }

    return *materialsModifier;
}

void FilamentRenderer::AssignMaterial(const GeometryHandle& geometryId, const MaterialInstanceHandle& materialId)
{
    auto wMaterialInstance = resourcesManager->GetMaterialInstance(materialId);
    auto found = entities.find(geometryId);
    if (found != entities.end() && false == wMaterialInstance.expired()) {
        auto& renderableManger = engine->getRenderableManager();
        filament::RenderableManager::Instance inst = renderableManger.getInstance(found->second.self);
        renderableManger.setMaterialInstanceAt(inst, 0, wMaterialInstance.lock().get());
    }
    // TODO: Log if failed
}

filament::VertexBuffer* FilamentRenderer::AllocateVertexBuffer(FilamentRenderer::AllocatedEntity& owner, const size_t verticesCount)
{
    using namespace filament;

    VertexBuffer* vbuf = VertexBuffer::Builder()
            .bufferCount(2)
            .vertexCount(verticesCount)
            .normalized(VertexAttribute::TANGENTS)
            .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0)
            .attribute(VertexAttribute::TANGENTS, 1, VertexBuffer::AttributeType::FLOAT4, 0)
            .build(*engine);

    if (vbuf) {
        owner.vb = resourcesManager->AddVertexBuffer(vbuf);
    }

    return vbuf;
}

filament::IndexBuffer* FilamentRenderer::AllocateIndexBuffer(FilamentRenderer::AllocatedEntity& owner, const size_t indicesCount, const size_t indexStride)
{
    using namespace filament;

    IndexBuffer* ibuf = IndexBuffer::Builder()
            .bufferType(indexStride == 2 ? IndexBuffer::IndexType::USHORT : IndexBuffer::IndexType::UINT)
            .indexCount(indicesCount)
            .build(*engine);

    if (ibuf) {
        owner.ib = resourcesManager->AddIndexBuffer(ibuf);
    }

    return ibuf;
}

void FilamentRenderer::RemoveEntity(REHandle_abstract id)
{
    auto found = entities.find(id);
    if (found != entities.end()) {
        const auto& data = found->second;
        scene->remove(data.self);

        resourcesManager->Destroy(data.vb);
        resourcesManager->Destroy(data.ib);
        engine->destroy(data.self);

        entities.erase(found);
    }
}

}
}
