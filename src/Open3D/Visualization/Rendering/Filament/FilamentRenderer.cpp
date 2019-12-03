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

#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentCamera.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEntitiesMods.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentView.h"

namespace open3d
{
namespace visualization
{

typedef REHandle<eEntityType::VertexBuffer> VertexBufferHandle;
typedef REHandle<eEntityType::IndexBuffer> IndexBufferHandle;

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

    // TODO: remove after tests
    auto sun = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::SUN)
            .direction(filament::math::float3{ -0.707, -.707, 0.0 })
            .intensity(100000)
            .castShadows(true)
            .build(*engine, sun);

    scene->addEntity(sun);

    view->SetViewport({0,0,1280,720});
    view->GetCamera()->LookAt(0, 0, 0,   80, 80, 80,   0, 1, 0);
    view->SetClearColor(filament::LinearColorA(0.5f,0.5f,1.f,1.f));
}

FilamentRenderer::~FilamentRenderer()
{
    view.reset();

    engine->destroy(scene);
    engine->destroy(renderer);
    engine->destroy(swapChain);

    filament::Engine::destroy(engine);
}

void FilamentRenderer::Draw()
{
    if (renderer->beginFrame(swapChain)) {
        // for each View
        renderer->render(view->GetNativeView());
        renderer->endFrame();
    }
}

GeometryHandle FilamentRenderer::AddGeometry(const geometry::Geometry3D& geometry, const MaterialInstanceHandle& materialId)
{
    using namespace geometry;
    using namespace filament;

    if (geometry.GetGeometryType() != Geometry::GeometryType::TriangleMesh)
    {
        return GeometryHandle::kBad;
    }

    auto triangleMesh = static_cast<const TriangleMesh&>(geometry);
    const size_t nVertices = triangleMesh.vertices_.size();

    VertexBuffer* vbuf = AllocateVertexBuffer(nVertices);

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
    IndexBuffer* ibuf = AllocateIndexBuffer(triangleMesh.triangles_.size() * 3, indexStride);

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

    const auto bboxMax = triangleMesh.GetMaxBound();
    const auto bboxMin = triangleMesh.GetMinBound();

    utils::Entity renderable = utils::EntityManager::get().create();
    RenderableManager::Builder builder(1);
    builder.boundingBox({{ bboxMin.x(), bboxMin.y(), bboxMin.z() }, { bboxMax.x(), bboxMax.y(), bboxMax.z() }})
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vbuf, ibuf)
        .culling(false);

    auto matInstance = GetMaterialInstance(materialId);
    if (matInstance) {
        builder.material(0, matInstance);
    }

    builder.build(*engine, renderable);

    auto handle = GeometryHandle::Next();
    entities[handle] = renderable;

    scene->addEntity(renderable);

    return handle;
}

void FilamentRenderer::UpdateGeometry(const GeometryHandle& id, const geometry::Geometry3D& geometry)
{}

void FilamentRenderer::RemoveGeometry(const GeometryHandle& geometryId)
{}

LightHandle FilamentRenderer::AddLight(const LightDescription& descr)
{
    return LightHandle::Next();
}

void FilamentRenderer::RemoveLight(const LightHandle& id)
{}

CameraHandle FilamentRenderer::AddCamera(const CameraDescription& descr)
{
    return CameraHandle::Next();
}

void FilamentRenderer::RemoveCamera(const CameraHandle& id)
{}

MaterialHandle FilamentRenderer::AddMaterial(const void* materialData, const size_t dataSize)
{
    using namespace filament;

    Material* mat = Material::Builder()
            .package(materialData, dataSize)
            .build(*engine);

    MaterialHandle handle;
    if (mat) {
        handle = MaterialHandle::Next();
        utilites[handle] = mat;
    }

    return handle;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id)
{
    materialsModifier->Reset();

    auto found = utilites.find(id);
    if (found != utilites.end()) {
        auto material = static_cast<filament::Material*>(found->second);
        auto matInstance = material->createInstance();

        auto instanceId = MaterialInstanceHandle::Next();
        utilites[instanceId] = matInstance;

        materialsModifier->InitWithMaterialInstance(matInstance, instanceId);
    }

    return *materialsModifier;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialInstanceHandle& id)
{
    materialsModifier->Reset();

    auto matInstance = GetMaterialInstance(id);
    if (matInstance) {
        materialsModifier->InitWithMaterialInstance(matInstance, id);
    }

    return *materialsModifier;
}

void FilamentRenderer::AssignMaterial(const GeometryHandle& geometryId, const MaterialInstanceHandle& materialId)
{
    auto matInstance = GetMaterialInstance(materialId);
    auto found = entities.find(geometryId);
    if (found != entities.end() && matInstance) {
        auto& renderableManger = engine->getRenderableManager();
        filament::RenderableManager::Instance inst = renderableManger.getInstance(found->second);
        renderableManger.setMaterialInstanceAt(inst, 0, matInstance);
    }
    // TODO: Log if failed
}

filament::VertexBuffer* FilamentRenderer::AllocateVertexBuffer(const size_t verticesCount)
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
        utilites[VertexBufferHandle::Next()] = vbuf;
    }

    return vbuf;
}

filament::IndexBuffer* FilamentRenderer::AllocateIndexBuffer(const size_t indicesCount, const size_t indexStride)
{
    using namespace filament;

    IndexBuffer* ibuf = IndexBuffer::Builder()
            .bufferType(indexStride == 2 ? IndexBuffer::IndexType::USHORT : IndexBuffer::IndexType::UINT)
            .indexCount(indicesCount)
            .build(*engine);

    if (ibuf) {
        utilites[IndexBufferHandle::Next()] = ibuf;
    }

    return ibuf;
}

filament::MaterialInstance* FilamentRenderer::GetMaterialInstance(const MaterialInstanceHandle& materialId) const
{
    filament::MaterialInstance* matInstance = nullptr;

    auto found = utilites.find(materialId);
    if (found != utilites.end()) {
        matInstance = static_cast<filament::MaterialInstance*>(found->second);
    }

    return matInstance;
}

}
}
