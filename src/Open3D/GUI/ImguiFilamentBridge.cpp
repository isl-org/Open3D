// Altered from Filament's ImGuiHelper.cpp

/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ImguiFilamentBridge.h"

#include "Color.h"
#include "Gui.h"
#include "Theme.h"

#include <vector>
#include <unordered_map>

#include <imgui.h>

#include <filamat/MaterialBuilder.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TransformManager.h>
#include <utils/EntityManager.h>

using namespace filament::math;
using namespace filament;
using namespace utils;

namespace open3d {
namespace gui {

struct ImguiFilamentBridge::Impl {
    filament::Engine* engine;
    filament::Material* material = nullptr;
    std::vector<filament::VertexBuffer*> vertexBuffers;
    std::vector<filament::IndexBuffer*> indexBuffers;
    std::vector<filament::MaterialInstance*> materialInstances;
    utils::Entity renderable;
    filament::Texture* texture = nullptr;
    unsigned char *fontPixels = nullptr;
    bool hasSynced = false;
};

ImguiFilamentBridge::ImguiFilamentBridge(Engine* engine, filament::Scene* scene,
                                         filament::Material *uiblitMaterial)
: impl_(new ImguiFilamentBridge::Impl()) {
    impl_->engine = engine;

    impl_->material = uiblitMaterial;

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable = em.create();
    scene->addEntity(impl_->renderable);
}

void ImguiFilamentBridge::createAtlasTextureAlpha8(unsigned char* pixels,
                                                   int width, int height,
                                                   int bytesPerPx) {
    impl_->engine->destroy(impl_->texture);
    size_t size = (size_t) (width * height);
    Texture::PixelBufferDescriptor pb(
            pixels, size,
            Texture::Format::R, Texture::Type::UBYTE);
    impl_->texture = Texture::Builder()
            .width((uint32_t) width)
            .height((uint32_t) height)
            .levels((uint8_t) 1)
            .format(Texture::InternalFormat::R8)
            .sampler(Texture::Sampler::SAMPLER_2D)
            .build(*impl_->engine);
    impl_->texture->setImage(*impl_->engine, 0, std::move(pb));

    TextureSampler sampler(TextureSampler::MinFilter::LINEAR, TextureSampler::MagFilter::LINEAR);
    impl_->material->setDefaultParameter("albedo", impl_->texture, sampler);
}

ImguiFilamentBridge::~ImguiFilamentBridge() {
    impl_->engine->destroy(impl_->renderable);
    for (auto& mi : impl_->materialInstances) {
        impl_->engine->destroy(mi);
    }
    impl_->engine->destroy(impl_->material);
    impl_->engine->destroy(impl_->texture);
    for (auto& vb : impl_->vertexBuffers) {
       impl_-> engine->destroy(vb);
    }
    for (auto& ib : impl_->indexBuffers) {
        impl_->engine->destroy(ib);
    }
}

// To help with mapping unique scissor rectangles to material instances, we create a 64-bit
// key from a 4-tuple that defines an AABB in screen space.
static uint64_t makeScissorKey(int fbheight, const ImVec4& clipRect) {
    uint16_t left = (uint16_t) clipRect.x;
    uint16_t bottom = (uint16_t) (fbheight - clipRect.w);
    uint16_t width = (uint16_t) (clipRect.z - clipRect.x);
    uint16_t height = (uint16_t) (clipRect.w - clipRect.y);
    return
            ((uint64_t)left << 0ull) |
            ((uint64_t)bottom << 16ull) |
            ((uint64_t)width << 32ull) |
            ((uint64_t)height << 48ull);
}

void ImguiFilamentBridge::update(ImDrawData* imguiData) {
    impl_->hasSynced = false;
    auto& rcm = impl_->engine->getRenderableManager();

    // Avoid rendering when minimized and scale coordinates for retina displays.
    ImGuiIO& io = ImGui::GetIO();
    int fbwidth = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fbheight = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fbwidth == 0 || fbheight == 0)
        return;
    imguiData->ScaleClipRects(io.DisplayFramebufferScale);

    // Ensure that we have enough vertex buffers and index buffers.
    createBuffers(imguiData->CmdListsCount);

    // Count how many primitives we'll need, then create a Renderable builder.
    // Also count how many unique scissor rectangles are required.
    size_t nPrims = 0;
    std::unordered_map<uint64_t, filament::MaterialInstance*> scissorRects;
    for (int cmdListIndex = 0; cmdListIndex < imguiData->CmdListsCount; cmdListIndex++) {
        const ImDrawList* cmds = imguiData->CmdLists[cmdListIndex];
        nPrims += cmds->CmdBuffer.size();
        for (const auto& pcmd : cmds->CmdBuffer) {
            scissorRects[makeScissorKey(fbheight, pcmd.ClipRect)] = nullptr;
        }
    }
    auto rbuilder = RenderableManager::Builder(nPrims);
    rbuilder.boundingBox({{ 0, 0, 0 }, { 10000, 10000, 10000 }}).culling(false);

    // Ensure that we have a material instance for each scissor rectangle.
    size_t previousSize = impl_->materialInstances.size();
    if (scissorRects.size() > impl_->materialInstances.size()) {
        impl_->materialInstances.resize(scissorRects.size());
        for (size_t i = previousSize; i < impl_->materialInstances.size(); i++) {
            impl_->materialInstances[i] = impl_->material->createInstance();
        }
    }

    // Push each unique scissor rectangle to a MaterialInstance.
    size_t matIndex = 0;
    for (auto& pair : scissorRects) {
        pair.second = impl_->materialInstances[matIndex++];
        uint32_t left = (pair.first >> 0ull) & 0xffffull;
        uint32_t bottom = (pair.first >> 16ull) & 0xffffull;
        uint32_t width = (pair.first >> 32ull) & 0xffffull;
        uint32_t height = (pair.first >> 48ull) & 0xffffull;
        pair.second->setScissor(left, bottom, width, height);
    }

    // Recreate the Renderable component and point it to the vertex buffers.
    rcm.destroy(impl_->renderable);
    int bufferIndex = 0;
    int primIndex = 0;
    for (int cmdListIndex = 0; cmdListIndex < imguiData->CmdListsCount; cmdListIndex++) {
        const ImDrawList* cmds = imguiData->CmdLists[cmdListIndex];
        size_t indexOffset = 0;
        populateVertexData(bufferIndex,
                cmds->VtxBuffer.Size * sizeof(ImDrawVert), cmds->VtxBuffer.Data,
                cmds->IdxBuffer.Size * sizeof(ImDrawIdx), cmds->IdxBuffer.Data);
        for (const auto& pcmd : cmds->CmdBuffer) {
            if (pcmd.UserCallback) {
                pcmd.UserCallback(cmds, &pcmd);
            } else {
                uint64_t skey = makeScissorKey(fbheight, pcmd.ClipRect);
                auto miter = scissorRects.find(skey);
                assert(miter != scissorRects.end());
                rbuilder
                        .geometry(primIndex, RenderableManager::PrimitiveType::TRIANGLES,
                                impl_->vertexBuffers[bufferIndex], impl_->indexBuffers[bufferIndex],
                                indexOffset, pcmd.ElemCount)
                        .blendOrder(primIndex, primIndex)
                        .material(primIndex, miter->second);
                primIndex++;
            }
            indexOffset += pcmd.ElemCount;
        }
        bufferIndex++;
    }
    if (imguiData->CmdListsCount > 0) {
        rbuilder.build(*impl_->engine, impl_->renderable);
    }
}

void ImguiFilamentBridge::createVertexBuffer(size_t bufferIndex, size_t capacity) {
    syncThreads();
    impl_->engine->destroy(impl_->vertexBuffers[bufferIndex]);
    impl_->vertexBuffers[bufferIndex] = VertexBuffer::Builder()
            .vertexCount(capacity)
            .bufferCount(1)
            .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT2, 0,
                    sizeof(ImDrawVert))
            .attribute(VertexAttribute::UV0, 0, VertexBuffer::AttributeType::FLOAT2,
                    sizeof(filament::math::float2), sizeof(ImDrawVert))
            .attribute(VertexAttribute::COLOR, 0, VertexBuffer::AttributeType::UBYTE4,
                    2 * sizeof(filament::math::float2), sizeof(ImDrawVert))
            .normalized(VertexAttribute::COLOR)
            .build(*impl_->engine);
}

void ImguiFilamentBridge::createIndexBuffer(size_t bufferIndex, size_t capacity) {
    syncThreads();
    impl_->engine->destroy(impl_->indexBuffers[bufferIndex]);
    impl_->indexBuffers[bufferIndex] = IndexBuffer::Builder()
            .indexCount(capacity)
            .bufferType(IndexBuffer::IndexType::USHORT)
            .build(*impl_->engine);
}

void ImguiFilamentBridge::createBuffers(int numRequiredBuffers) {
    if (numRequiredBuffers > impl_->vertexBuffers.size()) {
        size_t previousSize = impl_->vertexBuffers.size();
        impl_->vertexBuffers.resize(numRequiredBuffers, nullptr);
        for (size_t i = previousSize; i < impl_->vertexBuffers.size(); i++) {
            // Pick a reasonable starting capacity; it will grow if needed.
            createVertexBuffer(i, 1000);
        }
    }
    if (numRequiredBuffers > impl_->indexBuffers.size()) {
        size_t previousSize = impl_->indexBuffers.size();
        impl_->indexBuffers.resize(numRequiredBuffers, nullptr);
        for (size_t i = previousSize; i < impl_->indexBuffers.size(); i++) {
            // Pick a reasonable starting capacity; it will grow if needed.
            createIndexBuffer(i, 5000);
        }
    }
}

void ImguiFilamentBridge::populateVertexData(size_t bufferIndex,
                                             size_t vbSizeInBytes,
                                             void* vbImguiData,
                                             size_t ibSizeInBytes,
                                             void* ibImguiData)
{
    // Create a new vertex buffer if the size isn't large enough, then copy the ImGui data into
    // a staging area since Filament's render thread might consume the data at any time.
    size_t requiredVertCount = vbSizeInBytes / sizeof(ImDrawVert);
    size_t capacityVertCount = impl_->vertexBuffers[bufferIndex]->getVertexCount();
    if (requiredVertCount > capacityVertCount) {
        createVertexBuffer(bufferIndex, requiredVertCount);
    }
    size_t nVbBytes = requiredVertCount * sizeof(ImDrawVert);
    void* vbFilamentData = malloc(nVbBytes);
    memcpy(vbFilamentData, vbImguiData, nVbBytes);
    impl_->vertexBuffers[bufferIndex]->setBufferAt(*impl_->engine, 0,
            VertexBuffer::BufferDescriptor(vbFilamentData, nVbBytes,
                [](void* buffer, size_t size, void* user) {
                    free(buffer);
                }, /* user = */ nullptr));

    // Create a new index buffer if the size isn't large enough, then copy the ImGui data into
    // a staging area since Filament's render thread might consume the data at any time.
    size_t requiredIndexCount = ibSizeInBytes / 2;
    size_t capacityIndexCount = impl_->indexBuffers[bufferIndex]->getIndexCount();
    if (requiredIndexCount > capacityIndexCount) {
        createIndexBuffer(bufferIndex, requiredIndexCount);
    }
    size_t nIbBytes = requiredIndexCount * 2;
    void* ibFilamentData = malloc(nIbBytes);
    memcpy(ibFilamentData, ibImguiData, nIbBytes);
    impl_->indexBuffers[bufferIndex]->setBuffer(*impl_->engine,
            IndexBuffer::BufferDescriptor(ibFilamentData, nIbBytes,
                [](void* buffer, size_t size, void* user) {
                    free(buffer);
                }, /* user = */ nullptr));
}

void ImguiFilamentBridge::syncThreads() {
#if UTILS_HAS_THREADING
    if (!impl_->hasSynced) {
        // This is called only when ImGui needs to grow a vertex buffer, which occurs a few times
        // after launching and rarely (if ever) after that.
        Fence::waitAndDestroy(impl_->engine->createFence());
        impl_->hasSynced = true;
    }
#endif
}

} // gui
} // open3d
