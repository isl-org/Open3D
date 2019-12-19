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

#include "Application.h"
#include "Color.h"
#include "Gui.h"
#include "Theme.h"
#include "Window.h"

#include "Open3D/Visualization/Rendering/Filament/FilamentCamera.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentRenderer.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentScene.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentView.h"

#include <unordered_map>
#include <vector>

#include <imgui.h>

#include <filamat/MaterialBuilder.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <utils/EntityManager.h>

#include <fcntl.h>
#include <cerrno>
#if !defined(WIN32)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace filament::math;
using namespace filament;
using namespace utils;

namespace open3d {
namespace gui {

static std::string getIOErrorString(int errnoVal) {
    switch (errnoVal) {
        case EPERM:
            return "Operation not permitted";
        case EACCES:
            return "Access denied";
        case EAGAIN:
            return "EAGAIN";
#ifndef WIN32
        case EDQUOT:
            return "Over quota";
#endif
        case EEXIST:
            return "File already exists";
        case EFAULT:
            return "Bad filename pointer";
        case EINTR:
            return "open() interrupted by a signal";
        case EIO:
            return "I/O error";
        case ELOOP:
            return "Too many symlinks, could be a loop";
        case EMFILE:
            return "Process is out of file descriptors";
        case ENAMETOOLONG:
            return "Filename is too long";
        case ENFILE:
            return "File system table is full";
        case ENOENT:
            return "No such file or directory";
        case ENOSPC:
            return "No space available to create file";
        case ENOTDIR:
            return "Bad path";
        case EOVERFLOW:
            return "File is too big";
        case EROFS:
            return "Can't modify file on read-only filesystem";
        default: {
            std::stringstream s;
            s << "IO error " << errnoVal << " (see cerrno)";
            return s.str();
        }
    }
}

static bool readBinaryFile(const std::string& path,
                           std::vector<char>* bytes,
                           std::string* errorStr) {
    bytes->clear();
    if (errorStr) {
        *errorStr = "";
    }

    // Open file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        if (errorStr) {
            *errorStr = getIOErrorString(errno);
        }
        return false;
    }

    // Get file size
    size_t filesize = (size_t)lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);  // reset file pointer back to beginning

    // Read data
    bytes->resize(filesize);
    read(fd, bytes->data(), filesize);

    // We're done, close and return
    close(fd);
    return true;
}

static Material* loadMaterialTemplate(const std::string& path, Engine& engine) {
    std::vector<char> bytes;
    std::string errorStr;
    if (!readBinaryFile(path, &bytes, &errorStr)) {
        std::cout << "[ERROR] Could not read " << path << ": " << errorStr
                  << std::endl;
        return nullptr;
    }

    return Material::Builder()
            .package(bytes.data(), bytes.size())
            .build(engine);
}

struct ImguiFilamentBridge::Impl {
    // Bridge is managing filament resources by itself
    filament::Material* material = nullptr;
    std::vector<filament::VertexBuffer*> vertexBuffers;
    std::vector<filament::IndexBuffer*> indexBuffers;
    std::vector<filament::MaterialInstance*> materialInstances;

    utils::Entity renderable;
    filament::Texture* texture = nullptr;
    unsigned char* fontPixels = nullptr;
    bool hasSynced = false;

    visualization::FilamentView* view = nullptr;  // we are not owning this
};

ImguiFilamentBridge::ImguiFilamentBridge(
        visualization::FilamentRenderer* renderer, const Size& windowSize)
    : impl_(new ImguiFilamentBridge::Impl()) {
    // The UI needs a special material (just a pass-through blit)
    std::string resourcePath = Application::GetInstance().GetResourcePath();
    impl_->material =
            loadMaterialTemplate(resourcePath + "/ui_blit.filamat",
                                 visualization::EngineInstance::GetInstance());

    auto sceneHandle = renderer->CreateScene();
    renderer->ConvertToGuiScene(sceneHandle);
    auto scene = renderer->GetGuiScene();

    auto viewId = scene->AddView(0, 0, windowSize.width, windowSize.height);
    impl_->view =
            dynamic_cast<visualization::FilamentView*>(scene->GetView(viewId));

    auto nativeView = impl_->view->GetNativeView();
    nativeView->setClearTargets(false, false, false);
    nativeView->setRenderTarget(View::TargetBufferFlags::DEPTH_AND_STENCIL);
    nativeView->setPostProcessingEnabled(false);
    nativeView->setShadowsEnabled(false);

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable = em.create();
    scene->GetNativeScene()->addEntity(impl_->renderable);
}

ImguiFilamentBridge::ImguiFilamentBridge(Engine* engine,
                                         filament::Scene* scene,
                                         filament::Material* uiblitMaterial)
    : impl_(new ImguiFilamentBridge::Impl()) {
    impl_->material = uiblitMaterial;

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable = em.create();
    scene->addEntity(impl_->renderable);
}

void ImguiFilamentBridge::createAtlasTextureAlpha8(unsigned char* pixels,
                                                   int width,
                                                   int height,
                                                   int bytesPerPx) {
    auto& engineInstance = visualization::EngineInstance::GetInstance();

    engineInstance.destroy(impl_->texture);

    size_t size = (size_t)(width * height);
    Texture::PixelBufferDescriptor pb(pixels, size, Texture::Format::R,
                                      Texture::Type::UBYTE);
    impl_->texture = Texture::Builder()
                             .width((uint32_t)width)
                             .height((uint32_t)height)
                             .levels((uint8_t)1)
                             .format(Texture::InternalFormat::R8)
                             .sampler(Texture::Sampler::SAMPLER_2D)
                             .build(engineInstance);
    impl_->texture->setImage(engineInstance, 0, std::move(pb));

    TextureSampler sampler(TextureSampler::MinFilter::LINEAR,
                           TextureSampler::MagFilter::LINEAR);
    impl_->material->setDefaultParameter("albedo", impl_->texture, sampler);
}

ImguiFilamentBridge::~ImguiFilamentBridge() {
    auto& engineInstance = visualization::EngineInstance::GetInstance();

    engineInstance.destroy(impl_->renderable);
    for (auto& mi : impl_->materialInstances) {
        engineInstance.destroy(mi);
    }
    engineInstance.destroy(impl_->material);
    engineInstance.destroy(impl_->texture);
    for (auto& vb : impl_->vertexBuffers) {
        engineInstance.destroy(vb);
    }
    for (auto& ib : impl_->indexBuffers) {
        engineInstance.destroy(ib);
    }
}

// To help with mapping unique scissor rectangles to material instances, we
// create a 64-bit key from a 4-tuple that defines an AABB in screen space.
static uint64_t makeScissorKey(int fbheight, const ImVec4& clipRect) {
    uint16_t left = (uint16_t)clipRect.x;
    uint16_t bottom = (uint16_t)(fbheight - clipRect.w);
    uint16_t width = (uint16_t)(clipRect.z - clipRect.x);
    uint16_t height = (uint16_t)(clipRect.w - clipRect.y);
    return ((uint64_t)left << 0ull) | ((uint64_t)bottom << 16ull) |
           ((uint64_t)width << 32ull) | ((uint64_t)height << 48ull);
}

void ImguiFilamentBridge::update(ImDrawData* imguiData) {
    impl_->hasSynced = false;

    auto& engineInstance = visualization::EngineInstance::GetInstance();

    auto& rcm = engineInstance.getRenderableManager();

    // Avoid rendering when minimized and scale coordinates for retina displays.
    ImGuiIO& io = ImGui::GetIO();
    int fbwidth = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fbheight = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fbwidth == 0 || fbheight == 0) return;
    imguiData->ScaleClipRects(io.DisplayFramebufferScale);

    // Ensure that we have enough vertex buffers and index buffers.
    createBuffers(imguiData->CmdListsCount);

    // Count how many primitives we'll need, then create a Renderable builder.
    // Also count how many unique scissor rectangles are required.
    size_t nPrims = 0;
    std::unordered_map<uint64_t, filament::MaterialInstance*> scissorRects;
    for (int cmdListIndex = 0; cmdListIndex < imguiData->CmdListsCount;
         cmdListIndex++) {
        const ImDrawList* cmds = imguiData->CmdLists[cmdListIndex];
        nPrims += cmds->CmdBuffer.size();
        for (const auto& pcmd : cmds->CmdBuffer) {
            scissorRects[makeScissorKey(fbheight, pcmd.ClipRect)] = nullptr;
        }
    }
    auto rbuilder = RenderableManager::Builder(nPrims);
    rbuilder.boundingBox({{0, 0, 0}, {10000, 10000, 10000}}).culling(false);

    // Ensure that we have a material instance for each scissor rectangle.
    size_t previousSize = impl_->materialInstances.size();
    if (scissorRects.size() > impl_->materialInstances.size()) {
        impl_->materialInstances.resize(scissorRects.size());
        for (size_t i = previousSize; i < impl_->materialInstances.size();
             i++) {
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
    for (int cmdListIndex = 0; cmdListIndex < imguiData->CmdListsCount;
         cmdListIndex++) {
        const ImDrawList* cmds = imguiData->CmdLists[cmdListIndex];
        size_t indexOffset = 0;
        populateVertexData(
                bufferIndex, cmds->VtxBuffer.Size * sizeof(ImDrawVert),
                cmds->VtxBuffer.Data, cmds->IdxBuffer.Size * sizeof(ImDrawIdx),
                cmds->IdxBuffer.Data);
        for (const auto& pcmd : cmds->CmdBuffer) {
            if (pcmd.UserCallback) {
                pcmd.UserCallback(cmds, &pcmd);
            } else {
                uint64_t skey = makeScissorKey(fbheight, pcmd.ClipRect);
                auto miter = scissorRects.find(skey);
                assert(miter != scissorRects.end());
                rbuilder.geometry(primIndex,
                                  RenderableManager::PrimitiveType::TRIANGLES,
                                  impl_->vertexBuffers[bufferIndex],
                                  impl_->indexBuffers[bufferIndex], indexOffset,
                                  pcmd.ElemCount)
                        .blendOrder(primIndex, primIndex)
                        .material(primIndex, miter->second);
                primIndex++;
            }
            indexOffset += pcmd.ElemCount;
        }
        bufferIndex++;
    }
    if (imguiData->CmdListsCount > 0) {
        rbuilder.build(engineInstance, impl_->renderable);
    }
}

void ImguiFilamentBridge::onWindowResized(const Window& window) {
    auto size = window.GetSize();
    impl_->view->SetViewport(0, 0, size.width, size.height);

    auto camera = impl_->view->GetCamera();
    camera->SetProjection(visualization::Camera::Projection::Ortho, 0.0,
                          size.width, size.height, 0.0, 0.0, 1.0);
}

void ImguiFilamentBridge::createVertexBuffer(size_t bufferIndex,
                                             size_t capacity) {
    syncThreads();

    auto& engineInstance = visualization::EngineInstance::GetInstance();

    engineInstance.destroy(impl_->vertexBuffers[bufferIndex]);
    impl_->vertexBuffers[bufferIndex] =
            VertexBuffer::Builder()
                    .vertexCount(capacity)
                    .bufferCount(1)
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT2, 0,
                               sizeof(ImDrawVert))
                    .attribute(VertexAttribute::UV0, 0,
                               VertexBuffer::AttributeType::FLOAT2,
                               sizeof(filament::math::float2),
                               sizeof(ImDrawVert))
                    .attribute(VertexAttribute::COLOR, 0,
                               VertexBuffer::AttributeType::UBYTE4,
                               2 * sizeof(filament::math::float2),
                               sizeof(ImDrawVert))
                    .normalized(VertexAttribute::COLOR)
                    .build(engineInstance);
}

void ImguiFilamentBridge::createIndexBuffer(size_t bufferIndex,
                                            size_t capacity) {
    syncThreads();

    auto& engineInstance = visualization::EngineInstance::GetInstance();

    engineInstance.destroy(impl_->indexBuffers[bufferIndex]);
    impl_->indexBuffers[bufferIndex] =
            IndexBuffer::Builder()
                    .indexCount(capacity)
                    .bufferType(IndexBuffer::IndexType::USHORT)
                    .build(engineInstance);
}

void ImguiFilamentBridge::createBuffers(size_t numRequiredBuffers) {
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
                                             void* ibImguiData) {
    auto& engineInstance = visualization::EngineInstance::GetInstance();

    // Create a new vertex buffer if the size isn't large enough, then copy the
    // ImGui data into a staging area since Filament's render thread might
    // consume the data at any time.
    size_t requiredVertCount = vbSizeInBytes / sizeof(ImDrawVert);
    size_t capacityVertCount =
            impl_->vertexBuffers[bufferIndex]->getVertexCount();
    if (requiredVertCount > capacityVertCount) {
        createVertexBuffer(bufferIndex, requiredVertCount);
    }
    size_t nVbBytes = requiredVertCount * sizeof(ImDrawVert);
    void* vbFilamentData = malloc(nVbBytes);
    memcpy(vbFilamentData, vbImguiData, nVbBytes);
    impl_->vertexBuffers[bufferIndex]->setBufferAt(
            engineInstance, 0,
            VertexBuffer::BufferDescriptor(
                    vbFilamentData, nVbBytes,
                    [](void* buffer, size_t size, void* user) { free(buffer); },
                    /* user = */ nullptr));

    // Create a new index buffer if the size isn't large enough, then copy the
    // ImGui data into a staging area since Filament's render thread might
    // consume the data at any time.
    size_t requiredIndexCount = ibSizeInBytes / 2;
    size_t capacityIndexCount =
            impl_->indexBuffers[bufferIndex]->getIndexCount();
    if (requiredIndexCount > capacityIndexCount) {
        createIndexBuffer(bufferIndex, requiredIndexCount);
    }
    size_t nIbBytes = requiredIndexCount * 2;
    void* ibFilamentData = malloc(nIbBytes);
    memcpy(ibFilamentData, ibImguiData, nIbBytes);
    impl_->indexBuffers[bufferIndex]->setBuffer(
            engineInstance,
            IndexBuffer::BufferDescriptor(
                    ibFilamentData, nIbBytes,
                    [](void* buffer, size_t size, void* user) { free(buffer); },
                    /* user = */ nullptr));
}

void ImguiFilamentBridge::syncThreads() {
#if UTILS_HAS_THREADING
    if (!impl_->hasSynced) {
        auto& engineInstance = visualization::EngineInstance::GetInstance();

        // This is called only when ImGui needs to grow a vertex buffer, which
        // occurs a few times after launching and rarely (if ever) after that.
        Fence::waitAndDestroy(engineInstance.createFence());
        impl_->hasSynced = true;
    }
#endif
}

}  // namespace gui
}  // namespace open3d
