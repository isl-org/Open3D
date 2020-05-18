// Altered from Filament's ImGuiHelper.cpp
// Filament code is from somewhere close to v1.4.3 and is:
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
// Open3D alterations are:
// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/GUI/ImguiFilamentBridge.h"

#include <fcntl.h>
#include <filament/Fence.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/filamat/MaterialBuilder.h>
#include <filament/utils/EntityManager.h>
#include <imgui.h>
#include <cerrno>
#include <cstddef>  // <filament/Engine> recursive includes needs this, std::size_t especially
#include <iostream>
#include <unordered_map>
#include <vector>

#if !defined(WIN32)
#include <unistd.h>
#else
#include <io.h>
#endif

#include "Open3D/GUI/Application.h"
#include "Open3D/GUI/Color.h"
#include "Open3D/GUI/Gui.h"
#include "Open3D/GUI/Theme.h"
#include "Open3D/GUI/Window.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentCamera.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentRenderer.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentScene.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentView.h"

using namespace filament::math;
using namespace filament;
using namespace utils;

namespace open3d {
namespace gui {

static std::string GetIOErrorString(int errno_val) {
    switch (errno_val) {
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
            s << "IO error " << errno_val << " (see cerrno)";
            return s.str();
        }
    }
}

static bool ReadBinaryFile(const std::string& path,
                           std::vector<char>* bytes,
                           std::string* error_str) {
    bytes->clear();
    if (error_str) {
        *error_str = "";
    }

    // Open file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        if (error_str) {
            *error_str = GetIOErrorString(errno);
        }
        return false;
    }

    // Get file size
    off_t filesize = (off_t)lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);  // reset file pointer back to beginning

    // Read data
    bytes->resize(filesize);
    bool result = true;
    if (read(fd, bytes->data(), filesize) != filesize) {
        result = false;
    }

    // We're done, close and return
    close(fd);
    return result;
}

static Material* LoadMaterialTemplate(const std::string& path, Engine& engine) {
    std::vector<char> bytes;
    std::string error_str;
    if (!ReadBinaryFile(path, &bytes, &error_str)) {
        std::cout << "[ERROR] Could not read " << path << ": " << error_str
                  << std::endl;
        return nullptr;
    }

    return Material::Builder()
            .package(bytes.data(), bytes.size())
            .build(engine);
}

struct ImguiFilamentBridge::Impl {
    // Bridge manages filament resources directly
    filament::Material* material_ = nullptr;
    std::vector<filament::VertexBuffer*> vertex_buffers_;
    std::vector<filament::IndexBuffer*> index_buffers_;
    std::vector<filament::MaterialInstance*> material_instances_;

    utils::Entity renderable_;
    filament::Texture* texture_ = nullptr;
    bool has_synced_ = false;

    visualization::FilamentView* view_ = nullptr;  // we do not own this
};

ImguiFilamentBridge::ImguiFilamentBridge(
        visualization::FilamentRenderer* renderer, const Size& window_size)
    : impl_(new ImguiFilamentBridge::Impl()) {
    // The UI needs a special material (just a pass-through blit)
    std::string resource_path = Application::GetInstance().GetResourcePath();
    impl_->material_ =
            LoadMaterialTemplate(resource_path + "/ui_blit.filamat",
                                 visualization::EngineInstance::GetInstance());

    auto scene_handle = renderer->CreateScene();
    renderer->ConvertToGuiScene(scene_handle);
    auto scene = renderer->GetGuiScene();

    auto view_id = scene->AddView(0, 0, window_size.width, window_size.height);
    impl_->view_ =
            dynamic_cast<visualization::FilamentView*>(scene->GetView(view_id));

    auto native_view = impl_->view_->GetNativeView();
    native_view->setClearTargets(false, false, false);
    native_view->setRenderTarget(View::TargetBufferFlags::DEPTH_AND_STENCIL);
    native_view->setPostProcessingEnabled(false);
    native_view->setShadowsEnabled(false);

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable_ = em.create();
    scene->GetNativeScene()->addEntity(impl_->renderable_);
}

ImguiFilamentBridge::ImguiFilamentBridge(filament::Engine* engine,
                                         filament::Scene* scene,
                                         filament::Material* uiblit_material)
    : impl_(new ImguiFilamentBridge::Impl()) {
    impl_->material_ = uiblit_material;

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable_ = em.create();
    scene->addEntity(impl_->renderable_);
}

void ImguiFilamentBridge::CreateAtlasTextureAlpha8(unsigned char* pixels,
                                                   int width,
                                                   int height,
                                                   int bytes_per_px) {
    auto& engine = visualization::EngineInstance::GetInstance();

    engine.destroy(impl_->texture_);

    size_t size = (size_t)(width * height);
    Texture::PixelBufferDescriptor pb(pixels, size, Texture::Format::R,
                                      Texture::Type::UBYTE);
    impl_->texture_ = Texture::Builder()
                              .width((uint32_t)width)
                              .height((uint32_t)height)
                              .levels((uint8_t)1)
                              .format(Texture::InternalFormat::R8)
                              .sampler(Texture::Sampler::SAMPLER_2D)
                              .build(engine);
    impl_->texture_->setImage(engine, 0, std::move(pb));

    TextureSampler sampler(TextureSampler::MinFilter::LINEAR,
                           TextureSampler::MagFilter::LINEAR);
    impl_->material_->setDefaultParameter("albedo", impl_->texture_, sampler);
}

ImguiFilamentBridge::~ImguiFilamentBridge() {
    auto& engine = visualization::EngineInstance::GetInstance();

    engine.destroy(impl_->renderable_);
    for (auto& mi : impl_->material_instances_) {
        engine.destroy(mi);
    }
    engine.destroy(impl_->material_);
    engine.destroy(impl_->texture_);
    for (auto& vb : impl_->vertex_buffers_) {
        engine.destroy(vb);
    }
    for (auto& ib : impl_->index_buffers_) {
        engine.destroy(ib);
    }
}

// To help with mapping unique scissor rectangles to material instances, we
// create a 64-bit key from a 4-tuple that defines an AABB in screen space.
static uint64_t MakeScissorKey(int fb_height, const ImVec4& clip_rect) {
    uint16_t left = (uint16_t)clip_rect.x;
    uint16_t bottom = (uint16_t)(fb_height - clip_rect.w);
    uint16_t width = (uint16_t)(clip_rect.z - clip_rect.x);
    uint16_t height = (uint16_t)(clip_rect.w - clip_rect.y);
    return ((uint64_t)left << 0ull) | ((uint64_t)bottom << 16ull) |
           ((uint64_t)width << 32ull) | ((uint64_t)height << 48ull);
}

void ImguiFilamentBridge::Update(ImDrawData* imgui_data) {
    impl_->has_synced_ = false;

    auto& engine = visualization::EngineInstance::GetInstance();

    auto& rcm = engine.getRenderableManager();

    // Avoid rendering when minimized and scale coordinates for retina displays.
    ImGuiIO& io = ImGui::GetIO();
    int fbwidth = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fbheight = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fbwidth == 0 || fbheight == 0) return;
    imgui_data->ScaleClipRects(io.DisplayFramebufferScale);

    // Ensure that we have enough vertex buffers and index buffers.
    CreateBuffers(imgui_data->CmdListsCount);

    // Count how many primitives we'll need, then create a Renderable builder.
    // Also count how many unique scissor rectangles are required.
    size_t num_prims = 0;
    std::unordered_map<uint64_t, filament::MaterialInstance*> scissor_rects;
    for (int idx = 0; idx < imgui_data->CmdListsCount; idx++) {
        const ImDrawList* cmds = imgui_data->CmdLists[idx];
        num_prims += cmds->CmdBuffer.size();
        for (const auto& pcmd : cmds->CmdBuffer) {
            scissor_rects[MakeScissorKey(fbheight, pcmd.ClipRect)] = nullptr;
        }
    }
    auto rbuilder = RenderableManager::Builder(num_prims);
    rbuilder.boundingBox({{0, 0, 0}, {10000, 10000, 10000}}).culling(false);

    // Ensure that we have a material instance for each scissor rectangle.
    size_t previous_size = impl_->material_instances_.size();
    if (scissor_rects.size() > impl_->material_instances_.size()) {
        impl_->material_instances_.resize(scissor_rects.size());
        for (size_t i = previous_size; i < impl_->material_instances_.size();
             i++) {
            impl_->material_instances_[i] = impl_->material_->createInstance();
        }
    }

    // Push each unique scissor rectangle to a MaterialInstance.
    size_t mat_index = 0;
    for (auto& pair : scissor_rects) {
        pair.second = impl_->material_instances_[mat_index++];
        uint32_t left = (pair.first >> 0ull) & 0xffffull;
        uint32_t bottom = (pair.first >> 16ull) & 0xffffull;
        uint32_t width = (pair.first >> 32ull) & 0xffffull;
        uint32_t height = (pair.first >> 48ull) & 0xffffull;
        pair.second->setScissor(left, bottom, width, height);
    }

    // Recreate the Renderable component and point it to the vertex buffers.
    rcm.destroy(impl_->renderable_);
    int buffer_index = 0;
    int prim_index = 0;
    for (int cmd_idx = 0; cmd_idx < imgui_data->CmdListsCount; cmd_idx++) {
        const ImDrawList* cmds = imgui_data->CmdLists[cmd_idx];
        size_t indexOffset = 0;
        PopulateVertexData(
                buffer_index, cmds->VtxBuffer.Size * sizeof(ImDrawVert),
                cmds->VtxBuffer.Data, cmds->IdxBuffer.Size * sizeof(ImDrawIdx),
                cmds->IdxBuffer.Data);
        for (const auto& pcmd : cmds->CmdBuffer) {
            if (pcmd.UserCallback) {
                pcmd.UserCallback(cmds, &pcmd);
            } else {
                uint64_t skey = MakeScissorKey(fbheight, pcmd.ClipRect);
                auto miter = scissor_rects.find(skey);
                assert(miter != scissor_rects.end());
                rbuilder.geometry(prim_index,
                                  RenderableManager::PrimitiveType::TRIANGLES,
                                  impl_->vertex_buffers_[buffer_index],
                                  impl_->index_buffers_[buffer_index],
                                  indexOffset, pcmd.ElemCount)
                        .blendOrder(prim_index, prim_index)
                        .material(prim_index, miter->second);
                prim_index++;
            }
            indexOffset += pcmd.ElemCount;
        }
        buffer_index++;
    }
    if (imgui_data->CmdListsCount > 0) {
        rbuilder.build(engine, impl_->renderable_);
    }
}

void ImguiFilamentBridge::OnWindowResized(const Window& window) {
    auto size = window.GetSize();
    impl_->view_->SetViewport(0, 0, size.width, size.height);

    auto camera = impl_->view_->GetCamera();
    camera->SetProjection(visualization::Camera::Projection::Ortho, 0.0,
                          size.width, size.height, 0.0, 0.0, 1.0);
}

void ImguiFilamentBridge::CreateVertexBuffer(size_t buffer_index,
                                             size_t capacity) {
    SyncThreads();

    auto& engine = visualization::EngineInstance::GetInstance();

    engine.destroy(impl_->vertex_buffers_[buffer_index]);
    impl_->vertex_buffers_[buffer_index] =
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
                    .build(engine);
}

void ImguiFilamentBridge::CreateIndexBuffer(size_t buffer_index,
                                            size_t capacity) {
    SyncThreads();

    auto& engine = visualization::EngineInstance::GetInstance();

    engine.destroy(impl_->index_buffers_[buffer_index]);
    impl_->index_buffers_[buffer_index] =
            IndexBuffer::Builder()
                    .indexCount(capacity)
                    .bufferType(IndexBuffer::IndexType::USHORT)
                    .build(engine);
}

void ImguiFilamentBridge::CreateBuffers(size_t num_required_buffers) {
    if (num_required_buffers > impl_->vertex_buffers_.size()) {
        size_t previous_size = impl_->vertex_buffers_.size();
        impl_->vertex_buffers_.resize(num_required_buffers, nullptr);
        for (size_t i = previous_size; i < impl_->vertex_buffers_.size(); i++) {
            // Pick a reasonable starting capacity; it will grow if needed.
            CreateVertexBuffer(i, 1000);
        }
    }
    if (num_required_buffers > impl_->index_buffers_.size()) {
        size_t previous_size = impl_->index_buffers_.size();
        impl_->index_buffers_.resize(num_required_buffers, nullptr);
        for (size_t i = previous_size; i < impl_->index_buffers_.size(); i++) {
            // Pick a reasonable starting capacity; it will grow if needed.
            CreateIndexBuffer(i, 5000);
        }
    }
}

void ImguiFilamentBridge::PopulateVertexData(size_t buffer_index,
                                             size_t vb_size_in_bytes,
                                             void* vb_imgui_data,
                                             size_t ib_size_in_bytes,
                                             void* ib_imgui_data) {
    auto& engine = visualization::EngineInstance::GetInstance();

    // Create a new vertex buffer if the size isn't large enough, then copy the
    // ImGui data into a staging area since Filament's render thread might
    // consume the data at any time.
    size_t required_vert_count = vb_size_in_bytes / sizeof(ImDrawVert);
    size_t capacity_vert_count =
            impl_->vertex_buffers_[buffer_index]->getVertexCount();
    if (required_vert_count > capacity_vert_count) {
        CreateVertexBuffer(buffer_index, required_vert_count);
    }
    size_t num_vb_bytes = required_vert_count * sizeof(ImDrawVert);
    void* vb_filament_data = malloc(num_vb_bytes);
    memcpy(vb_filament_data, vb_imgui_data, num_vb_bytes);
    impl_->vertex_buffers_[buffer_index]->setBufferAt(
            engine, 0,
            VertexBuffer::BufferDescriptor(
                    vb_filament_data, num_vb_bytes,
                    [](void* buffer, size_t size, void* user) { free(buffer); },
                    /* user = */ nullptr));

    // Create a new index buffer if the size isn't large enough, then copy the
    // ImGui data into a staging area since Filament's render thread might
    // consume the data at any time.
    size_t required_index_count = ib_size_in_bytes / 2;
    size_t capacity_index_count =
            impl_->index_buffers_[buffer_index]->getIndexCount();
    if (required_index_count > capacity_index_count) {
        CreateIndexBuffer(buffer_index, required_index_count);
    }
    size_t num_ib_bytes = required_index_count * 2;
    void* ib_filament_data = malloc(num_ib_bytes);
    memcpy(ib_filament_data, ib_imgui_data, num_ib_bytes);
    impl_->index_buffers_[buffer_index]->setBuffer(
            engine,
            IndexBuffer::BufferDescriptor(
                    ib_filament_data, num_ib_bytes,
                    [](void* buffer, size_t size, void* user) { free(buffer); },
                    /* user = */ nullptr));
}

void ImguiFilamentBridge::SyncThreads() {
#if UTILS_HAS_THREADING
    if (!impl_->has_synced_) {
        auto& engine = visualization::EngineInstance::GetInstance();

        // This is called only when ImGui needs to grow a vertex buffer, which
        // occurs a few times after launching and rarely (if ever) after that.
        Fence::waitAndDestroy(engine.createFence());
        impl_->has_synced_ = true;
    }
#endif
}

}  // namespace gui
}  // namespace open3d
