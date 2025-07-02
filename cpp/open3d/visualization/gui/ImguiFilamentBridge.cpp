// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
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

// clang-format off
// NOTE: this must precede header files that bring in Filament headers
#include <cstring>
// clang-format on

#include "open3d/visualization/gui/ImguiFilamentBridge.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4305: LightManager.h needs to specify some constants as floats
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4305)
#endif  // _MSC_VER

#include <filament/Fence.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
#include <utils/EntityManager.h>

#include <cerrno>
#include <iostream>
#include <map>
#include <vector>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <fcntl.h>
#include <imgui.h>

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/Gui.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/filament/FilamentCamera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

using namespace filament::math;
using namespace filament;
using namespace utils;

namespace open3d {
namespace visualization {
namespace gui {

static Material* LoadMaterialTemplate(const std::string& path, Engine& engine) {
    std::vector<char> bytes;
    std::string error_str;
    if (!utility::filesystem::FReadToBuffer(path, bytes, &error_str)) {
        std::cout << "[ERROR] Could not read " << path << ": " << error_str
                  << std::endl;
        return nullptr;
    }

    return Material::Builder()
            .package(bytes.data(), bytes.size())
            .build(engine);
}

class MaterialPool {
public:
    MaterialPool() {};

    MaterialPool(filament::Engine* engine,
                 filament::Material* material_template) {
        engine_ = engine;
        template_ = material_template;
        reset();
    }

    // Drains the pool and deallocates all the objects.
    void drain() {
        for (auto* o : pool_) {
            engine_->destroy(o);
        }
        pool_.clear();
    }

    // Invalidates all the pointers previously given out and "refills"
    // the pool with them.
    void reset() { next_index_ = 0; }

    // Returns an object from the pool. The pool retains ownership.
    filament::MaterialInstance* pull() {
        if (next_index_ >= pool_.size()) {
            pool_.push_back(template_->createInstance());
        }
        return pool_[next_index_++];
    }

private:
    filament::Engine* engine_ = nullptr;             // we do not own this
    filament::Material* template_ = nullptr;         // we do not own this
    std::vector<filament::MaterialInstance*> pool_;  // we DO own these
    size_t next_index_ = 0;
};

static const char* kUiBlitTexParamName = "albedo";
static const char* kImageTexParamName = "image";

struct ImguiFilamentBridge::Impl {
    // Bridge manages filament resources directly
    filament::Material* uiblit_material_ = nullptr;
    filament::Material* image_material_ = nullptr;
    std::vector<filament::VertexBuffer*> vertex_buffers_;
    std::vector<filament::IndexBuffer*> index_buffers_;

    // Use material pools to avoid allocating materials during every draw
    MaterialPool uiblit_pool_;
    MaterialPool image_pool_;

    utils::Entity renderable_;
    filament::Texture* font_texture_ = nullptr;
    bool has_synced_ = false;

    visualization::rendering::FilamentRenderer* renderer_ =
            nullptr;  // we do not own this
    visualization::rendering::FilamentView* view_ =
            nullptr;  // we do not own this
};

ImguiFilamentBridge::ImguiFilamentBridge(
        visualization::rendering::FilamentRenderer* renderer,
        const Size& window_size)
    : impl_(new ImguiFilamentBridge::Impl()) {
    impl_->renderer_ = renderer;
    // The UI needs a special material (just a pass-through blit)
    std::string resource_path = Application::GetInstance().GetResourcePath();
    impl_->uiblit_material_ = LoadMaterialTemplate(
            resource_path + "/ui_blit.filamat",
            visualization::rendering::EngineInstance::GetInstance());
    impl_->image_material_ = LoadMaterialTemplate(
            resource_path + "/img_blit.filamat",
            visualization::rendering::EngineInstance::GetInstance());

    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    impl_->uiblit_pool_ = MaterialPool(&engine, impl_->uiblit_material_);
    impl_->image_pool_ = MaterialPool(&engine, impl_->image_material_);

    auto scene_handle = renderer->CreateScene();
    renderer->ConvertToGuiScene(scene_handle);
    auto scene = renderer->GetGuiScene();

    auto view_id = scene->AddView(0, 0, window_size.width, window_size.height);
    impl_->view_ = dynamic_cast<visualization::rendering::FilamentView*>(
            scene->GetView(view_id));

    auto native_view = impl_->view_->GetNativeView();
    native_view->setPostProcessingEnabled(false);
    native_view->setShadowingEnabled(false);

    EntityManager& em = utils::EntityManager::get();
    impl_->renderable_ = em.create();
    scene->GetNativeScene()->addEntity(impl_->renderable_);
}

void ImguiFilamentBridge::CreateAtlasTextureAlpha8(unsigned char* pixels,
                                                   int width,
                                                   int height,
                                                   int bytes_per_px) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();

    engine.destroy(impl_->font_texture_);

    size_t size = (size_t)(width * height);
    Texture::PixelBufferDescriptor pb(pixels, size, Texture::Format::R,
                                      Texture::Type::UBYTE);
    impl_->font_texture_ = Texture::Builder()
                                   .width((uint32_t)width)
                                   .height((uint32_t)height)
                                   .levels((uint8_t)1)
                                   .format(Texture::InternalFormat::R8)
                                   .sampler(Texture::Sampler::SAMPLER_2D)
                                   .build(engine);
    impl_->font_texture_->setImage(engine, 0, std::move(pb));

    TextureSampler sampler(TextureSampler::MinFilter::LINEAR,
                           TextureSampler::MagFilter::LINEAR);
    impl_->uiblit_material_->setDefaultParameter(kUiBlitTexParamName,
                                                 impl_->font_texture_, sampler);
}

ImguiFilamentBridge::~ImguiFilamentBridge() {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();

    engine.destroy(impl_->renderable_);
    impl_->uiblit_pool_.drain();
    impl_->image_pool_.drain();
    engine.destroy(impl_->uiblit_material_);
    engine.destroy(impl_->image_material_);
    engine.destroy(impl_->font_texture_);
    for (auto& vb : impl_->vertex_buffers_) {
        engine.destroy(vb);
    }
    for (auto& ib : impl_->index_buffers_) {
        engine.destroy(ib);
    }
}

// To help with mapping unique scissor rectangles to material instances, we
// create a 64-bit key from a 4-tuple that defines an AABB in screen space.
class ScissorRectKey {
public:
    ScissorRectKey(int fb_height, const ImVec4& clip_rect, ImTextureID tex_id) {
        left_ = (uint16_t)clip_rect.x;
        bottom_ = (uint16_t)(fb_height - clip_rect.w);
        width_ = (uint16_t)(clip_rect.z - clip_rect.x);
        height_ = (uint16_t)(clip_rect.w - clip_rect.y);
        rect_ = ((uint64_t)left_ << 0ull) | ((uint64_t)bottom_ << 16ull) |
                ((uint64_t)width_ << 32ull) | ((uint64_t)height_ << 48ull);
        id_ = tex_id;
    }

    bool operator==(const ScissorRectKey& other) const {
        if (id_ == other.id_) {
            return (rect_ == other.rect_);
        }
        return false;
    }

    bool operator!=(const ScissorRectKey& other) const {
        return !operator==(other);
    }

    bool operator<(const ScissorRectKey& other) const {
        if (id_ == other.id_) {
            return (rect_ < other.rect_);
        }
        return (id_ < other.id_);
    }

    // Used for comparisons
    uint64_t rect_;
    ImTextureID id_;

    // Not used for comparisons
    uint16_t left_;
    uint16_t bottom_;
    uint16_t width_;
    uint16_t height_;
};

void ImguiFilamentBridge::Update(ImDrawData* imgui_data) {
    impl_->has_synced_ = false;

    auto& engine = visualization::rendering::EngineInstance::GetInstance();

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
    std::map<ScissorRectKey, filament::MaterialInstance*> scissor_rects;
    for (int idx = 0; idx < imgui_data->CmdListsCount; idx++) {
        const ImDrawList* cmds = imgui_data->CmdLists[idx];
        num_prims += cmds->CmdBuffer.size();
        for (const auto& pcmd : cmds->CmdBuffer) {
            scissor_rects[ScissorRectKey(fbheight, pcmd.ClipRect,
                                         pcmd.TextureId)] = nullptr;
        }
    }
    auto rbuilder = RenderableManager::Builder(num_prims);
    rbuilder.boundingBox({{0, 0, 0}, {10000, 10000, 10000}}).culling(false);

    // Push each unique scissor rectangle to a MaterialInstance.
    impl_->uiblit_pool_.reset();
    impl_->image_pool_.reset();
    TextureSampler sampler(TextureSampler::MinFilter::LINEAR,
                           TextureSampler::MagFilter::LINEAR);
    for (auto& pair : scissor_rects) {
        if (pair.first.id_ == 0) {
            pair.second = impl_->uiblit_pool_.pull();
            // Don't need to set texture, since we set the font texture
            // as the default when we created this material.
        } else {
            pair.second = impl_->image_pool_.pull();
            auto tex_id_long = reinterpret_cast<uintptr_t>(pair.first.id_);
            auto tex_id = std::uint16_t(tex_id_long);
            auto tex_handle = visualization::rendering::TextureHandle(tex_id);
            auto tex = visualization::rendering::EngineInstance::
                               GetResourceManager()
                                       .GetTexture(tex_handle);
            auto tex_sh_ptr = tex.lock();
            pair.second->setParameter(kImageTexParamName, tex_sh_ptr.get(),
                                      sampler);
        }
        pair.second->setScissor(pair.first.left_, pair.first.bottom_,
                                pair.first.width_, pair.first.height_);
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
                auto skey =
                        ScissorRectKey(fbheight, pcmd.ClipRect, pcmd.TextureId);
                auto miter = scissor_rects.find(skey);
                if (miter != scissor_rects.end()) {
                    rbuilder.geometry(
                                    prim_index,
                                    RenderableManager::PrimitiveType::TRIANGLES,
                                    impl_->vertex_buffers_[buffer_index],
                                    impl_->index_buffers_[buffer_index],
                                    indexOffset, pcmd.ElemCount)
                            .blendOrder(prim_index, prim_index)
                            .material(prim_index, miter->second);
                    prim_index++;
                } else {
                    utility::LogError("Internal error: material not found.");
                }
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
    camera->SetProjection(visualization::rendering::Camera::Projection::Ortho,
                          0.0, size.width, size.height, 0.0, 0.0, 1.0);
}

void ImguiFilamentBridge::CreateVertexBuffer(size_t buffer_index,
                                             size_t capacity) {
    SyncThreads();

    auto& engine = visualization::rendering::EngineInstance::GetInstance();

    engine.destroy(impl_->vertex_buffers_[buffer_index]);
    impl_->vertex_buffers_[buffer_index] =
            VertexBuffer::Builder()
                    .vertexCount(std::uint32_t(capacity))
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

    auto& engine = visualization::rendering::EngineInstance::GetInstance();

    engine.destroy(impl_->index_buffers_[buffer_index]);
    impl_->index_buffers_[buffer_index] =
            IndexBuffer::Builder()
                    .indexCount(std::uint32_t(capacity))
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
    auto& engine = visualization::rendering::EngineInstance::GetInstance();

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
        auto& engine = visualization::rendering::EngineInstance::GetInstance();

        // This is called only when ImGui needs to grow a vertex buffer, which
        // occurs a few times after launching and rarely (if ever) after that.
        Fence::waitAndDestroy(engine.createFence());
        impl_->has_synced_ = true;
    }
#endif
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
