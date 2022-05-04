// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

#include <utils/Entity.h>

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
// 4305: LightManager.h needs to specify some constants as floats
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293 4305)
#endif  // _MSC_VER

#include <backend/PixelBufferDescriptor.h>
#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentCamera.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"
#include "open3d/visualization/rendering/filament/FilamentRenderToBuffer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

namespace open3d {
namespace visualization {
namespace rendering {

FilamentRenderer::FilamentRenderer(filament::Engine& engine,
                                   void* native_drawable,
                                   FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    swap_chain_ = engine_.createSwapChain(native_drawable,
                                          filament::SwapChain::CONFIG_READABLE);
    renderer_ = engine_.createRenderer();

    materials_modifier_ = std::make_unique<FilamentMaterialModifier>();
}

FilamentRenderer::FilamentRenderer(filament::Engine& engine,
                                   int width,
                                   int height,
                                   FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    swap_chain_ = engine_.createSwapChain(width, height,
                                          filament::SwapChain::CONFIG_READABLE);
    renderer_ = engine_.createRenderer();

    materials_modifier_ = std::make_unique<FilamentMaterialModifier>();
}

FilamentRenderer::~FilamentRenderer() {
    scenes_.clear();

    engine_.destroy(renderer_);
    engine_.destroy(swap_chain_);
}

SceneHandle FilamentRenderer::CreateScene() {
    auto handle = SceneHandle::Next();
    scenes_[handle] =
            std::make_unique<FilamentScene>(engine_, resource_mgr_, *this);

    return handle;
}

Scene* FilamentRenderer::GetScene(const SceneHandle& id) const {
    auto found = scenes_.find(id);
    if (found != scenes_.end()) {
        return found->second.get();
    }

    return nullptr;
}

void FilamentRenderer::DestroyScene(const SceneHandle& id) {
    scenes_.erase(id);
}

void FilamentRenderer::SetClearColor(const Eigen::Vector4f& color) {
    filament::Renderer::ClearOptions co;
    co.clearColor.r = color.x();
    co.clearColor.g = color.y();
    co.clearColor.b = color.z();
    co.clearColor.a = color.w();
    co.clear = false;
    co.discard = true;
    renderer_->setClearOptions(co);
}

void FilamentRenderer::SetOnAfterDraw(std::function<void()> callback) {
    on_after_draw_ = callback;
}

void FilamentRenderer::UpdateSwapChain() {
    void* native_win = swap_chain_->getNativeWindow();
    engine_.destroy(swap_chain_);
    swap_chain_ = engine_.createSwapChain(native_win);
}

void FilamentRenderer::UpdateBitmapSwapChain(int width, int height) {
    engine_.destroy(swap_chain_);
    swap_chain_ = engine_.createSwapChain(width, height,
                                          filament::SwapChain::CONFIG_READABLE);
}

void FilamentRenderer::BeginFrame() {
    // We will complete render to buffer requests first
    if (!buffer_renderers_.empty()) {
        for (auto& br : buffer_renderers_) {
            if (br->pending_) {
                br->Render();
            }
        }

        // Force the engine to render, otherwise it sometimes doesn't
        // render for a while, especially on Linux. This means the read
        // pixels callback does not get called until sometime later,
        // possibly several draws later.
        engine_.flushAndWait();

        buffer_renderers_.clear();  // Cleanup
    }

    frame_started_ = renderer_->beginFrame(swap_chain_);
}

void FilamentRenderer::Draw() {
    if (frame_started_) {
        // Draw 3D scenes into textures
        for (const auto& pair : scenes_) {
            pair.second->Draw(*renderer_);
        }

        // Draw the UI. This should come after the 3D scene(s), as SceneWidget
        // will draw the textures as an image, and this way we will have the
        // current frame's content from above.
        if (gui_scene_) {
            gui_scene_->Draw(*renderer_);
        }

        if (on_after_draw_) {
            on_after_draw_();
        }
    }
}

void FilamentRenderer::EndFrame() {
    if (frame_started_) {
        renderer_->endFrame();
        if (needs_wait_after_draw_) {
            engine_.flushAndWait();
            needs_wait_after_draw_ = false;
        }
    }
}

namespace {

struct UserData {
    std::function<void(std::shared_ptr<core::Tensor>)> callback;
    std::shared_ptr<core::Tensor> image;

    UserData(std::function<void(std::shared_ptr<core::Tensor>)> cb,
             std::shared_ptr<core::Tensor> img)
        : callback(cb), image(img) {}
};

void ReadPixelsCallback(void*, size_t, void* user) {
    auto* user_data = static_cast<UserData*>(user);
    user_data->callback(user_data->image);
    delete user_data;
}

}  // namespace

void FilamentRenderer::RequestReadPixels(
        int width,
        int height,
        std::function<void(std::shared_ptr<core::Tensor>)> callback) {
    core::SizeVector shape{height, width, 3};
    core::Dtype dtype = core::UInt8;
    int64_t nbytes = shape.NumElements() * dtype.ByteSize();

    auto image = std::make_shared<core::Tensor>(shape, dtype);
    auto* user_data = new UserData(callback, image);

    using namespace filament;
    using namespace backend;

    PixelBufferDescriptor pd(image->GetDataPtr(), nbytes, PixelDataFormat::RGB,
                             PixelDataType::UBYTE, ReadPixelsCallback,
                             user_data);
    renderer_->readPixels(0, 0, width, height, std::move(pd));
    needs_wait_after_draw_ = true;
}

MaterialHandle FilamentRenderer::AddMaterial(
        const ResourceLoadRequest& request) {
    return resource_mgr_.CreateMaterial(request);
}

MaterialInstanceHandle FilamentRenderer::AddMaterialInstance(
        const MaterialHandle& material) {
    return resource_mgr_.CreateMaterialInstance(material);
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id) {
    materials_modifier_->Reset();

    auto instance_id = resource_mgr_.CreateMaterialInstance(id);

    if (instance_id) {
        auto w_material_instance =
                resource_mgr_.GetMaterialInstance(instance_id);
        materials_modifier_->Init(w_material_instance.lock(), instance_id);
    } else {
        utility::LogWarning(
                "Failed to create material instance for material handle {}.",
                id);
    }

    return *materials_modifier_;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(
        const MaterialInstanceHandle& id) {
    materials_modifier_->Reset();

    auto w_material_instance = resource_mgr_.GetMaterialInstance(id);
    if (!w_material_instance.expired()) {
        materials_modifier_->Init(w_material_instance.lock(), id);
    } else {
        utility::LogWarning(
                "Failed to modify material instance: unknown instance handle "
                "{}.",
                id);
    }

    return *materials_modifier_;
}

void FilamentRenderer::RemoveMaterialInstance(
        const MaterialInstanceHandle& id) {
    resource_mgr_.Destroy(id);
}

TextureHandle FilamentRenderer::AddTexture(const ResourceLoadRequest& request,
                                           bool srgb) {
    if (request.path_.empty()) {
        request.error_callback_(request, -1,
                                "Texture can be loaded only from file");
        return {};
    }

    return resource_mgr_.CreateTexture(request.path_.data(), srgb);
}

bool FilamentRenderer::UpdateTexture(
        TextureHandle texture,
        const std::shared_ptr<geometry::Image> image,
        bool srgb) {
    return resource_mgr_.UpdateTexture(texture, image, srgb);
}

bool FilamentRenderer::UpdateTexture(TextureHandle texture,
                                     const t::geometry::Image& image,
                                     bool srgb) {
    return resource_mgr_.UpdateTexture(texture, image, srgb);
}

void FilamentRenderer::RemoveTexture(const TextureHandle& id) {
    resource_mgr_.Destroy(id);
}

IndirectLightHandle FilamentRenderer::AddIndirectLight(
        const ResourceLoadRequest& request) {
    if (request.path_.empty()) {
        request.error_callback_(
                request, -1, "Indirect lights can be loaded only from files");
        return {};
    }

    return resource_mgr_.CreateIndirectLight(request);
}

void FilamentRenderer::RemoveIndirectLight(const IndirectLightHandle& id) {
    resource_mgr_.Destroy(id);
}

SkyboxHandle FilamentRenderer::AddSkybox(const ResourceLoadRequest& request) {
    if (request.path_.empty()) {
        request.error_callback_(request, -1,
                                "Skyboxes can be loaded only from files");
        return {};
    }

    return resource_mgr_.CreateSkybox(request);
}

void FilamentRenderer::RemoveSkybox(const SkyboxHandle& id) {
    resource_mgr_.Destroy(id);
}

std::shared_ptr<RenderToBuffer> FilamentRenderer::CreateBufferRenderer() {
    auto renderer = std::make_shared<FilamentRenderToBuffer>(engine_);
    buffer_renderers_.insert(renderer);
    return renderer;
}

void FilamentRenderer::ConvertToGuiScene(const SceneHandle& id) {
    auto found = scenes_.find(id);
    // TODO: assert(found != scenes_.end())
    if (found != scenes_.end()) {
        if (gui_scene_ != nullptr) {
            utility::LogWarning(
                    "FilamentRenderer::ConvertToGuiScene: guiScene_ is already "
                    "set");
        }
        gui_scene_ = std::move(found->second);
        scenes_.erase(found);
    }
}

TextureHandle FilamentRenderer::AddTexture(
        const std::shared_ptr<geometry::Image> image, bool srgb) {
    return resource_mgr_.CreateTexture(image, srgb);
}

TextureHandle FilamentRenderer::AddTexture(const t::geometry::Image& image,
                                           bool srgb) {
    return resource_mgr_.CreateTexture(image, srgb);
}

// void FilamentRenderer::OnBufferRenderDestroyed(FilamentRenderToBuffer*
// render) {
//    buffer_renderers_.erase(render);
//}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
