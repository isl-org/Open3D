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

#include "Open3D/GUI/ImageLabel.h"

#include <filament/Texture.h>
#include <imgui.h>
#include <string>

#include "Open3D/GUI/Theme.h"
#include "Open3D/GUI/Util.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h"
#include "Open3D/Visualization/Rendering/Renderer.h"

namespace open3d {
namespace gui {

struct ImageLabel::Impl {
    std::string image_path_;
    ImageLabel::Scaling scaling_ = ImageLabel::Scaling::ASPECT;
    std::shared_ptr<geometry::Image> image_data_;  // temporary storage
    ImVec2 image_size_;
    ImVec2 uv0_;
    ImVec2 uv1_;
    visualization::Renderer* renderer_;  // nullptr if texture_ isn't ours
    visualization::TextureHandle texture_;
};

ImageLabel::ImageLabel(const char* image_path) : impl_(new ImageLabel::Impl()) {
    impl_->image_path_ = image_path;
    impl_->image_data_ = io::CreateImageFromFile(image_path);
    if (impl_->image_data_ && impl_->image_data_->width_ == 0 &&
        impl_->image_data_->height_ == 0) {
        impl_->image_data_.reset();
    } else {
        impl_->image_size_ = ImVec2(float(impl_->image_data_->width_),
                                    float(impl_->image_data_->height_));
    }
    impl_->uv0_ = ImVec2(0.0f, 0.0f);
    impl_->uv1_ = ImVec2(1.0f, 1.0f);
    impl_->renderer_ = nullptr;
}

ImageLabel::ImageLabel(visualization::TextureHandle texture_id,
                       float u0 /*= 0.0f*/,
                       float v0 /*= 0.0f*/,
                       float u1 /*= 1.0f*/,
                       float v1 /*= 1.0f*/)
    : impl_(new ImageLabel::Impl()) {
    auto& resource_manager =
            visualization::EngineInstance::GetResourceManager();
    auto tex_weak = resource_manager.GetTexture(texture_id);
    auto tex_sh = tex_weak.lock();
    if (tex_sh) {
        float uvw = u1 - u0;
        float uvh = v1 - v0;
        impl_->image_size_ = ImVec2(uvw * float(tex_sh->getWidth()),
                                    uvh* float(tex_sh->getHeight()));
    }
    impl_->uv0_ = ImVec2(u0, v0);
    impl_->uv1_ = ImVec2(u1, v1);
    impl_->renderer_ = nullptr;
    impl_->texture_ = texture_id;
}

ImageLabel::~ImageLabel() {
    if (impl_->renderer_) {
        impl_->renderer_->RemoveTexture(impl_->texture_);
    }
}

void ImageLabel::SetScaling(Scaling scaling) { impl_->scaling_ = scaling; }

ImageLabel::Scaling ImageLabel::GetScaling() const { return impl_->scaling_; }

Size ImageLabel::CalcPreferredSize(const Theme& theme) const {
    if (impl_->image_size_.x != 0.0f && impl_->image_size_.y != 0.0f) {
        return Size(impl_->image_size_.x, impl_->image_size_.y);
    } else {
        return Size(5 * theme.font_size, 5 * theme.font_size);
    }
}

void ImageLabel::Layout(const Theme& theme) { Super::Layout(theme); }

Widget::DrawResult ImageLabel::Draw(const DrawContext& context) {
    auto& frame = GetFrame();

    if (impl_->image_data_ &&
        impl_->texture_ == visualization::TextureHandle::kBad) {
        impl_->texture_ = context.renderer.AddTexture(impl_->image_data_);
        if (impl_->texture_ != visualization::TextureHandle::kBad) {
            impl_->renderer_ = &context.renderer;
        } else {
            impl_->texture_ = visualization::TextureHandle();
        }
        impl_->image_data_.reset();
    }

    float width_px = impl_->image_size_.x;
    float height_px = impl_->image_size_.y;
    if (impl_->texture_ != visualization::TextureHandle::kBad) {
        ImVec2 size, uv0, uv1;
        switch (impl_->scaling_) {
            case Scaling::NONE: {
                float w = std::min(float(frame.width), width_px);
                float h = std::min(float(frame.height), height_px);
                size = ImVec2(w, h);
                uv0 = impl_->uv0_;
                uv1 = ImVec2(
                        uv0.x + (impl_->uv1_.x - impl_->uv0_.x) * w / width_px,
                        uv0.y + (impl_->uv1_.y - impl_->uv1_.y) * h /
                                        height_px);
                break;
            }
            case Scaling::ANY:
                size = ImVec2(frame.width, frame.height);
                uv0 = impl_->uv0_;
                uv1 = impl_->uv1_;
                break;
            case Scaling::ASPECT: {
                float aspect = width_px / height_px;
                float w_at_height = float(frame.height) * aspect;
                float h_at_width = float(frame.width) / aspect;
                if (w_at_height <= frame.width) {
                    size = ImVec2(w_at_height, float(frame.height));
                } else {
                    size = ImVec2(float(frame.width), h_at_width);
                }
                uv0 = impl_->uv0_;
                uv1 = impl_->uv1_;
                break;
            }
        }
        float x = std::max(0.0f, (float(frame.width) - size.x) / 2.0f);
        float y = std::max(0.0f, (float(frame.height) - size.y) / 2.0f);
        ImVec2 pos(frame.x + x - context.uiOffsetX,
                   frame.y + y - context.uiOffsetY);
        ImTextureID image_id =
                reinterpret_cast<ImTextureID>(impl_->texture_.GetId());
        ImGui::SetCursorPos(pos);
        ImGui::Image(image_id, size, uv0, uv1);
    } else {
        // Draw error message if we don't have an image, instead of
        // quietly failing or something.
        const char* error_text = "  Error\nloading\n image";
        Color fg(1.0, 1.0, 1.0);
        ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(frame.x, frame.y),
                ImVec2(frame.GetRight(), frame.GetBottom()),
                IM_COL32(255, 0, 0, 255));
        ImGui::GetWindowDrawList()->AddRect(
                ImVec2(frame.x, frame.y),
                ImVec2(frame.GetRight(), frame.GetBottom()),
                IM_COL32(255, 255, 255, 255));
        ImGui::PushStyleColor(ImGuiCol_Text, util::colorToImgui(fg));

        auto padding = ImGui::GetStyle().FramePadding;
        float wrap_width = frame.width - std::ceil(2.0f * padding.x);
        float wrapX = ImGui::GetCursorPos().x + wrap_width;
        auto* font = ImGui::GetFont();
        auto text_size = font->CalcTextSizeA(
                context.theme.font_size, wrap_width, wrap_width, error_text);
        float x = (float(frame.width) - text_size.x) / 2.0f;
        float y = (float(frame.height) - text_size.y) / 2.0f;

        ImGui::SetCursorPos(ImVec2(x + frame.x - context.uiOffsetX,
                                   y + frame.y - context.uiOffsetY));
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextWrapped("%s", error_text);
        ImGui::PopTextWrapPos();

        ImGui::PopStyleColor();
    }

    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace open3d
