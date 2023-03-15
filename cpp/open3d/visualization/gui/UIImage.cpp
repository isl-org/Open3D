// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/UIImage.h"

// 4293:  Filament's utils/algorithm.h utils::details::clz() does strange
//        things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//        32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//        determine the if statement does not run.)
// 4146: PixelBufferDescriptor assert unsigned is positive before subtracting
//       but MSVC can't figure that out.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4293 4146)
#endif  // _MSC_VER

#include <filament/Texture.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <string>

#include "open3d/geometry/Image.h"
#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

namespace open3d {
namespace visualization {
namespace gui {

struct UIImage::Impl {
    std::string image_path_;
    UIImage::Scaling scaling_ = UIImage::Scaling::ASPECT;
    std::shared_ptr<geometry::Image> image_data_;      // temporary storage
    std::shared_ptr<t::geometry::Image> timage_data_;  // temporary storage
    float image_width_;
    float image_height_;
    float u0_;
    float v0_;
    float u1_;
    float v1_;
    visualization::rendering::Renderer*
            renderer_;  // nullptr if texture_ isn't ours
    visualization::rendering::TextureHandle texture_;
};

UIImage::UIImage(const char* image_path) : impl_(new UIImage::Impl()) {
    impl_->image_path_ = image_path;
    impl_->image_data_ = io::CreateImageFromFile(image_path);
    if (impl_->image_data_) {
        if (impl_->image_data_ && impl_->image_data_->width_ == 0 &&
            impl_->image_data_->height_ == 0) {
            impl_->image_data_.reset();
        } else {
            impl_->image_width_ = float(impl_->image_data_->width_);
            impl_->image_height_ = float(impl_->image_data_->height_);
        }
    }
    impl_->u0_ = 0.0f;
    impl_->v0_ = 0.0f;
    impl_->u1_ = 1.0f;
    impl_->v1_ = 1.0f;
    impl_->renderer_ = nullptr;
}

UIImage::UIImage(std::shared_ptr<geometry::Image> image)
    : impl_(new UIImage::Impl()) {
    impl_->image_data_ = image;
    if (impl_->image_data_) {
        if (impl_->image_data_ && impl_->image_data_->width_ == 0 &&
            impl_->image_data_->height_ == 0) {
            impl_->image_data_.reset();
        } else {
            impl_->image_width_ = float(impl_->image_data_->width_);
            impl_->image_height_ = float(impl_->image_data_->height_);
        }
    }
    impl_->u0_ = 0.0f;
    impl_->v0_ = 0.0f;
    impl_->u1_ = 1.0f;
    impl_->v1_ = 1.0f;
    impl_->renderer_ = nullptr;
}

UIImage::UIImage(std::shared_ptr<t::geometry::Image> image)
    : impl_(new UIImage::Impl()) {
    impl_->timage_data_ = image;
    if (impl_->timage_data_->IsEmpty()) {
        impl_->timage_data_.reset();
    } else {
        impl_->image_width_ = float(impl_->timage_data_->GetCols());
        impl_->image_height_ = float(impl_->timage_data_->GetRows());
    }
    impl_->u0_ = 0.0f;
    impl_->v0_ = 0.0f;
    impl_->u1_ = 1.0f;
    impl_->v1_ = 1.0f;
    impl_->renderer_ = nullptr;
}

UIImage::UIImage(visualization::rendering::TextureHandle texture_id,
                 float u0 /*= 0.0f*/,
                 float v0 /*= 0.0f*/,
                 float u1 /*= 1.0f*/,
                 float v1 /*= 1.0f*/)
    : impl_(new UIImage::Impl()) {
    auto& resource_manager =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto tex_weak = resource_manager.GetTexture(texture_id);
    auto tex_sh = tex_weak.lock();
    if (tex_sh) {
        float uvw = u1 - u0;
        float uvh = v1 - v0;
        impl_->image_width_ = uvw * float(tex_sh->getWidth());
        impl_->image_height_ = uvh * float(tex_sh->getHeight());
    }
    impl_->u0_ = u0;
    impl_->v0_ = v0;
    impl_->u1_ = u1;
    impl_->v1_ = v1;
    impl_->renderer_ = nullptr;
    impl_->texture_ = texture_id;
}

UIImage::~UIImage() {
    if (impl_->renderer_) {
        impl_->renderer_->RemoveTexture(impl_->texture_);
    }
}

void UIImage::UpdateImage(std::shared_ptr<geometry::Image> image) {
    impl_->image_data_ = image;
    impl_->timage_data_.reset();
}

void UIImage::UpdateImage(std::shared_ptr<t::geometry::Image> image) {
    impl_->image_data_.reset();
    impl_->timage_data_ = image;
}

void UIImage::SetScaling(Scaling scaling) { impl_->scaling_ = scaling; }

UIImage::Scaling UIImage::GetScaling() const { return impl_->scaling_; }

Size UIImage::CalcPreferredSize(const LayoutContext& context,
                                const Widget::Constraints& constraints) const {
    if (impl_->image_width_ != 0.0f && impl_->image_height_ != 0.0f) {
        if (impl_->scaling_ == Scaling::ASPECT &&
            (constraints.width < impl_->image_width_ ||
             constraints.height < impl_->image_height_)) {
            float aspect = impl_->image_width_ / impl_->image_height_;
            float w_at_height = float(constraints.height) * aspect;
            float h_at_width = float(constraints.width) / aspect;
            if (w_at_height <= constraints.width) {
                return Size(int(std::round(w_at_height)), constraints.height);
            } else {
                return Size(constraints.width, int(std::round(h_at_width)));
            }
        } else {
            return Size(int(std::round(impl_->image_width_)),
                        int(std::round(impl_->image_height_)));
        }
    } else {
        return Size(0, 0);
    }
}

UIImage::DrawParams UIImage::CalcDrawParams(
        visualization::rendering::Renderer& renderer, const Rect& frame) const {
    DrawParams params;

    if (impl_->image_data_) {  // Legacy Image
        if (impl_->texture_ == visualization::rendering::TextureHandle::kBad) {
            impl_->texture_ = renderer.AddTexture(impl_->image_data_);
            if (impl_->texture_ !=
                visualization::rendering::TextureHandle::kBad) {
                impl_->renderer_ = &renderer;
                impl_->image_width_ = float(impl_->image_data_->width_);
                impl_->image_height_ = float(impl_->image_data_->height_);
                params.image_size_changed = true;
            } else {
                impl_->texture_ = visualization::rendering::TextureHandle();
            }
        } else {
            if (!renderer.UpdateTexture(impl_->texture_, impl_->image_data_,
                                        false)) {
                impl_->texture_ = renderer.AddTexture(impl_->image_data_);
                impl_->image_width_ = float(impl_->image_data_->width_);
                impl_->image_height_ = float(impl_->image_data_->height_);
                params.image_size_changed = true;
            }
        }
        impl_->image_data_.reset();
    } else if (impl_->timage_data_) {  // TGeoemetry Image
        if (impl_->texture_ == visualization::rendering::TextureHandle::kBad) {
            impl_->texture_ = renderer.AddTexture(*impl_->timage_data_.get());
            if (impl_->texture_ !=
                visualization::rendering::TextureHandle::kBad) {
                impl_->renderer_ = &renderer;
                impl_->image_width_ = float(impl_->timage_data_->GetCols());
                impl_->image_height_ = float(impl_->timage_data_->GetRows());
                params.image_size_changed = true;
            } else {
                impl_->texture_ = visualization::rendering::TextureHandle();
            }
        } else {
            if (!renderer.UpdateTexture(impl_->texture_, *impl_->timage_data_,
                                        false)) {
                impl_->texture_ =
                        renderer.AddTexture(*impl_->timage_data_.get());
                impl_->image_width_ = float(impl_->timage_data_->GetCols());
                impl_->image_height_ = float(impl_->timage_data_->GetRows());
                params.image_size_changed = true;
            }
        }
        impl_->timage_data_.reset();
    }

    params.texture = impl_->texture_;

    float width_px = impl_->image_width_;
    float height_px = impl_->image_height_;
    if (impl_->texture_ != visualization::rendering::TextureHandle::kBad) {
        switch (impl_->scaling_) {
            case Scaling::NONE: {
                float w = std::min(float(frame.width), width_px);
                float h = std::min(float(frame.height), height_px);
                params.width = w;
                params.height = h;
                params.u0 = impl_->u0_;
                params.v0 = impl_->v0_;
                params.u1 =
                        impl_->u0_ + (impl_->u1_ - impl_->u0_) * w / width_px;
                params.v1 =
                        impl_->v0_ + (impl_->v1_ - impl_->v0_) * h / height_px;
                break;
            }
            case Scaling::ANY:
                params.width = float(frame.width);
                params.height = float(frame.height);
                params.u0 = impl_->u0_;
                params.v0 = impl_->v0_;
                params.u1 = impl_->u1_;
                params.v1 = impl_->v1_;
                break;
            case Scaling::ASPECT: {
                float aspect = width_px / height_px;
                float w_at_height = float(frame.height) * aspect;
                float h_at_width = float(frame.width) / aspect;
                if (w_at_height <= frame.width) {
                    params.width = w_at_height;
                    params.height = float(frame.height);
                } else {
                    params.width = float(frame.width);
                    params.height = h_at_width;
                }
                params.u0 = impl_->u0_;
                params.v0 = impl_->v0_;
                params.u1 = impl_->u1_;
                params.v1 = impl_->v1_;
                break;
            }
        }
        float x = std::max(0.0f, (float(frame.width) - params.width) / 2.0f);
        float y = std::max(0.0f, (float(frame.height) - params.height) / 2.0f);
        params.pos_x = frame.x + x;
        params.pos_y = frame.y + y;
    } else {
        params.pos_x = float(frame.x);
        params.pos_y = float(frame.y);
        params.width = float(frame.width);
        params.height = float(frame.height);
        params.u0 = 0.0f;
        params.v0 = 0.0f;
        params.u1 = 1.0f;
        params.v1 = 1.0f;
    }

    return params;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
