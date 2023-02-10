// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Label3D.h"

#include <string>

namespace open3d {
namespace visualization {
namespace gui {

static const Color DEFAULT_COLOR(0, 0, 0, 1);

struct Label3D::Impl {
    std::string text_;
    Eigen::Vector3f position_;
    Color color_ = DEFAULT_COLOR;
    float scale_ = 1.f;
};

Label3D::Label3D(const Eigen::Vector3f& pos, const char* text /*= nullptr*/)
    : impl_(new Label3D::Impl()) {
    SetPosition(pos);
    if (text) {
        SetText(text);
    }
}

Label3D::~Label3D() {}

const char* Label3D::GetText() const { return impl_->text_.c_str(); }

void Label3D::SetText(const char* text) { impl_->text_ = text; }

Eigen::Vector3f Label3D::GetPosition() const { return impl_->position_; }

void Label3D::SetPosition(const Eigen::Vector3f& pos) {
    impl_->position_ = pos;
}

Color Label3D::GetTextColor() const { return impl_->color_; }

void Label3D::SetTextColor(const Color& color) { impl_->color_ = color; }

float Label3D::GetTextScale() const { return impl_->scale_; }

void Label3D::SetTextScale(float scale) { impl_->scale_ = scale; }

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
