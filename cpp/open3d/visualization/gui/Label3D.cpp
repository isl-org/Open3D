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
