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

#include "Label.h"

#include "Theme.h"

#include <imgui.h>

#include <cmath>
#include <string>

namespace open3d {
namespace gui {

struct Label::Impl {
    std::string text;
};

Label::Label(const char* text /*= nullptr*/) : impl_(new Label::Impl()) {
    if (text) {
        impl_->text = text;
    }
}

Label::~Label() {}

const char* Label::GetText() const { return impl_->text.c_str(); }

Size Label::CalcPreferredSize(const Theme& theme) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    auto size = ImGui::GetFont()->CalcTextSizeA(theme.fontSize, 10000, 10000,
                                                impl_->text.c_str());
    return Size(std::ceil(size.x + 2.0f * padding.x),
                std::ceil(em + 2.0f * padding.y));
}

Widget::DrawResult Label::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));
    ImGui::Text("%s", impl_->text.c_str());
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace open3d
