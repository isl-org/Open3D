// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "ColorEdit.h"

#include "Theme.h"

#include <imgui.h>

#include <cmath>
#include <sstream>

namespace open3d {
namespace gui {

namespace {
static int gNextColorEditId = 1;
}

struct ColorEdit::Impl {
    std::string id;
    Color value;
    std::function<void(const Color&)> onValueChanged;
};

ColorEdit::ColorEdit() : impl_(new ColorEdit::Impl()) {
    std::stringstream s;
    s << "##colorEdit_" << gNextColorEditId++;
    impl_->id = s.str();
}

ColorEdit::~ColorEdit() {}

void ColorEdit::SetValue(const Color& color) { impl_->value = color; }

void ColorEdit::SetValue(const float r, const float g, const float b) {
    impl_->value.SetColor(r, g, b);
}

const Color& ColorEdit::GetValue() const { return impl_->value; }

void ColorEdit::SetOnValueChanged(
        std::function<void(const Color&)> onValueChanged) {
    impl_->onValueChanged = onValueChanged;
}

Size ColorEdit::CalcPreferredSize(const Theme& theme) const {
    auto lineHeight = ImGui::GetTextLineHeight();
    auto height = lineHeight + 2.0 * ImGui::GetStyle().FramePadding.y;

    return Size(Widget::DIM_GROW, std::ceil(height));
}

ColorEdit::DrawResult ColorEdit::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));

    auto newValue = impl_->value;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(GetFrame().width);
    ImGui::ColorEdit3(impl_->id.c_str(), newValue.GetMutablePointer());
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    if (impl_->value != newValue) {
        impl_->value = newValue;
        if (impl_->onValueChanged) {
            impl_->onValueChanged(newValue);
        }

        return Widget::DrawResult::REDRAW;
    }
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace open3d
