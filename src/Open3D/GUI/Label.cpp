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

// If word-wrapping is enabled, there isn't a preferred size, per se.
// But we don't want to make the preferred width too long, or it gets hard to
// read. Somewhere between 60 and 90 characters is the max. Proportional
// width fonts have approximately 2.5 characters per em. So the width should
// be something like 24 - 36 em.
// See https://practicaltypography.com/line-length.html
// and https://pearsonified.com/characters-per-line/
static const int PREFERRED_WRAP_WIDTH_EM = 35;

struct Label::Impl {
    std::string text;
    bool isSingleLine = true;
};

Label::Label(const char *text /*= nullptr*/)
: impl_(new Label::Impl()) {
    if (text) {
        SetText(text);
    }
}

Label::~Label() {
}

const char* Label::GetText() const {
    return impl_->text.c_str();
}

void Label::SetText(const char *text) {
    impl_->text = text;
    impl_->isSingleLine = !(impl_->text.find('\n') != std::string::npos);
}

Size Label::CalcPreferredSize(const Theme& theme) const {
    auto em = theme.fontSize;
    auto padding = ImGui::GetStyle().FramePadding;
    auto *font = ImGui::GetFont();

    if (impl_->isSingleLine) {
        auto size = font->CalcTextSizeA(theme.fontSize, 10000, 0.0,
                                        impl_->text.c_str());
        return Size(std::ceil(size.x + 2.0f * padding.x),
                    std::ceil(em + 2.0f * padding.y));
    } else {
        ImVec2 size(0, 0);
        size_t lineStart = 0;
        auto lineEnd = impl_->text.find('\n');
        float wrapWidth = PREFERRED_WRAP_WIDTH_EM * em;
        float spacing = ImGui::GetTextLineHeightWithSpacing() -
                        ImGui::GetTextLineHeight();
        do {
            ImVec2 sz;
            if (lineEnd == std::string::npos) {
                sz = font->CalcTextSizeA(theme.fontSize, FLT_MAX, wrapWidth,
                                         impl_->text.c_str() + lineStart);
                lineStart = lineEnd;
            } else {
                sz = font->CalcTextSizeA(theme.fontSize, FLT_MAX, wrapWidth,
                                         impl_->text.c_str() + lineStart,
                                         impl_->text.c_str() + lineEnd);
                lineStart = lineEnd + 1;
                lineEnd = impl_->text.find('\n', lineStart);
            }
            size.x = std::max(size.x, sz.x);
            size.y += sz.y + spacing;
        } while (lineStart != std::string::npos);

        return Size(std::ceil(size.x) + std::ceil(2.0f * padding.x),
                    std::ceil(size.y - spacing) + std::ceil(2.0f * padding.y));
    }
}

Widget::DrawResult Label::Draw(const DrawContext& context) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                               frame.y - context.uiOffsetY));
    ImGui::PushItemWidth(frame.width);
    if (impl_->isSingleLine) {
        ImGui::TextUnformatted(impl_->text.c_str());
    } else {
        auto padding = ImGui::GetStyle().FramePadding;
        float wrapX = ImGui::GetCursorPos().x + frame.width - std::ceil(2.0f * padding.x);
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextWrapped("%s", impl_->text.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::PopItemWidth();
    return Widget::DrawResult::NONE;
}

} // gui
} // open3d
