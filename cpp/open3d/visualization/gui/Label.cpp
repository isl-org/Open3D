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

#include "open3d/visualization/gui/Label.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

// If word-wrapping is enabled, there isn't a preferred size, per se.
// But we don't want to make the preferred width too long, or it gets hard to
// read. Somewhere between 60 and 90 characters is the max. Proportional
// width fonts have approximately 2.5 characters per em. So the width should
// be something like 24 - 36 em.
// See https://practicaltypography.com/line-length.html
// and https://pearsonified.com/characters-per-line/
static const int PREFERRED_WRAP_WIDTH_EM = 35;

static const Color DEFAULT_COLOR(0, 0, 0, 0);

struct Label::Impl {
    std::string text_;
    Color color_ = DEFAULT_COLOR;
    bool is_single_line = true;
};

Label::Label(const char* text /*= nullptr*/) : impl_(new Label::Impl()) {
    if (text) {
        SetText(text);
    }
}

Label::~Label() {}

const char* Label::GetText() const { return impl_->text_.c_str(); }

void Label::SetText(const char* text) {
    impl_->text_ = text;
    impl_->is_single_line = !(impl_->text_.find('\n') != std::string::npos);
}

Color Label::GetTextColor() const { return impl_->color_; }

void Label::SetTextColor(const Color& color) { impl_->color_ = color; }

Size Label::CalcPreferredSize(const Theme& theme) const {
    auto em = theme.font_size;
    auto padding = ImGui::GetStyle().FramePadding;
    auto* font = ImGui::GetFont();

    if (impl_->is_single_line) {
        auto size = font->CalcTextSizeA(float(theme.font_size), 10000, 0.0,
                                        impl_->text_.c_str());
        return Size(int(std::ceil(size.x + 2.0f * padding.x)),
                    int(std::ceil(em + 2.0f * padding.y)));
    } else {
        ImVec2 size(0, 0);
        size_t line_start = 0;
        auto line_end = impl_->text_.find('\n');
        float wrap_width = float(PREFERRED_WRAP_WIDTH_EM * em);
        float spacing = ImGui::GetTextLineHeightWithSpacing() -
                        ImGui::GetTextLineHeight();
        do {
            ImVec2 sz;
            if (line_end == std::string::npos) {
                sz = font->CalcTextSizeA(float(theme.font_size), FLT_MAX,
                                         wrap_width,
                                         impl_->text_.c_str() + line_start);
                line_start = line_end;
            } else {
                sz = font->CalcTextSizeA(float(theme.font_size), FLT_MAX,
                                         wrap_width,
                                         impl_->text_.c_str() + line_start,
                                         impl_->text_.c_str() + line_end);
                line_start = line_end + 1;
                line_end = impl_->text_.find('\n', line_start);
            }
            size.x = std::max(size.x, sz.x);
            size.y += sz.y + spacing;
        } while (line_start != std::string::npos);

        return Size(int(std::ceil(size.x)) + int(std::ceil(2.0f * padding.x)),
                    int(std::ceil(size.y - spacing)) +
                            int(std::ceil(2.0f * padding.y)));
    }
}

Widget::DrawResult Label::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(ImVec2(float(frame.x), float(frame.y)));
    ImGui::PushItemWidth(float(frame.width));
    bool is_default_color = (impl_->color_ == DEFAULT_COLOR);
    if (!is_default_color) {
        ImGui::PushStyleColor(ImGuiCol_Text, colorToImgui(impl_->color_));
    }
    if (impl_->is_single_line) {
        ImGui::TextUnformatted(impl_->text_.c_str());
    } else {
        auto padding = ImGui::GetStyle().FramePadding;
        float wrapX = ImGui::GetCursorPos().x + frame.width -
                      std::ceil(2.0f * padding.x);
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextWrapped("%s", impl_->text_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!is_default_color) {
        ImGui::PopStyleColor();
    }
    ImGui::PopItemWidth();
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
