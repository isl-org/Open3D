// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Label.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "open3d/visualization/gui/Application.h"
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
    FontId font_id_ = Application::DEFAULT_FONT_ID;
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

FontId Label::GetFontId() const { return impl_->font_id_; }

void Label::SetFontId(const FontId font_id) { impl_->font_id_ = font_id; }

Size Label::CalcPreferredSize(const LayoutContext& context,
                              const Constraints& constraints) const {
    ImGui::PushFont((ImFont*)context.fonts.GetFont(impl_->font_id_));

    auto padding = ImGui::GetStyle().FramePadding;
    auto* font = ImGui::GetFont();
    Size pref;

    if (impl_->is_single_line) {
        float wrap_width = float(constraints.width);
        auto size =
                font->CalcTextSizeA(font->FontSize, float(constraints.width),
                                    wrap_width, impl_->text_.c_str());
        pref = Size(int(std::ceil(size.x + 2.0f * padding.x)),
                    int(std::ceil(size.y + 2.0f * padding.y)));
    } else {
        ImVec2 size(0, 0);
        size_t line_start = 0;
        auto line_end = impl_->text_.find('\n');
        auto em = int(std::round(font->FontSize));
        float wrap_width = float(
                std::min(constraints.width, PREFERRED_WRAP_WIDTH_EM * em));
        float spacing = ImGui::GetTextLineHeightWithSpacing() -
                        ImGui::GetTextLineHeight();
        do {
            ImVec2 sz;
            if (line_end == std::string::npos) {
                sz = font->CalcTextSizeA(font->FontSize, FLT_MAX, wrap_width,
                                         impl_->text_.c_str() + line_start);
                line_start = line_end;
            } else {
                sz = font->CalcTextSizeA(font->FontSize, FLT_MAX, wrap_width,
                                         impl_->text_.c_str() + line_start,
                                         impl_->text_.c_str() + line_end);
                line_start = line_end + 1;
                line_end = impl_->text_.find('\n', line_start);
            }
            size.x = std::max(size.x, sz.x);
            size.y += sz.y + spacing;
        } while (line_start != std::string::npos);

        pref = Size(int(std::ceil(size.x)) + int(std::ceil(2.0f * padding.x)),
                    int(std::ceil(size.y - spacing)) +
                            int(std::ceil(2.0f * padding.y)));
    }

    ImGui::PopFont();
    return pref;
}

Widget::DrawResult Label::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));
    ImGui::PushItemWidth(float(frame.width));
    bool is_default_color = (impl_->color_ == DEFAULT_COLOR);
    if (!is_default_color) {
        ImGui::PushStyleColor(ImGuiCol_Text, colorToImgui(impl_->color_));
    }
    ImGui::PushFont((ImFont*)context.fonts.GetFont(impl_->font_id_));

    auto padding = ImGui::GetStyle().FramePadding;
    float wrapX = ImGui::GetCursorPos().x + frame.width - padding.x;
    ImGui::PushTextWrapPos(wrapX);
    ImGui::TextWrapped("%s", impl_->text_.c_str());
    ImGui::PopTextWrapPos();

    ImGui::PopFont();
    if (!is_default_color) {
        ImGui::PopStyleColor();
    }
    ImGui::PopItemWidth();
    // Tooltip (if it exists) is in the system font, so do after popping font
    DrawImGuiTooltip();
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
