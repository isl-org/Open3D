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

#include "open3d/visualization/gui/RadioButton.h"

#include <imgui.h>

#include <cmath>
#include <string>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static const int NO_SELECTION = -1;
static int g_next_radiobtn_id = 1;
}  // namespace

struct RadioButton::Impl {
    std::string id_;
    RadioButton::Type type_;
    int selected_index_ = -1;
    std::vector<std::string> items_;
    std::function<void(int)> on_selection_changed_;
};

RadioButton::RadioButton(Type type) : impl_(new RadioButton::Impl()) {
    impl_->type_ = type;
    impl_->id_ = "##radiobtn_" + std::to_string(g_next_radiobtn_id++);
}

RadioButton::~RadioButton() = default;

void RadioButton::SetItems(const std::vector<std::string>& items) {
    impl_->items_ = items;
    impl_->selected_index_ = items.empty() ? NO_SELECTION : 0;
}

int RadioButton::GetSelectedIndex() const { return impl_->selected_index_; }

const char* RadioButton::GetSelectedValue() const {
    if (impl_->selected_index_ < 0 ||
        impl_->selected_index_ >= int(impl_->items_.size())) {
        return "";
    } else {
        return impl_->items_[impl_->selected_index_].c_str();
    }
}

void RadioButton::SetSelectedIndex(int index) {
    if (index >= 0) {
        impl_->selected_index_ = std::min(int(impl_->items_.size() - 1), index);
    }
}

void RadioButton::SetOnSelectionChanged(std::function<void(int)> callback) {
    impl_->on_selection_changed_ = callback;
}

Size RadioButton::CalcPreferredSize(const LayoutContext& context,
                                    const Constraints& constraints) const {
    auto fh = ImGui::GetFrameHeight();
    auto& st = ImGui::GetStyle();
    auto spacing = st.ItemSpacing.x + st.ItemInnerSpacing.x;
    ImVec2 size(0, 0);
    auto* font = ImGui::GetFont();
    for (auto& item : impl_->items_) {
        auto item_size = font->CalcTextSizeA(float(context.theme.font_size),
                                             float(constraints.width), 0.0,
                                             item.c_str());
        if (impl_->type_ == Type::VERT) {
            size.x = std::max(size.x, item_size.x);
            size.y += fh;
        } else {
            size.x += fh + item_size.x + spacing;
            size.y = fh;
        }
    }
    if (impl_->type_ == Type::VERT) {
        size.x += fh + st.ItemInnerSpacing.x;  // box + spacing to text
    }
    return Size{int(std::ceil(size.x + 2.0f * st.FramePadding.x)),
                int(std::ceil(size.y + 2.0f * st.FramePadding.y))};
}

Widget::DrawResult RadioButton::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));
    auto result = Widget::DrawResult::NONE;

    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));

    auto selected_idx = impl_->selected_index_;
    for (size_t i = 0; i < impl_->items_.size(); ++i) {
        if (impl_->type_ == Type::HORIZ && i > 0) {
            ImGui::SameLine();
        } else {
            auto pos = ImGui::GetCursorScreenPos();
            pos.x = float(frame.x);
            ImGui::SetCursorScreenPos(pos);
        }

        bool is_selected = (int(i) == impl_->selected_index_);
        if (is_selected) {
            ImGui::PushStyleColor(
                    ImGuiCol_FrameBg,
                    colorToImgui(context.theme.radiobtn_background_on_color));
            ImGui::PushStyleColor(
                    ImGuiCol_FrameBgHovered,
                    colorToImgui(
                            context.theme.radiobtn_background_hover_on_color));
        } else {
            ImGui::PushStyleColor(
                    ImGuiCol_FrameBg,
                    colorToImgui(context.theme.radiobtn_background_off_color));
            ImGui::PushStyleColor(
                    ImGuiCol_FrameBgHovered,
                    colorToImgui(
                            context.theme.radiobtn_background_hover_off_color));
        }
        ImGui::RadioButton(impl_->items_[i].c_str(), &selected_idx, i);
        ImGui::PopStyleColor(2);
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    if (selected_idx != impl_->selected_index_) {
        impl_->selected_index_ = selected_idx;
        if (impl_->on_selection_changed_) {
            impl_->on_selection_changed_(impl_->selected_index_);
        }
        result = Widget::DrawResult::REDRAW;
    }
    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
