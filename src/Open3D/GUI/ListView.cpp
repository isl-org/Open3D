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

#include "Open3D/GUI/ListView.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "Open3D/GUI/Theme.h"

namespace open3d {
namespace gui {

namespace {
static const int NO_SELECTION = -1;
static int g_next_list_box_id = 1;
}  // namespace

struct ListView::Impl {
    std::string imgui_id_;
    std::vector<std::string> items_;
    int selected_index_ = NO_SELECTION;
    std::function<void(const char *, bool)> on_value_changed_;
};

ListView::ListView() : impl_(new ListView::Impl()) {
    std::stringstream s;
    s << "##listview_" << g_next_list_box_id++;
    impl_->imgui_id_ = s.str();
}

ListView::~ListView() {}

void ListView::SetItems(const std::vector<std::string> &items) {
    impl_->items_ = items;
    impl_->selected_index_ = NO_SELECTION;
}

int ListView::GetSelectedIndex() const { return impl_->selected_index_; }

const char *ListView::GetSelectedValue() const {
    if (impl_->selected_index_ < 0 ||
        impl_->selected_index_ >= int(impl_->items_.size())) {
        return "";
    } else {
        return impl_->items_[impl_->selected_index_].c_str();
    }
}

void ListView::SetSelectedIndex(int index) {
    impl_->selected_index_ = std::min(int(impl_->items_.size() - 1), index);
}

void ListView::SetOnValueChanged(
        std::function<void(const char *, bool)> onValueChanged) {
    impl_->on_value_changed_ = onValueChanged;
}

Size ListView::CalcPreferredSize(const Theme &theme) const {
    auto padding = ImGui::GetStyle().FramePadding;
    auto *font = ImGui::GetFont();
    ImVec2 size(0, 0);

    for (auto &item : impl_->items_) {
        auto itemSize = font->CalcTextSizeA(theme.font_size, Widget::DIM_GROW,
                                            0.0, item.c_str());
        size.x = std::max(size.x, itemSize.x);
        size.y += ImGui::GetFrameHeight();
    }
    return Size(std::ceil(size.x + 2.0f * padding.x), Widget::DIM_GROW);
}

Widget::DrawResult ListView::Draw(const DrawContext &context) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));
    ImGui::PushItemWidth(frame.width);

    int heightNItems = int(std::floor(frame.height / ImGui::GetFrameHeight()));

    auto result = Widget::DrawResult::NONE;
    auto newSelectedIdx = impl_->selected_index_;
    bool isDoubleClick = false;
    DrawImGuiPushEnabledState();
    if (ImGui::ListBoxHeader(impl_->imgui_id_.c_str(), impl_->items_.size(),
                             heightNItems)) {
        for (size_t i = 0; i < impl_->items_.size(); ++i) {
            bool isSelected = (int(i) == impl_->selected_index_);
            if (ImGui::Selectable(impl_->items_[i].c_str(), &isSelected,
                                  ImGuiSelectableFlags_AllowDoubleClick)) {
                if (isSelected) {
                    newSelectedIdx = i;
                }
                // Dear ImGUI seems to have a bug where it registers a
                // double-click as long as you haven't moved the mouse,
                // no matter how long the time between clicks was.
                if (ImGui::IsMouseDoubleClicked(0)) {
                    isDoubleClick = true;
                }
            }
        }
        ImGui::ListBoxFooter();

        if (newSelectedIdx != impl_->selected_index_ || isDoubleClick) {
            impl_->selected_index_ = newSelectedIdx;
            if (impl_->on_value_changed_) {
                impl_->on_value_changed_(GetSelectedValue(), isDoubleClick);
                result = Widget::DrawResult::REDRAW;
            }
        }
    }
    DrawImGuiPopEnabledState();

    ImGui::PopItemWidth();
    return result;
}

}  // namespace gui
}  // namespace open3d
