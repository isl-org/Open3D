// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#include "open3d/visualization/gui/MultiSelectListView.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_multi_list_box_id = 1;
}  // namespace

struct MultiSelectListView::Impl {
    std::string imgui_id_;
    std::vector<std::string> items_;
    std::vector<bool> select_status_;
    int max_items_ = -1;
    std::function<void(int, const char *, bool)> on_value_changed_;
};

MultiSelectListView::MultiSelectListView() : impl_(new MultiSelectListView::Impl()) {
    impl_->imgui_id_ = "##multiselectlistview_" + std::to_string(g_next_multi_list_box_id++);
}

MultiSelectListView::~MultiSelectListView() {}

void MultiSelectListView::SetItems(const std::vector<std::string> &items) {
    impl_->items_ = items;
    impl_->select_status_.resize(items.size());
    for(auto i = 0u; i < items.size(); i++) {
        impl_->select_status_[i] = false;
    }
}

void MultiSelectListView::SetMaxVisibleItems(int items) {
    impl_->max_items_ = items;
}

/// Returns the currently selected item in the list.
std::vector<int> MultiSelectListView::GetSelectedIndices() const {
    std::vector<int> selected;
    for (auto i = 0u; i < impl_->select_status_.size(); i++) {
        if (impl_->select_status_[i]) {
            selected.push_back(int(i));
        }
    }
    return selected;
}
/// Returns the value of the currently selected item in the list.
const char * MultiSelectListView::GetValue(int index) const {
    if (index < 0 || index >= int(impl_->items_.size())) {
        return "";
    } else {
        return impl_->items_[index].c_str();
    }
}

void MultiSelectListView::SetSelectedIndex(int index, bool selected) {
    index = std::min(int(impl_->items_.size() - 1), index);
    impl_->select_status_[index] = selected;
}

void MultiSelectListView::SetOnValueChanged(
        std::function<void(int, const char *, bool)> on_value_changed) {
    impl_->on_value_changed_ = on_value_changed;
}

Size MultiSelectListView::CalcPreferredSize(const LayoutContext &context,
                                 const Constraints &constraints) const {
    auto padding = ImGui::GetStyle().FramePadding;
    auto *font = ImGui::GetFont();
    ImVec2 size(0, 0);

    for (auto &item : impl_->items_) {
        auto item_size = font->CalcTextSizeA(float(context.theme.font_size),
                                             float(constraints.width), 0.0,
                                             item.c_str());
        size.x = std::max(size.x, item_size.x);
        size.y += ImGui::GetFrameHeight();
    }
    auto h = Widget::DIM_GROW;
    if (impl_->max_items_ > 0) {
        auto nr = std::max(int(impl_->items_.size()), 3);
        nr = std::min(nr, impl_->max_items_);
        h = int(ImGui::GetFrameHeight() * (float)nr + 2.0f * padding.y);
    }
    return Size(int(std::ceil(size.x + 2.0f * padding.x)), h);
}

Size MultiSelectListView::CalcMinimumSize(const LayoutContext &context) const {
    return Size(0, 3 * context.theme.font_size);
}

Widget::DrawResult MultiSelectListView::Draw(const DrawContext &context) {
    auto &frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) + ImGui::GetScrollY()));
    ImGui::PushItemWidth(float(frame.width));

    ImGui::PushStyleColor(ImGuiCol_FrameBg,
                          colorToImgui(context.theme.list_background_color));
    ImGui::PushStyleColor(ImGuiCol_Header,  // selection color
                          colorToImgui(context.theme.list_selected_color));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,  // hover color
                          colorToImgui(Color(0, 0, 0, 0)));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  // click-hold color
                          colorToImgui(context.theme.list_selected_color));

    int height_in_items =
            int(std::floor(frame.height / ImGui::GetFrameHeight()));

    auto result = Widget::DrawResult::NONE;
    bool is_double_click = false;
    DrawImGuiPushEnabledState();
    if (ImGui::ListBoxHeader(impl_->imgui_id_.c_str(),
                             int(impl_->items_.size()), height_in_items)) {
        for (size_t i = 0; i < impl_->items_.size(); ++i) {
            bool is_selected = impl_->select_status_[i];
            // ImGUI's list wants to hover over items, which is not done by
            // any major OS, is pretty unnecessary (you can see the cursor
            // right over the row), and acts really weird. Worse, the hover
            // is drawn instead of the selection color. So to get rid of it
            // we need hover to be the selected color iff this item is
            // selected, otherwise we want it to be transparent.
            if (is_selected) {
                ImGui::PushStyleColor(
                        ImGuiCol_HeaderHovered,
                        colorToImgui(context.theme.list_selected_color));
            } else {
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                                      colorToImgui(Color(0, 0, 0, 0)));
            }
            if (ImGui::Selectable(impl_->items_[i].c_str(), &is_selected,
                                  ImGuiSelectableFlags_AllowDoubleClick)) {

                // Dear ImGUI seems to have a bug where it registers a
                // double-click as long as you haven't moved the mouse,
                // no matter how long the time between clicks was.
                if (ImGui::IsMouseDoubleClicked(0)) {
                    is_double_click = true;
                }

                if (is_selected != impl_->select_status_[i]) {
                    impl_->select_status_[i] = is_selected;
                    if (impl_->on_value_changed_) {
                        impl_->on_value_changed_(int(i), GetValue(int(i)), is_double_click);
                    }
                    result = Widget::DrawResult::REDRAW;
                }
            }
            ImGui::PopStyleColor();
        }
        ImGui::ListBoxFooter();
    }
    DrawImGuiPopEnabledState();

    ImGui::PopStyleColor(4);

    ImGui::PopItemWidth();
    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
