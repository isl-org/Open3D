// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/TabControl.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>

#include "open3d/visualization/gui/Theme.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_tab_control_id = 1;

int CalcTabHeight(const Theme& theme) {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    return int(std::ceil(em + 2.0f * ImGui::GetStyle().FramePadding.y));
}
}  // namespace

struct TabControl::Impl {
    std::vector<std::string> tab_names_;
    std::string imgui_id_;
    int current_index_ = 0;
    int next_selected_index_ = -1;
    std::function<void(int)> on_changed_;
};

TabControl::TabControl() : impl_(new TabControl::Impl()) {
    impl_->imgui_id_ =
            "##tabcontrol_" + std::to_string(g_next_tab_control_id++);
}

TabControl::~TabControl() {}

void TabControl::AddTab(const char* name, std::shared_ptr<Widget> panel) {
    AddChild(panel);
    // Add spaces around the name to add padding
    impl_->tab_names_.push_back(std::string(" ") + name + std::string(" "));
}

void TabControl::SetOnSelectedTabChanged(std::function<void(int)> on_changed) {
    impl_->on_changed_ = on_changed;
}
void TabControl::SetSelectedTabIndex(int index) {
    if (index >= 0 && (size_t)index < impl_->tab_names_.size()) {
        impl_->next_selected_index_ = index;
    }
}
int TabControl::GetSelectedTabIndex() { return impl_->current_index_; }

Size TabControl::CalcPreferredSize(const LayoutContext& context,
                                   const Constraints& constraints) const {
    int width = 0, height = 0;
    for (auto& child : GetChildren()) {
        auto size = child->CalcPreferredSize(context, constraints);
        width = std::max(width, size.width);
        height = std::max(height, size.height);
    }

    return Size(width, height + CalcTabHeight(context.theme) + 2);
}

void TabControl::Layout(const LayoutContext& context) {
    auto tabHeight = CalcTabHeight(context.theme);
    auto frame = GetFrame();
    auto child_rect = Rect(frame.x, frame.y + tabHeight, frame.width,
                           frame.height - tabHeight);

    for (auto& child : GetChildren()) {
        child->SetFrame(child_rect);
    }

    Super::Layout(context);
}

TabControl::DrawResult TabControl::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

    auto result = Widget::DrawResult::NONE;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));
    if (ImGui::BeginTabBar(impl_->imgui_id_.c_str())) {
        for (int i = 0; i < int(impl_->tab_names_.size()); ++i) {
            ImGuiTabItemFlags flags = 0;
            if (impl_->next_selected_index_ == i) {
                flags |= ImGuiTabItemFlags_SetSelected;
                impl_->next_selected_index_ = -1;
                result = DrawResult::REDRAW;
            }
            if (ImGui::BeginTabItem(impl_->tab_names_[i].c_str(), NULL,
                                    flags)) {
                auto r = GetChildren()[i]->Draw(context);
                if (r > result) {
                    result = r;
                }
                ImGui::EndTabItem();

                if (i != impl_->current_index_) {
                    impl_->current_index_ = i;
                    if (impl_->on_changed_) {
                        impl_->on_changed_(i);
                    }
                }
            }
        }
        ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
