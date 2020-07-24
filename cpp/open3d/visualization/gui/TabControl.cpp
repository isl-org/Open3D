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

#include "open3d/visualization/gui/TabControl.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "open3d/visualization/gui/Theme.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_tab_control_id_ = 1;

int CalcTabHeight(const Theme& theme) {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    return std::ceil(em + 2.0f * ImGui::GetStyle().FramePadding.y);
}
}  // namespace

struct TabControl::Impl {
    std::vector<std::string> tab_names_;
    std::string imgui_id_;
};

TabControl::TabControl() : impl_(new TabControl::Impl()) {
    std::stringstream s;
    s << "tabcontrol_" << g_next_tab_control_id_++;
    impl_->imgui_id_ = s.str();
}

TabControl::~TabControl() {}

void TabControl::AddTab(const char* name, std::shared_ptr<Widget> panel) {
    AddChild(panel);
    impl_->tab_names_.push_back(name);
}

Size TabControl::CalcPreferredSize(const Theme& theme) const {
    int width = 0, height = 0;
    for (auto& child : GetChildren()) {
        auto size = child->CalcPreferredSize(theme);
        width = std::max(width, size.width);
        height = std::max(height, size.height);
    }

    return Size(width, height + CalcTabHeight(theme) + 2);
}

void TabControl::Layout(const Theme& theme) {
    auto tabHeight = CalcTabHeight(theme);
    auto frame = GetFrame();
    auto child_rect = Rect(frame.x, frame.y + tabHeight, frame.width,
                           frame.height - tabHeight);

    for (auto& child : GetChildren()) {
        child->SetFrame(child_rect);
    }

    Super::Layout(theme);
}

TabControl::DrawResult TabControl::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(ImVec2(frame.x, frame.y));

    auto result = Widget::DrawResult::NONE;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(GetFrame().width);
    if (ImGui::BeginTabBar(impl_->imgui_id_.c_str())) {
        for (size_t i = 0; i < impl_->tab_names_.size(); ++i) {
            if (ImGui::BeginTabItem(impl_->tab_names_[i].c_str())) {
                auto r = GetChildren()[i]->Draw(context);
                if (r != Widget::DrawResult::NONE) {
                    result = r;
                }
                ImGui::EndTabItem();
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
