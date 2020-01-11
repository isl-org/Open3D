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

#include "ListView.h"

#include "Theme.h"

#include <imgui.h>

#include <cmath>
#include <sstream>

namespace open3d {
namespace gui {

namespace {
static const int NO_SELECTION = -1;
static int gNextListBoxId = 1;
}

struct ListView::Impl {
    std::string imguiId;
    std::vector<std::string> items;
    int selectedIndex = NO_SELECTION;
    std::function<void(const char *, bool)> onValueChanged;
};

ListView::ListView()
: impl_(std::make_unique<ListView::Impl>()) {
    std::stringstream s;
    s << "##combobox_" << gNextListBoxId++;
    impl_->imguiId = s.str();
}

ListView::~ListView() {
}

void ListView::SetItems(const std::vector<std::string> &items)  {
    impl_->items = items;
    impl_->selectedIndex = NO_SELECTION;
}

int ListView::GetSelectedIndex() const {
    return impl_->selectedIndex;
}

const char* ListView::GetSelectedValue() const {
    if (impl_->selectedIndex < 0
        || impl_->selectedIndex >= impl_->items.size()) {
        return "";
    } else {
        return impl_->items[impl_->selectedIndex].c_str();
    }
}

void ListView::SetSelectedIndex(int index) {
    impl_->selectedIndex = std::min(int(impl_->items.size()), index);
}

void ListView::SetOnValueChanged(std::function<void(const char *, bool)> onValueChanged) {
    impl_->onValueChanged = onValueChanged;
}

Size ListView::CalcPreferredSize(const Theme& theme) const {
    auto padding = ImGui::GetStyle().FramePadding;
    auto *font = ImGui::GetFont();
    ImVec2 size(0, 0);

    for (auto &item : impl_->items) {
        auto itemSize = font->CalcTextSizeA(theme.fontSize, Widget::DIM_GROW,
                                            0.0, item.c_str());
        size.x = std::max(size.x, itemSize.x);
        size.y += ImGui::GetFrameHeight();
    }
    return Size(std::ceil(size.x + 2.0f * padding.x),
                Widget::DIM_GROW);
}

Widget::DrawResult ListView::Draw(const DrawContext& context) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                               frame.y - context.uiOffsetY));
    ImGui::PushItemWidth(frame.width);

    int heightNItems = int(std::floor(frame.height / ImGui::GetFrameHeight()));

    auto result = Widget::DrawResult::NONE;
    auto newSelectedIdx = impl_->selectedIndex;
    bool isDoubleClick = false;
    DrawImGuiPushEnabledState();
    if (ImGui::ListBoxHeader(impl_->imguiId.c_str(), impl_->items.size(),
                             heightNItems)) {
        for (size_t i = 0;  i < impl_->items.size();  ++i) {
            bool isSelected = (i == impl_->selectedIndex);
            if (ImGui::Selectable(impl_->items[i].c_str(), &isSelected,
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

        if (newSelectedIdx != impl_->selectedIndex || isDoubleClick) {
            impl_->selectedIndex = newSelectedIdx;
            if (impl_->onValueChanged) {
                impl_->onValueChanged(GetSelectedValue(), isDoubleClick);
                result = Widget::DrawResult::CLICKED;
            }
        }
    }
    DrawImGuiPopEnabledState();

    ImGui::PopItemWidth();
    return result;
}

} // namespace gui
} // namespace open3d
