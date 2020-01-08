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

#include "Combobox.h"

#include "Theme.h"
#include "Util.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace open3d {
namespace gui {

namespace {
static int gNextComboboxId = 1;

int CalcItemHeight(const Theme& theme) {
    auto em = ImGui::GetTextLineHeight();
    auto padding = ImGui::GetStyle().FramePadding.y;
    return std::ceil(em + 2.0 * padding);
}

}
struct Combobox::Impl {
    std::string imguiId;
    std::vector<std::string> items;
    int currentIndex = 0;
    int selectedIndex = -1;
    std::function<void(const char *)> onValueChanged;
};

Combobox::Combobox()
: impl_(new Combobox::Impl()) {
    std::stringstream s;
    s << "##combobox_" << gNextComboboxId++;
    impl_->imguiId = s.str();
}

Combobox::Combobox(const std::vector<const char*>& items)
: Combobox() {
    for (auto &item : items) {
        AddItem(item);
    }
}

Combobox::~Combobox() {
}

void Combobox::AddItem(const char *name) {
    impl_->items.push_back(name);
}

int Combobox::GetSelectedIndex() const {
    return impl_->currentIndex;
}

const char* Combobox::GetSelectedValue() const {
    if (impl_->currentIndex >= 0
        && impl_->currentIndex < int(impl_->items.size())) {
        return impl_->items[impl_->currentIndex].c_str();
    } else {
        return "";
    }
}

void Combobox::SetSelectedIndex(int index) {
    if (index >= 0 && index < int(impl_->items.size())) {
        impl_->currentIndex = index;
    }
}

void Combobox::SetOnValueChanged(std::function<void(const char *)> onValueChanged) {
    impl_->onValueChanged = onValueChanged;
}

Size Combobox::CalcPreferredSize(const Theme& theme) const {
    auto em = ImGui::GetTextLineHeight();
    int width = 0;
    for (auto &item : impl_->items) {
        auto size = ImGui::GetFont()->CalcTextSizeA(theme.fontSize,
                                                    10000, 10000,
                                                    item.c_str());
        width = std::max(width, int(std::ceil(size.x)));
    }
    return Size(width + em, CalcItemHeight(theme));
}

Combobox::DrawResult Combobox::Draw(const DrawContext& context) {
    bool valueChanged = false;
    bool wasOpen = ImGui::IsPopupOpen(impl_->imguiId.c_str());
    bool didOpen = false;

    auto &frame = GetFrame();
    ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                               frame.y - context.uiOffsetY));

    ImGui::PushStyleColor(ImGuiCol_Button, util::colorToImgui(context.theme.comboboxArrowBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, util::colorToImgui(context.theme.comboboxArrowBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, util::colorToImgui(context.theme.comboboxArrowBackgroundColor));

    ImGui::PushItemWidth(frame.width);
    if (ImGui::BeginCombo(impl_->imguiId.c_str(), GetSelectedValue())) {
        if (!wasOpen) {
            didOpen = true;
        }
        for (size_t i = 0;  i < impl_->items.size();  ++i) {
            bool isSelected = (impl_->selectedIndex == int(i));
            if (ImGui::Selectable(impl_->items[i].c_str(), &isSelected, 0)) {
                impl_->currentIndex = i;
                impl_->selectedIndex = -1;
                valueChanged = true;
                if (impl_->onValueChanged) {
                    impl_->onValueChanged(impl_->items[i].c_str());
                }
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();  // keyboard focus
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::PopStyleColor(3);

    return ((valueChanged || didOpen) ? Widget::DrawResult::CLICKED
                                      : Widget::DrawResult::NONE);
}

} // gui
} // open3d
