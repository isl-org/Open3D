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

#include "open3d/visualization/gui/Combobox.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_combobox_id = 1;

int CalcItemHeight(const Theme& theme) {
    auto em = ImGui::GetTextLineHeight();
    auto padding = ImGui::GetStyle().FramePadding.y;
    return std::ceil(em + 2.0 * padding);
}

}  // namespace
struct Combobox::Impl {
    std::string imgui_id_;
    std::vector<std::string> items_;
    int current_index_ = 0;
    std::function<void(const char*, int)> on_value_changed_;
};

Combobox::Combobox() : impl_(new Combobox::Impl()) {
    std::stringstream s;
    s << "##combobox_" << g_next_combobox_id++;
    impl_->imgui_id_ = s.str();
}

Combobox::Combobox(const std::vector<const char*>& items) : Combobox() {
    for (auto& item : items) {
        AddItem(item);
    }
}

Combobox::~Combobox() {}

void Combobox::ClearItems() {
    impl_->items_.clear();
    impl_->current_index_ = 0;
}

int Combobox::AddItem(const char* name) {
    impl_->items_.push_back(name);
    return impl_->items_.size() - 1;
}

void Combobox::ChangeItem(int index, const char* new_name) {
    impl_->items_[index] = new_name;
}

void Combobox::ChangeItem(const char* orig_name, const char* new_name) {
    for (size_t i = 0; i < impl_->items_.size(); ++i) {
        if (impl_->items_[i] == orig_name) {
            impl_->items_[i] = new_name;
            break;
        }
    }
}

void Combobox::RemoveItem(const char* name) {
    for (size_t i = 0; i < impl_->items_.size(); ++i) {
        if (impl_->items_[i] == name) {
            RemoveItem(int(i));
            break;
        }
    }
}

void Combobox::RemoveItem(int index) {
    if (index >= 0 && index < int(impl_->items_.size())) {
        impl_->items_.erase(impl_->items_.begin() + index);
        if (impl_->current_index_ >= int(impl_->items_.size())) {
            impl_->current_index_ = impl_->items_.size() - 1;
        }
    }
}

const char* Combobox::GetItem(int index) const {
    return impl_->items_[index].c_str();
}

int Combobox::GetSelectedIndex() const { return impl_->current_index_; }

const char* Combobox::GetSelectedValue() const {
    if (impl_->current_index_ >= 0 &&
        impl_->current_index_ < int(impl_->items_.size())) {
        return impl_->items_[impl_->current_index_].c_str();
    } else {
        return "";
    }
}

void Combobox::SetSelectedIndex(int index) {
    if (index >= 0 && index < int(impl_->items_.size())) {
        impl_->current_index_ = index;
    }
}

void Combobox::SetSelectedValue(const char* value) {
    std::string svalue = value;
    for (size_t i = 0; i < impl_->items_.size(); ++i) {
        if (impl_->items_[i] == svalue) {
            SetSelectedIndex(i);
            return;
        }
    }
}

void Combobox::SetOnValueChanged(
        std::function<void(const char*, int)> on_value_changed) {
    impl_->on_value_changed_ = on_value_changed;
}

Size Combobox::CalcPreferredSize(const Theme& theme) const {
    auto button_width = ImGui::GetFrameHeight();  // button is square
    auto padding = ImGui::GetStyle().FramePadding;
    int width = 0;
    for (auto& item : impl_->items_) {
        auto size = ImGui::GetFont()->CalcTextSizeA(theme.font_size, 10000,
                                                    10000, item.c_str());
        width = std::max(width, int(std::ceil(size.x)));
    }
    return Size(width + button_width + 2.0 * padding.x, CalcItemHeight(theme));
}

Combobox::DrawResult Combobox::Draw(const DrawContext& context) {
    bool value_changed = false;
    bool was_open = ImGui::IsPopupOpen(impl_->imgui_id_.c_str());
    bool did_open = false;

    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));

    ImGui::PushStyleColor(
            ImGuiCol_Button,
            colorToImgui(context.theme.combobox_arrow_background_color));
    ImGui::PushStyleColor(
            ImGuiCol_ButtonHovered,
            colorToImgui(context.theme.combobox_arrow_background_color));
    ImGui::PushStyleColor(
            ImGuiCol_ButtonActive,
            colorToImgui(context.theme.combobox_arrow_background_color));

    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(frame.width);
    if (ImGui::BeginCombo(impl_->imgui_id_.c_str(), GetSelectedValue())) {
        if (!was_open) {
            did_open = true;
        }
        for (size_t i = 0; i < impl_->items_.size(); ++i) {
            bool isSelected = false;
            if (ImGui::Selectable(impl_->items_[i].c_str(), &isSelected, 0)) {
                impl_->current_index_ = i;
                value_changed = true;
                if (impl_->on_value_changed_) {
                    impl_->on_value_changed_(GetSelectedValue(), i);
                }
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();  // keyboard focus
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    ImGui::PopStyleColor(3);

    return ((value_changed || did_open) ? Widget::DrawResult::REDRAW
                                        : Widget::DrawResult::NONE);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
