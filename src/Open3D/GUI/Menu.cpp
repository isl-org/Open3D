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

#include "Menu.h"

#include "Theme.h"
#include "Widget.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace open3d {
namespace gui {

static const float EXTRA_PADDING_Y = 1.0f;

struct Menu::MenuItem {
    Menu::ItemId id;
    std::string name;
    std::string shortcut;
    std::shared_ptr<Menu> submenu;
    bool isEnabled = true;
    bool isChecked = false;
    bool isSeparator = false;
};

struct Menu::Impl {
    std::vector<Menu::MenuItem> items;
    std::unordered_map<int, size_t> id2idx;
};

Menu::Menu()
: impl_(new Menu::Impl()) {
}

Menu::~Menu() {
}

void Menu::AddItem(const char *name, const char *shortcut, ItemId itemId /*= NO_ITEM*/) {
    impl_->id2idx[itemId] = impl_->items.size();
    impl_->items.push_back({ itemId, name, shortcut, nullptr });
}

void Menu::AddMenu(const char *name, std::shared_ptr<Menu> submenu) {
    impl_->items.push_back({ NO_ITEM, name, "", submenu });
}

void Menu::AddSeparator() {
    impl_->items.push_back({ NO_ITEM, "", "", nullptr, false, false, true });
}

bool Menu::IsEnabled(ItemId itemId) const {
    auto *item = FindMenuItem(itemId);
    if (item) {
        return item->isEnabled;
    }
    return false;
}

void Menu::SetEnabled(ItemId itemId, bool enabled) {
    auto *item = FindMenuItem(itemId);
    if (item) {
        item->isEnabled = enabled;
    }
}

bool Menu::IsChecked(ItemId itemId) const {
    auto *item = FindMenuItem(itemId);
    if (item) {
        return item->isChecked;
    }
    return false;
}

void Menu::SetChecked(ItemId itemId, bool checked) {
    auto *item = FindMenuItem(itemId);
    if (item) {
        item->isChecked = checked;
    }
}

Menu::MenuItem* Menu::FindMenuItem(ItemId itemId) const {
    auto it = impl_->id2idx.find(itemId);
    if (it != impl_->id2idx.end()) {
        return &impl_->items[it->second];
    }
    for (auto &item : impl_->items) {
        if (item.submenu) {
            auto *possibility = item.submenu->FindMenuItem(itemId);
            if (possibility) {
                return possibility;
            }
        }
    }
    return nullptr;
}

int Menu::CalcHeight(const Theme& theme) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    return std::ceil(em + 2.0f * (padding.y + EXTRA_PADDING_Y));
}

Menu::ItemId Menu::DrawMenuBar(const DrawContext& context) {
    ItemId activatedId = NO_ITEM;

    ImVec2 size;
    size.x = ImGui::GetIO().DisplaySize.x;
    size.y = CalcHeight(context.theme);
    auto padding = ImGui::GetStyle().FramePadding;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(padding.x, padding.y + EXTRA_PADDING_Y));

    ImGui::BeginMainMenuBar();
    for (auto &item : impl_->items) {
        if (item.submenu) {
            auto id = item.submenu->Draw(context, item.name.c_str());
            if (id >= 0) {
                activatedId = id;
            }
        }
    }

    // Before we end the menu bar, draw a one pixel line at the bottom.
    // This gives a little definition to the end of the menu, otherwise
    // it just ends and looks a bit odd. This should probably be a pretty
    // subtle difference from the menubar background.
    auto y = size.y - 1;
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddLine(ImVec2(0, y), ImVec2(size.x, y),
                      context.theme.menubarBorderColor.ToABGR32(), 1.0f);

    ImGui::EndMainMenuBar();

    ImGui::PopStyleVar();

    return activatedId;
}

Menu::ItemId Menu::Draw(const DrawContext& context, const char *name) {
    ItemId activatedId = NO_ITEM;

    // The default ImGUI menus are hideous:  there is no margin and the items
    // are spaced way too tightly. However, you can't just add WindowPadding
    // because then the highlight doesn't extend to the window edge. So we need
    // to draw the menu item in pieces. First to get the highlight (if necessary),
    // then draw the actual item inset to the left and right to get the text
    // and checkbox. Unfortunately, there is no way to get a right margin
    // without the window padding.

    auto *font = ImGui::GetFont();
    int em = std::ceil(ImGui::GetTextLineHeight());
    int padding = context.theme.defaultMargin;
    int nameWidth = 0, shortcutWidth = 0;
    for (auto &item : impl_->items) {
        auto size1 = font->CalcTextSizeA(context.theme.fontSize, 10000, 10000,
                                         item.name.c_str());
        auto size2 = font->CalcTextSizeA(context.theme.fontSize, 10000, 10000,
                                         item.shortcut.c_str());
        nameWidth = std::max(nameWidth, int(std::ceil(size1.x)));
        shortcutWidth = std::max(shortcutWidth, int(std::ceil(size2.x)));
    }
    int width = padding +
                nameWidth + 2 * em +
                shortcutWidth + 2 * em +
                std::ceil(1.5 * em) + padding;  // checkbox

    ImGui::SetNextWindowContentWidth(width);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, context.theme.defaultMargin));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, context.theme.fontSize / 3);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(context.theme.defaultMargin,
                                                          context.theme.defaultMargin));

    if (ImGui::BeginMenu(name)) {
        for (size_t i = 0;  i < impl_->items.size();  ++i) {
            auto &item = impl_->items[i];
            if (item.isSeparator) {
                ImGui::Separator();
            } else if (item.submenu) {
                ImGui::SetCursorPosX(padding);
                auto possibility = item.submenu->Draw(context, item.name.c_str());
                if (possibility != NO_ITEM) {
                    activatedId = possibility;
                }
            } else {
                // Save y position, then draw empty item for the highlight.
                // Set the enabled flag, in case the real item isn't.
                auto y = ImGui::GetCursorPosY();
                if (ImGui::MenuItem("", "", false, item.isEnabled)) {
                    activatedId = item.id;
                }
                // Restore the y position, and draw the menu item with the
                // proper margins on top.
                // Note: can't set width (width - 2 * padding) because
                //       SetNextItemWidth is ignored.
                ImGui::SetCursorPos(ImVec2(padding, y));
                ImGui::MenuItem(item.name.c_str(), item.shortcut.c_str(),
                                item.isChecked, item.isEnabled);
            }
        }
        ImGui::EndMenu();
    }

    ImGui::PopStyleVar(3);

    return activatedId;
}

} // gui
} // open3d
