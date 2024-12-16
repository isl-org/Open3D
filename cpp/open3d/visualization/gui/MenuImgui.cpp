// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/MenuImgui.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

static const float EXTRA_PADDING_Y = 1.0f;

namespace {

std::string CalcShortcutText(KeyName key) {
    return "";
    // Dear ImGUI doesn't currently support shortcut keys
    // if (key == KEY_NONE) {
    //     return "";
    // }
    // char k = char(key);
    // if (k >= 'a' && k <= 'z') {
    //     k -= 32; // uppercase
    // }
    // return std::string("Ctrl + ") + char(k);
}

}  // namespace

struct MenuImgui::Impl {
    struct MenuItem {
        MenuBase::ItemId id_;
        std::string name_;
        KeyName shortcut_key_;
        std::shared_ptr<MenuImgui> submenu_;
        MenuImgui::Impl *submenu_impl_ =
                nullptr;  // so FindMenuItem needn't be a friend
        bool is_enabled_ = true;
        bool is_checked_ = false;
        bool is_separator_ = false;
    };

    std::vector<MenuItem> items_;
    std::unordered_map<int, size_t> id2idx_;
    bool submenu_visible_ = false;
    bool submenu_visibility_changed_ = false;

    MenuItem *FindMenuItem(ItemId item_id) {
        auto it = this->id2idx_.find(item_id);
        if (it != this->id2idx_.end()) {
            return &this->items_[it->second];
        }
        for (auto &item : this->items_) {
            if (item.submenu_) {
                auto *possibility = item.submenu_impl_->FindMenuItem(item_id);
                if (possibility) {
                    return possibility;
                }
            }
        }
        return nullptr;
    }
};

MenuImgui::MenuImgui() : impl_(new MenuImgui::Impl()) {}

MenuImgui::~MenuImgui() {}

void *MenuImgui::GetNativePointer() { return nullptr; }

void MenuImgui::AddItem(const char *name,
                        ItemId itemId /*= NO_ITEM*/,
                        KeyName key /*= KEY_NONE*/) {
    InsertItem(int(impl_->items_.size()), name, itemId, key);
}

void MenuImgui::AddMenu(const char *name, std::shared_ptr<MenuBase> submenu) {
    InsertMenu(int(impl_->items_.size()), name, submenu);
}

void MenuImgui::AddSeparator() { InsertSeparator(int(impl_->items_.size())); }

void MenuImgui::InsertItem(int index,
                           const char *name,
                           ItemId item_id /*= NO_ITEM*/,
                           KeyName key /*= KEY_NONE*/) {
    for (auto &kv : impl_->id2idx_) {
        if (int(kv.second) >= index) {
            kv.second += 1;
        }
    }
    impl_->id2idx_[item_id] = impl_->items_.size();
    impl_->items_.insert(impl_->items_.begin() + index,
                         {item_id, name, key, nullptr});
}

void MenuImgui::InsertMenu(int index,
                           const char *name,
                           std::shared_ptr<MenuBase> submenu) {
    for (auto &kv : impl_->id2idx_) {
        if (int(kv.second) >= index) {
            kv.second += 1;
        }
    }
    auto imgui_submenu = std::dynamic_pointer_cast<MenuImgui>(submenu);
    impl_->items_.insert(impl_->items_.begin() + index,
                         {NO_ITEM, name, KEY_NONE, imgui_submenu,
                          imgui_submenu->impl_.get()});
}

void MenuImgui::InsertSeparator(int index) {
    for (auto &kv : impl_->id2idx_) {
        if (int(kv.second) >= index) {
            kv.second += 1;
        }
    }
    impl_->items_.insert(
            impl_->items_.begin() + index,
            {NO_ITEM, "", KEY_NONE, nullptr, nullptr, false, false, true});
}

int MenuImgui::GetNumberOfItems() const { return int(impl_->items_.size()); }

bool MenuImgui::IsEnabled(ItemId item_id) const {
    auto *item = impl_->FindMenuItem(item_id);
    if (item) {
        return item->is_enabled_;
    }
    return false;
}

void MenuImgui::SetEnabled(ItemId item_id, bool enabled) {
    auto *item = impl_->FindMenuItem(item_id);
    if (item) {
        item->is_enabled_ = enabled;
    }
}

bool MenuImgui::IsChecked(ItemId item_id) const {
    auto *item = impl_->FindMenuItem(item_id);
    if (item) {
        return item->is_checked_;
    }
    return false;
}

void MenuImgui::SetChecked(ItemId item_id, bool checked) {
    auto *item = impl_->FindMenuItem(item_id);
    if (item) {
        item->is_checked_ = checked;
    }
}

int MenuImgui::CalcHeight(const Theme &theme) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    return int(std::ceil(em + 2.0f * (padding.y + EXTRA_PADDING_Y)));
}

bool MenuImgui::CheckVisibilityChange() const {
    return impl_->submenu_visibility_changed_;
}

MenuBase::ItemId MenuImgui::DrawMenuBar(const DrawContext &context,
                                        bool is_enabled) {
    ItemId activate_id = NO_ITEM;

    ImVec2 size;
    size.x = ImGui::GetIO().DisplaySize.x;
    size.y = float(CalcHeight(context.theme));
    auto padding = ImGui::GetStyle().FramePadding;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(padding.x, padding.y + EXTRA_PADDING_Y));

    impl_->submenu_visibility_changed_ = false;
    ImGui::BeginMainMenuBar();
    for (auto &item : impl_->items_) {
        if (item.submenu_) {
            bool submenu_visible = item.submenu_impl_->submenu_visible_;
            auto id = item.submenu_->Draw(context, item.name_.c_str(),
                                          is_enabled);
            if (id >= 0) {
                activate_id = id;
            }
            if (submenu_visible != item.submenu_impl_->submenu_visible_) {
                impl_->submenu_visibility_changed_ = true;
            }
        }
    }

    // Before we end the menu bar, draw a one pixel line at the bottom.
    // This gives a little definition to the end of the menu, otherwise
    // it just ends and looks a bit odd. This should probably be a pretty
    // subtle difference from the menubar background.
    auto y = size.y - 1;
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddLine(ImVec2(0, y), ImVec2(size.x, y),
                       context.theme.menubar_border_color.ToABGR32(), 1.0f);

    ImGui::EndMainMenuBar();

    ImGui::PopStyleVar();

    return activate_id;
}

MenuBase::ItemId MenuImgui::Draw(const DrawContext &context,
                                 const char *name,
                                 bool is_enabled) {
    ItemId activate_id = NO_ITEM;

    // The default ImGUI menus are hideous:  there is no margin and the items
    // are spaced way too tightly. However, you can't just add WindowPadding
    // because then the highlight doesn't extend to the window edge. So we need
    // to draw the menu item in pieces. First to get the highlight (if
    // necessary), then draw the actual item inset to the left and right to get
    // the text and checkbox. Unfortunately, there is no way to get a right
    // margin without the window padding.

    auto *font = ImGui::GetFont();
    int em = int(std::ceil(ImGui::GetTextLineHeight()));
    int padding = context.theme.default_margin;
    int name_width = 0, shortcut_width = 0;
    for (auto &item : impl_->items_) {
        auto size1 = font->CalcTextSizeA(float(context.theme.font_size), 10000,
                                         10000, item.name_.c_str());
        auto shortcut = CalcShortcutText(item.shortcut_key_);
        auto size2 = font->CalcTextSizeA(float(context.theme.font_size), 10000,
                                         10000, shortcut.c_str());
        name_width = std::max(name_width, int(std::ceil(size1.x)));
        shortcut_width = std::max(shortcut_width, int(std::ceil(size2.x)));
    }
    int width = padding + name_width + 2 * em + shortcut_width + 2 * em +
                int(std::ceil(1.5 * em)) + padding;  // checkbox

    ImGui::SetNextWindowContentSize(ImVec2(float(width), 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(0, float(context.theme.default_margin)));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding,
                        float(context.theme.font_size) / 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(float(context.theme.default_margin),
                               float(context.theme.default_margin)));

    if (ImGui::BeginMenu(name, is_enabled)) {
        impl_->submenu_visible_ = true;

        for (size_t i = 0; i < impl_->items_.size(); ++i) {
            auto &item = impl_->items_[i];
            if (item.is_separator_) {
                ImGui::Separator();
            } else if (item.submenu_) {
                ImGui::SetCursorPosX(float(padding));
                auto possibility = item.submenu_->Draw(
                        context, item.name_.c_str(), is_enabled);
                if (possibility != NO_ITEM) {
                    activate_id = possibility;
                }
            } else {
                // Save y position, then draw empty item for the highlight.
                // Set the enabled flag, in case the real item isn't.
                auto y = ImGui::GetCursorPosY();
                if (ImGui::MenuItem(" ", "", false, item.is_enabled_)) {
                    activate_id = item.id_;
                }
                // Restore the y position, and draw the menu item with the
                // proper margins on top.
                // Note: can't set width (width - 2 * padding) because
                //       SetNextItemWidth is ignored.
                ImGui::SetCursorPos(ImVec2(float(padding), y));
                auto shortcut_text = CalcShortcutText(item.shortcut_key_);
                ImGui::MenuItem(item.name_.c_str(), shortcut_text.c_str(),
                                item.is_checked_, item.is_enabled_);
            }
        }
        ImGui::EndMenu();
    } else {
        impl_->submenu_visible_ = false;
    }

    ImGui::PopStyleVar(3);

    return activate_id;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
