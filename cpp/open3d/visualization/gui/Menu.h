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

#pragma once

#include <memory>

#include "open3d/visualization/gui/Events.h"

#define GUI_USE_NATIVE_MENUS 1

namespace open3d {
namespace visualization {
namespace gui {

struct DrawContext;
struct Theme;

/// The menu item action is handled by Window, rather than by registering a
/// a callback function with (non-existent) Menu::SetOnClicked(). This is
/// because on macOS the menubar is global over all application windows, so any
/// callback would need to go find the data object corresponding to the active
/// window.
class Menu {
    friend class Application;

public:
    using ItemId = int;
    static constexpr ItemId NO_ITEM = -1;

    Menu();
    virtual ~Menu();

    void AddItem(const char* name,
                 ItemId item_id = NO_ITEM,
                 KeyName key = KEY_NONE);
    void AddMenu(const char* name, std::shared_ptr<Menu> submenu);
    void AddSeparator();

    void InsertItem(int index,
                    const char* name,
                    ItemId item_id = NO_ITEM,
                    KeyName key = KEY_NONE);
    void InsertMenu(int index, const char* name, std::shared_ptr<Menu> submenu);
    void InsertSeparator(int index);

    int GetNumberOfItems() const;

    /// Searches the menu hierarchy down from this menu to find the item
    /// and returns true if the item is enabled.
    bool IsEnabled(ItemId item_id) const;
    /// Searches the menu hierarchy down from this menu to find the item
    /// and set it enabled according to \p enabled.
    void SetEnabled(ItemId item_id, bool enabled);

    bool IsChecked(ItemId item_id) const;
    void SetChecked(ItemId item_id, bool checked);

    int CalcHeight(const Theme& theme) const;

    /// Returns true if submenu visibility changed on last call to DrawMenuBar
    bool CheckVisibilityChange() const;

    ItemId DrawMenuBar(const DrawContext& context, bool is_enabled);
    ItemId Draw(const DrawContext& context, const char* name, bool is_enabled);

protected:
    void* GetNativePointer();  // nullptr if not using native menus

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
