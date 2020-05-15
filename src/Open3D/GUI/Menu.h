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

#include "Events.h"

#define GUI_USE_NATIVE_MENUS 1

namespace open3d {
namespace gui {

struct DrawContext;
struct Theme;

class Menu {
    friend class Application;

public:
    using ItemId = int;
    static constexpr ItemId NO_ITEM = -1;

    Menu();
    virtual ~Menu();

    void AddItem(const char* name,
                 ItemId itemId = NO_ITEM,
                 KeyName key = KEY_NONE);
    void AddMenu(const char* name, std::shared_ptr<Menu> submenu);
    void AddSeparator();

    /// Searches the menu hierarchy down from this menu to find the item
    /// and returns true if the item is enabled.
    bool IsEnabled(ItemId itemId) const;
    /// Searches the menu hierarchy down from this menu to find the item
    /// and set it enabled according to \param enabled.
    void SetEnabled(ItemId itemId, bool enabled);

    bool IsChecked(ItemId itemId) const;
    void SetChecked(ItemId itemId, bool checked);

    int CalcHeight(const Theme& theme) const;

    ItemId DrawMenuBar(const DrawContext& context, bool isEnabled);
    ItemId Draw(const DrawContext& context, const char* name, bool isEnabled);

protected:
    void* GetNativePointer();  // nullptr if not using native menus

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace open3d
