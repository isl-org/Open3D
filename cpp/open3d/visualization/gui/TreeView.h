// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <functional>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class TreeView : public Widget {
    using Super = Widget;

public:
    using ItemId = int;

    TreeView();
    ~TreeView();

    /// Returns the ID of the root item, that is,
    /// AddItem(GetRootItem(), "...") will be a top-level item.
    ItemId GetRootItem() const;
    /// Adds an item to the tree.
    ItemId AddItem(ItemId parent_id, const char* text);
    /// Removes an item an all its children (if any) from the tree
    void RemoveItem(ItemId item_id);
    /// Returns the text of the item, or nullptr if item_id cannot be found.
    const char* GetItemText(ItemId item_id) const;
    void SetItemText(ItemId item_id, const char* text);
    std::vector<ItemId> GetItemChildren(ItemId parent_id) const;

    bool GetCanSelectItemsWithChildren() const;
    /// If true, enables selecting items that have children.
    /// Items can be toggled open/closed with the triangles or by
    /// double-clicking. Default is false.
    void SetCanSelectItemsWithChildren(bool can_select);

    /// Returns the currently selected item id in the tree.
    ItemId GetSelectedItemId() const;
    /// Selects the indicated item of the list. Does not call onValueChanged.
    void SetSelectedItemId(ItemId item_id);

    Size CalcPreferredSize(const Theme& theme) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Calls onSelectionChanged(const char *sel_text, ItemId sel_item_id)
    /// when the list selection changes because of user action.
    void SetOnSelectionChanged(
            std::function<void(const char*, ItemId)> on_selection_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
