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

#include <functional>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class Combobox : public Widget {
public:
    Combobox();
    explicit Combobox(const std::vector<const char*>& items);
    ~Combobox() override;

    void ClearItems();
    /// Adds an item to the combobox. Its index is the order in which it is
    /// added, so the first item's index is 0, the second is 1, etc.
    /// Returns the index of the new item.
    int AddItem(const char* name);

    /// Changes the item's text. \param index must be valid, otherwise
    /// nothing will happen.
    void ChangeItem(int index, const char* name);
    /// If an item exists with \param orig_name, it will be changed to
    /// \param new_name.
    void ChangeItem(const char* orig_name, const char* new_name);

    /// Removes the first item matching the given text.
    void RemoveItem(const char* name);
    /// Removes the item at \param index.
    void RemoveItem(int index);

    int GetNumberOfItems() const;

    /// Returns the text of the item at \param index. \param index must be
    /// valid.
    const char* GetItem(int index) const;

    int GetSelectedIndex() const;
    /// Returns the text of the selected value, or "" if nothing is selected
    const char* GetSelectedValue() const;
    /// Sets the selected item by index. Does not call the onValueChanged
    /// callback.
    void SetSelectedIndex(int index);
    /// Sets the selected item by value. Does nothing if \param value is not an
    /// item, but will return false. Does not call the onValueChanged callback
    bool SetSelectedValue(const char* value);

    Size CalcPreferredSize(const Theme& theme) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Specifies a callback function which will be called when the value
    /// changes as a result of user action.
    void SetOnValueChanged(
            std::function<void(const char*, int)> on_value_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
