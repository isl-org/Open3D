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

class TextEdit : public Widget {
public:
    TextEdit();
    ~TextEdit();

    /// Returns the current text value displayed
    const char* GetText() const;
    /// Sets the current text value displayed. Does not call onTextChanged or
    /// onValueChanged.
    void SetText(const char* text);

    /// Returns the text displayed if the text value is empty.
    const char* GetPlaceholderText() const;
    /// Sets the text to display if the text value is empty.
    void SetPlaceholderText(const char* text);

    Size CalcPreferredSize(const Theme& theme) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Sets the function that is called whenever the text in the widget
    /// changes. This will be called for every keystroke and edit.
    void SetOnTextChanged(std::function<void(const char*)> on_text_changed);
    /// Sets the function that is called whenever the text is the widget
    /// is finished editing via pressing enter or clicking off the widget.
    void SetOnValueChanged(std::function<void(const char*)> on_value_changed);

protected:
    /// Returns true if new text is valid. Otherwise call SetText() with a
    /// valid value and return false.
    virtual bool ValidateNewText(const char* text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
