// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

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
