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

#include "Widget.h"

#include <functional>

namespace open3d {
namespace gui {

class Combobox : public Widget {
public:
    Combobox();
    explicit Combobox(const std::vector<const char*>& items);
    ~Combobox();

    void AddItem(const char *name);

    int GetSelectedIndex() const;
    // Returns "" if nothing is selected
    const char* GetSelectedValue() const;
    void SetSelectedIndex(int index);

    Size CalcPreferredSize(const Theme& theme) const override;

    DrawResult Draw(const DrawContext& context, float frameDelta) override;

    std::function<void(const char *)> OnValueChanged;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
