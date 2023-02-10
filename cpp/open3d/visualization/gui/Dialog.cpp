// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Dialog.h"

#include <string>

#include "open3d/visualization/gui/Window.h"

namespace open3d {
namespace visualization {
namespace gui {

struct Dialog::Impl {
    std::string title;
    Window *parent = nullptr;
};

Dialog::Dialog(const char *title) : impl_(new Dialog::Impl()) {}

Dialog::~Dialog() {}

Size Dialog::CalcPreferredSize(const LayoutContext &context,
                               const Constraints &constraints) const {
    if (GetChildren().size() == 1) {
        auto child = GetChildren()[0];
        return child->CalcPreferredSize(context, constraints);
    } else {
        return Super::CalcPreferredSize(context, constraints);
    }
}

void Dialog::Layout(const LayoutContext &context) {
    if (GetChildren().size() == 1) {
        auto child = GetChildren()[0];
        child->SetFrame(GetFrame());
        child->Layout(context);
    } else {
        Super::Layout(context);
    }
}

void Dialog::OnWillShow() {}

Widget::DrawResult Dialog::Draw(const DrawContext &context) {
    return Super::Draw(context);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
