// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class Window;

/// Base class for dialogs.
class Dialog : public Widget {
    using Super = Widget;

public:
    explicit Dialog(const char* title);
    virtual ~Dialog();

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;
    DrawResult Draw(const DrawContext& context) override;

    virtual void OnWillShow();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
