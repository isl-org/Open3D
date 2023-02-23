// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/visualizer/ViewControl.h"

namespace open3d {
namespace visualization {

class ViewControlWithEditing : public ViewControl {
public:
    enum EditingMode {
        FreeMode = 0,
        OrthoPositiveX = 1,
        OrthoNegativeX = 2,
        OrthoPositiveY = 3,
        OrthoNegativeY = 4,
        OrthoPositiveZ = 5,
        OrthoNegativeZ = 6,
    };

public:
    virtual ~ViewControlWithEditing() {}

    void Reset() override;
    void ChangeFieldOfView(double step) override;
    void Scale(double scale) override;
    void Rotate(double x, double y, double xo, double yo) override;
    void Translate(double x, double y, double xo, double yo) override;

    void SetEditingMode(EditingMode mode);
    std::string GetStatusString() const;

    EditingMode GetEditingMode() const { return editing_mode_; };
    void ToggleEditingX() {
        if (editing_mode_ == EditingMode::OrthoPositiveX) {
            SetEditingMode(EditingMode::OrthoNegativeX);
        } else {
            SetEditingMode(EditingMode::OrthoPositiveX);
        }
    }

    void ToggleEditingY() {
        if (editing_mode_ == EditingMode::OrthoPositiveY) {
            SetEditingMode(EditingMode::OrthoNegativeY);
        } else {
            SetEditingMode(EditingMode::OrthoPositiveY);
        }
    }

    void ToggleEditingZ() {
        if (editing_mode_ == EditingMode::OrthoPositiveZ) {
            SetEditingMode(EditingMode::OrthoNegativeZ);
        } else {
            SetEditingMode(EditingMode::OrthoPositiveZ);
        }
    }

    void ToggleLocking() { is_view_locked_ = !is_view_locked_; }
    bool IsLocked() const { return is_view_locked_; }

protected:
    EditingMode editing_mode_ = FreeMode;
    ViewParameters view_status_backup_;
    bool is_view_locked_ = false;
};

}  // namespace visualization
}  // namespace open3d
