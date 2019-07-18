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

#include "Open3D/Visualization/Visualizer/ViewControl.h"

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
