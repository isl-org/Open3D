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

#include "open3d/visualization/rendering/RotationInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

class CameraInteractorLogic : public RotationInteractorLogic {
    using Super = RotationInteractorLogic;

public:
    CameraInteractorLogic(Camera* c, double min_far_plane);

    void SetBoundingBox(
            const geometry::AxisAlignedBoundingBox& bounds) override;

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;
    void Dolly(int dy, DragType type) override;
    void Dolly(float z_dist, Camera::Transform matrix_in) override;

    void Pan(int dx, int dy) override;

    /// Sets camera field of view
    void Zoom(int dy, DragType drag_type);

    void RotateLocal(float angle_rad, const Eigen::Vector3f& axis);
    void MoveLocal(const Eigen::Vector3f& v);

    void RotateFly(int dx, int dy);

    void StartMouseDrag() override;
    void ResetMouseDrag();
    void UpdateMouseDragUI() override;
    void EndMouseDrag() override;

private:
    double fov_at_mouse_down_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
