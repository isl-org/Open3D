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

#include "open3d/visualization/rendering/MatrixInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

class RotationInteractorLogic : public MatrixInteractorLogic {
    using Super = MatrixInteractorLogic;

public:
    explicit RotationInteractorLogic(Camera *camera, double min_far_plane);
    ~RotationInteractorLogic();

    virtual void SetCenterOfRotation(const Eigen::Vector3f &center);

    // Panning is always relative to the camera's left (x) and up (y)
    // axis. Modifies center of rotation and the matrix.
    virtual void Pan(int dx, int dy);

    virtual void StartMouseDrag();
    virtual void UpdateMouseDragUI();
    virtual void EndMouseDrag();

protected:
    double min_far_plane_;
    Camera *camera_;

    Eigen::Vector3f CalcPanVectorWorld(int dx, int dy);
    void UpdateCameraFarPlane();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
