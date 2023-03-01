// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
