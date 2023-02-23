// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/rendering/CameraInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

class CameraSphereInteractorLogic : public CameraInteractorLogic {
    using Super = CameraInteractorLogic;

public:
    CameraSphereInteractorLogic(Camera* c, double min_far_plane);

    void Rotate(int dx, int dy) override;

    void StartMouseDrag() override;

private:
    float r_at_mousedown_;
    float theta_at_mousedown_;
    float phi_at_mousedown_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
