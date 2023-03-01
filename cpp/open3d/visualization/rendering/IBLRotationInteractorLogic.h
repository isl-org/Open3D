// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/rendering/MatrixInteractorLogic.h"
#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace visualization {
namespace rendering {

class Scene;

class IBLRotationInteractorLogic : public MatrixInteractorLogic {
    using Super = MatrixInteractorLogic;

public:
    IBLRotationInteractorLogic(Scene* scene, Camera* camera);

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;

    void StartMouseDrag();
    void UpdateMouseDragUI();
    void EndMouseDrag();

    Camera::Transform GetCurrentRotation() const;

private:
    Scene* scene_;
    Camera* camera_;
    bool skybox_currently_visible_ = false;
    Camera::Transform ibl_rotation_at_mouse_down_;

    void ClearUI();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
