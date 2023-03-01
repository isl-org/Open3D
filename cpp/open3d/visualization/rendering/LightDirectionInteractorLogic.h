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

class Camera;
class Scene;

class LightDirectionInteractorLogic : public MatrixInteractorLogic {
    using Super = MatrixInteractorLogic;

public:
    LightDirectionInteractorLogic(Scene* scene, Camera* camera);

    void Rotate(int dx, int dy) override;

    void StartMouseDrag();
    void UpdateMouseDragUI();
    void EndMouseDrag();

    Eigen::Vector3f GetCurrentDirection() const;

private:
    Scene* scene_;
    Camera* camera_;
    LightHandle dir_light_;
    Eigen::Vector3f light_dir_at_mouse_down_;

    struct UIObj {
        std::string name;
        Camera::Transform transform;
    };
    std::vector<UIObj> ui_objs_;

    void ClearUI();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
