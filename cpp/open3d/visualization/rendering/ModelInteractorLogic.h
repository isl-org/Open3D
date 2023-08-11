// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>

#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/RotationInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

class Open3DScene;

class ModelInteractorLogic : public RotationInteractorLogic {
    using Super = RotationInteractorLogic;

public:
    ModelInteractorLogic(Open3DScene* scene,
                         Camera* camera,
                         double min_far_plane);
    virtual ~ModelInteractorLogic();

    void SetBoundingBox(
            const geometry::AxisAlignedBoundingBox& bounds) override;

    void SetModel(GeometryHandle axes,
                  const std::vector<GeometryHandle>& objects);

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;
    void Dolly(float dy, DragType drag_type) override;
    void Pan(int dx, int dy) override;

    void StartMouseDrag() override;
    void UpdateMouseDragUI() override;
    void EndMouseDrag() override;

private:
    Open3DScene* scene_;
    bool is_axes_visible_;

    geometry::AxisAlignedBoundingBox bounds_at_mouse_down_;
    std::map<std::string, Camera::Transform> transforms_at_mouse_down_;

    void UpdateBoundingBox(const Camera::Transform& t);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
