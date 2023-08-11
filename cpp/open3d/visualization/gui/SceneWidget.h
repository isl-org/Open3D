// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>

#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}  // namespace camera

namespace geometry {
class AxisAlignedBoundingBox;
class Geometry3D;
}  // namespace geometry

namespace t {
namespace geometry {
class Geometry;
}  // namespace geometry
}  // namespace t

namespace visualization {
namespace rendering {
class Camera;
class CameraManipulator;
class MatrixInteractorLogic;
class Open3DScene;
class View;
}  // namespace rendering
}  // namespace visualization

namespace visualization {
namespace gui {

class Label3D;
class Color;

class SceneWidget : public Widget {
    using Super = Widget;

public:
    class MouseInteractor {
    public:
        virtual ~MouseInteractor() = default;

        virtual rendering::MatrixInteractorLogic& GetMatrixInteractor() = 0;
        virtual void Mouse(const MouseEvent& e) = 0;
        virtual void Key(const KeyEvent& e) = 0;
        virtual bool Tick(const TickEvent& e) { return false; }
    };

public:
    explicit SceneWidget();
    ~SceneWidget() override;

    void SetFrame(const Rect& f) override;

    enum Controls {
        ROTATE_CAMERA,
        ROTATE_CAMERA_SPHERE,
        FLY,
        ROTATE_SUN,
        ROTATE_IBL,
        ROTATE_MODEL,
        PICK_POINTS
    };
    void SetViewControls(Controls mode);

    void SetupCamera(float verticalFoV,
                     const geometry::AxisAlignedBoundingBox& scene_bounds,
                     const Eigen::Vector3f& center_of_rotation);
    void SetupCamera(const camera::PinholeCameraIntrinsic& intrinsic,
                     const Eigen::Matrix4d& extrinsic,
                     const geometry::AxisAlignedBoundingBox& scene_bounds);
    void SetupCamera(const Eigen::Matrix3d& intrinsic,
                     const Eigen::Matrix4d& extrinsic,
                     int intrinsic_width_px,
                     int intrinsic_height_px,
                     const geometry::AxisAlignedBoundingBox& scene_bounds);
    void LookAt(const Eigen::Vector3f& center,
                const Eigen::Vector3f& eye,
                const Eigen::Vector3f& up);

    void SetOnCameraChanged(
            std::function<void(visualization::rendering::Camera*)>
                    on_cam_changed);

    Eigen::Vector3f GetCenterOfRotation() const;
    void SetCenterOfRotation(const Eigen::Vector3f& center);

    /// Enables changing the directional light with the mouse.
    /// SceneWidget will update the light's direction, so onDirChanged is
    /// only needed if other things need to be updated (like a UI).
    void SetOnSunDirectionChanged(
            std::function<void(const Eigen::Vector3f&)> on_dir_changed);

    void SetScene(std::shared_ptr<rendering::Open3DScene> scene);
    std::shared_ptr<rendering::Open3DScene> GetScene() const;

    rendering::View* GetRenderView() const;  // is nullptr if no scene

    /// Enable (or disable) caching of scene to improve UI responsiveness when
    /// dealing with large scenes (especially point clouds)
    void EnableSceneCaching(bool enable);

    /// Forces the scene to redraw regardless of Renderer caching
    /// settings.
    void ForceRedraw();
    enum class Quality { FAST, BEST };
    void SetRenderQuality(Quality level);
    Quality GetRenderQuality() const;

    enum class CameraPreset {
        PLUS_X,  // at (X, 0, 0), looking (-1, 0, 0)
        PLUS_Y,  // at (0, Y, 0), looking (0, -1, 0)
        PLUS_Z   // at (0, 0, Z), looking (0, 0, 1) [default OpenGL camera]
    };
    void GoToCameraPreset(CameraPreset preset);

    struct PickableGeometry {
        std::string name;
        const geometry::Geometry3D* geometry = nullptr;
        const t::geometry::Geometry* tgeometry = nullptr;

        PickableGeometry(const std::string& n, const geometry::Geometry3D* g)
            : name(n), geometry(g) {}

        PickableGeometry(const std::string& n, const t::geometry::Geometry* t)
            : name(n), tgeometry(t) {}

        /// This is for programmatic use when you don't want to know if you
        /// have a geometry or a t::geometry; exactly one of g and t should be
        /// non-null; the other should be nullptr.
        PickableGeometry(const std::string& n,
                         const geometry::Geometry3D* g,
                         const t::geometry::Geometry* t)
            : name(n), geometry(g), tgeometry(t) {}
    };

    void SetSunInteractorEnabled(bool enable);

    void SetPickableGeometry(const std::vector<PickableGeometry>& geometry);
    void SetPickablePointSize(int px);
    void SetOnPointsPicked(
            std::function<void(
                    const std::map<
                            std::string,
                            std::vector<std::pair<size_t, Eigen::Vector3d>>>&,
                    int)> on_picked);
    void SetOnStartedPolygonPicking(std::function<void()> on_poly_pick);
    enum class PolygonPickAction { CANCEL = 0, SELECT };
    void DoPolygonPick(PolygonPickAction action);

    // 3D Labels
    std::shared_ptr<Label3D> AddLabel(const Eigen::Vector3f& pos,
                                      const char* text);
    void RemoveLabel(std::shared_ptr<Label3D> label);
    void ClearLabels();

    Widget::DrawResult Draw(const DrawContext& context) override;

    Widget::EventResult Mouse(const MouseEvent& e) override;
    Widget::EventResult Key(const KeyEvent& e) override;
    Widget::DrawResult Tick(const TickEvent& e) override;

private:
    visualization::rendering::Camera* GetCamera() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
