// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/visualizer/O3DVisualizerSelections.h"

namespace open3d {

namespace geometry {
class Geometry3D;
class Image;
}  // namespace geometry

namespace t {
namespace geometry {
class Geometry;
}  // namespace geometry
}  // namespace t

namespace visualization {

namespace rendering {
class Open3DScene;
struct TriangleMeshModel;
}  // namespace rendering

namespace visualizer {

class O3DVisualizer : public gui::Window {
    using Super = gui::Window;

public:
    enum class Shader { STANDARD, UNLIT, NORMALS, DEPTH };

    struct DrawObject {
        std::string name;
        std::shared_ptr<geometry::Geometry3D> geometry;
        std::shared_ptr<t::geometry::Geometry> tgeometry;
        std::shared_ptr<rendering::TriangleMeshModel> model;
        rendering::MaterialRecord material;
        std::string group;
        double time = 0.0;
        bool is_visible = true;

        // internal
        bool is_color_default = true;
    };

    struct UIState {
        gui::SceneWidget::Controls mouse_mode =
                gui::SceneWidget::Controls::ROTATE_CAMERA;
        Shader scene_shader = Shader::STANDARD;
        bool show_settings = false;
        bool show_skybox = true;
        bool show_axes = false;
        bool show_ground = false;
        rendering::Scene::GroundPlane ground_plane =
                rendering::Scene::GroundPlane::XZ;
        bool is_animating = false;
        std::set<std::string> enabled_groups;

        Eigen::Vector4f bg_color = {1.0f, 1.0f, 1.0f, 1.0f};
        int point_size = 3;
        int line_width = 2;

        bool use_ibl = false;
        bool use_sun = true;
        bool sun_follows_camera = true;
        std::string ibl_path = "";  // "" is default path
        int ibl_intensity = 0;
        int sun_intensity = 100000;
        Eigen::Vector3f sun_dir = {0.577f, -0.577f, -0.577f};
        Eigen::Vector3f sun_color = {1.0f, 1.0f, 1.0f};

        double current_time = 0.0;   // seconds
        double time_step = 1.0;      // seconds
        double frame_delay = 0.100;  // seconds
    };

    O3DVisualizer(const std::string& title, int width, int height);
    virtual ~O3DVisualizer();

    void AddAction(const std::string& name,
                   std::function<void(O3DVisualizer&)> callback);

    void SetBackground(const Eigen::Vector4f& bg_color,
                       std::shared_ptr<geometry::Image> bg_image = nullptr);

    void SetShader(Shader shader);

    /// Adds a legacy geometry to the Visualizer
    void AddGeometry(const std::string& name,
                     std::shared_ptr<geometry::Geometry3D> geom,
                     const rendering::MaterialRecord* material = nullptr,
                     const std::string& group = "",
                     double time = 0.0,
                     bool is_visible = true);

    /// Adds a t-geometry to the Visualizer, only DrawableGeometries are
    /// supported.
    void AddGeometry(const std::string& name,
                     std::shared_ptr<t::geometry::Geometry> tgeom,
                     const rendering::MaterialRecord* material = nullptr,
                     const std::string& group = "",
                     double time = 0.0,
                     bool is_visible = true);

    /// Adds a triangle mesh model to the Visualizer.
    void AddGeometry(const std::string& name,
                     std::shared_ptr<rendering::TriangleMeshModel> tgeom,
                     const rendering::MaterialRecord* material = nullptr,
                     const std::string& group = "",
                     double time = 0.0,
                     bool is_visible = true);

    /// Removes the named geometry from the Visualizer
    void RemoveGeometry(const std::string& name);

    /// Updates `update_flags` attributes of named geometry with the matching
    /// attributes from `tgeom`
    void UpdateGeometry(const std::string& name,
                        std::shared_ptr<t::geometry::Geometry> tgeom,
                        uint32_t update_flags);

    /// Show/hide the named geometry
    void ShowGeometry(const std::string& name, bool show);

    /// Returns Visualizer's internal DrawObject for the named geometry
    DrawObject GetGeometry(const std::string& name) const;
    rendering::MaterialRecord GetGeometryMaterial(
            const std::string& name) const;

    void ModifyGeometryMaterial(const std::string& name,
                                const rendering::MaterialRecord* material);

    /// Adds a label with text `text` at the 3D position `pos`.
    void Add3DLabel(const Eigen::Vector3f& pos, const char* text);

    /// Clears all 3D labels created with `Add3DLabel`
    void Clear3DLabels();

    void SetupCamera(float fov,
                     const Eigen::Vector3f& center,
                     const Eigen::Vector3f& eye,
                     const Eigen::Vector3f& up);
    void SetupCamera(const camera::PinholeCameraIntrinsic& intrinsic,
                     const Eigen::Matrix4d& extrinsic);
    void SetupCamera(const Eigen::Matrix3d& intrinsic,
                     const Eigen::Matrix4d& extrinsic,
                     int intrinsic_width_px,
                     int intrinsic_height_px);

    void ResetCameraToDefault();

    void ShowSettings(bool show);
    void ShowSkybox(bool show);
    void SetIBL(const std::string& path);
    void SetIBLIntensity(float intensity);
    void ShowAxes(bool show);
    void ShowGround(bool show);
    void SetGroundPlane(rendering::Scene::GroundPlane plane);
    void EnableSunFollowsCamera(bool enable);
    void EnableBasicMode(bool enable);
    void SetPointSize(int point_size);
    void SetLineWidth(int line_width);
    void EnableGroup(const std::string& group, bool enable);
    void SetMouseMode(gui::SceneWidget::Controls mode);

    std::vector<O3DVisualizerSelections::SelectionSet> GetSelectionSets() const;

    double GetAnimationFrameDelay() const;
    void SetAnimationFrameDelay(double secs);

    double GetAnimationTimeStep() const;
    void SetAnimationTimeStep(double time_step);

    double GetAnimationDuration() const;
    void SetAnimationDuration(double sec);

    double GetCurrentTime() const;
    void SetCurrentTime(double t);

    bool GetIsAnimating() const;
    void SetAnimating(bool is_animating);

    void SetOnAnimationFrame(std::function<void(O3DVisualizer&, double)> cb);

    enum class TickResult { NO_CHANGE, REDRAW };
    void SetOnAnimationTick(
            std::function<TickResult(O3DVisualizer&, double, double)> cb);

    void ExportCurrentImage(const std::string& path);

    UIState GetUIState() const;
    rendering::Open3DScene* GetScene() const;

    /// Starts the RPC interface. See io/rpc/ZMQReceiver for the parameters.
    void StartRPCInterface(const std::string& address, int timeout);

    void StopRPCInterface();

protected:
    void Layout(const gui::LayoutContext& context);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
