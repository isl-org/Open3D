// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/utility/Draw.h"

#include <sstream>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/Open3DScene.h"

namespace open3d {
namespace visualization {

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<geometry::Geometry3D> g,
                       bool vis /*= true*/) {
    this->name = n;
    this->geometry = g;
    this->is_visible = vis;
}

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<t::geometry::Geometry> tg,
                       bool vis /*= true*/) {
    this->name = n;
    this->tgeometry = tg;
    this->is_visible = vis;
}

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<rendering::TriangleMeshModel> m,
                       bool vis /*= true*/) {
    this->name = n;
    this->model = m;
    this->is_visible = vis;
}

// ----------------------------------------------------------------------------
void Draw(const std::vector<std::shared_ptr<geometry::Geometry3D>> &geometries,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(geometries.size());
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), geometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name /*= "Open3D"*/,
        int width /*= 1024*/,
        int height /*= 768*/,
        const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(tgeometries.size());
    for (size_t i = 0; i < tgeometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), tgeometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(const std::vector<std::shared_ptr<rendering::TriangleMeshModel>>
                  &models,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), models[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

// Advanced Draw function with full feature parity to draw.py.
// Implements the exact orchestration order and behavior from Python.
std::string Draw(const std::vector<DrawObject> &objects,
                 const std::string &window_name /*= "Open3D"*/,
                 int width /*= 1024*/,
                 int height /*= 768*/,
                 const std::vector<DrawAction> &actions /*= {}*/,
                 const DrawConfig &config) {
    // 1. Initialize GUI application instance
    gui::Application::GetInstance().Initialize();

    // 2. Create visualizer window
    auto draw = std::make_shared<visualizer::O3DVisualizer>(window_name, width,
                                                            height);

    // 3. Set background color and image (if provided)
    if (config.bg_color.has_value() || config.bg_image) {
        Eigen::Vector4f bg_color_vec(1.0f, 1.0f, 1.0f, 1.0f);
        if (config.bg_color.has_value()) {
            bg_color_vec = config.bg_color.value().cast<float>();
        }
        draw->SetBackground(bg_color_vec, config.bg_image);
    }

    // 4. Add action buttons
    for (const auto &a : actions) {
        draw->AddAction(a.name, a.callback);
    }

    // 5. Set point and line sizes
    if (config.point_size.has_value()) {
        draw->SetPointSize(config.point_size.value());
    }
    if (config.line_width.has_value()) {
        draw->SetLineWidth(config.line_width.value());
    }

    // 6. Add geometries
    for (const auto &o : objects) {
        const rendering::MaterialRecord *material =
                o.has_material ? &o.material : nullptr;
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry, material, o.group, o.time,
                              o.is_visible);
        } else if (o.tgeometry) {
            draw->AddGeometry(o.name, o.tgeometry, material, o.group, o.time,
                              o.is_visible);
        } else if (o.model) {
            draw->AddGeometry(o.name, o.model, material, o.group, o.time,
                              o.is_visible);
        } else {
            utility::LogWarning("Invalid object passed to Draw");
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }

    // 7. Reset camera to default view
    draw->ResetCameraToDefault();

    // 8. Setup camera using either lookat/eye/up or intrinsic/extrinsic
    if (config.lookat.has_value() && config.eye.has_value() &&
        config.up.has_value()) {
        draw->SetupCamera(config.field_of_view, config.lookat.value(),
                          config.eye.value(), config.up.value());
    } else if (config.intrinsic_matrix.has_value() &&
               config.extrinsic_matrix.has_value()) {
        draw->SetupCamera(config.intrinsic_matrix.value(),
                          config.extrinsic_matrix.value(), width, height);
    }

    // 9. Override near/far clip planes if requested. Read current camera state
    // so only the specified planes are changed; position/orientation are kept.
    if (config.near_plane.has_value() || config.far_plane.has_value()) {
        auto *camera = draw->GetScene()->GetCamera();
        double fov = camera->GetFieldOfView();
        double near = config.near_plane.has_value()
                              ? double(config.near_plane.value())
                              : camera->GetNear();
        double far = config.far_plane.has_value()
                             ? double(config.far_plane.value())
                             : camera->GetFar();
        double aspect = double(width) / double(height);
        camera->SetProjection(fov, aspect, near, far,
                              camera->GetFieldOfViewType());
    }

    // 10. Show world-space axes
    if (config.show_axes.has_value()) {
        draw->ShowAxes(config.show_axes.value());
    }

    // 11. Set animation parameters
    draw->SetAnimationTimeStep(config.animation_time_step);
    if (config.animation_duration.has_value()) {
        draw->SetAnimationDuration(config.animation_duration.value());
    }

    // 12. Set UI visibility
    if (config.show_ui.has_value()) {
        draw->ShowSettings(config.show_ui.value());
    }

    // 13. Set IBL and sky box
    if (config.ibl.has_value()) {
        draw->SetIBL(config.ibl.value());
    }
    if (config.ibl_intensity.has_value()) {
        draw->SetIBLIntensity(config.ibl_intensity.value());
    }
    if (config.show_skybox.has_value()) {
        draw->ShowSkybox(config.show_skybox.value());
    }

    // 14. Configure RPC interface and close callback
    if (!config.rpc_interface.empty()) {
        std::string rpc_addr = config.rpc_interface;
        if (rpc_addr == "default") {
            rpc_addr = "tcp://127.0.0.1:51454";
        }
        draw->StartRPCInterface(rpc_addr, 10000);

        // Install close callback to stop RPC when window closes
        draw->SetOnClose([draw]() {
            draw->StopRPCInterface();
            return true;
        });
    }

    // 15. Enable raw/basic rendering mode
    if (config.raw_mode.has_value()) {
        draw->EnableBasicMode(config.raw_mode.value());
    }

    // 16. Call user init callback
    if (config.on_init) {
        config.on_init(*draw);
    }

    // 17. Set animation callbacks
    if (config.on_animation_frame) {
        draw->SetOnAnimationFrame(config.on_animation_frame);
    }
    if (config.on_animation_tick) {
        draw->SetOnAnimationTick(config.on_animation_tick);
    }

    // 18. Add window and handle blocking vs non-blocking
    gui::Application::GetInstance().AddWindow(draw);

    if (config.non_blocking_and_return_uid) {
        // Non-blocking mode: return window UID
        auto uid = draw->GetWebRTCUID();
        draw.reset();  // release our reference
        return uid;
    } else {
        // Blocking mode: release our reference so window teardown happens
        // before global engine cleanup inside Application::Run().
        draw.reset();
        gui::Application::GetInstance().Run();
        return "";
    }
}

// Simple overload that creates a DrawObject version and calls advanced Draw
// with default config for backward compatibility.
void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    Draw(objects, window_name, width, height, actions, DrawConfig());
}

}  // namespace visualization
}  // namespace open3d
