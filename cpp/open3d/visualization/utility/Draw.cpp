// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/utility/Draw.h"

#include <sstream>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/Model.h"

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

void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name /*= "Open3D"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    gui::Application::GetInstance().Initialize();
    auto draw = std::make_shared<visualizer::O3DVisualizer>(window_name, width,
                                                            height);
    for (auto &o : objects) {
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry);
        } else if (o.tgeometry) {
            draw->AddGeometry(o.name, o.tgeometry);
        } else if (o.model) {
            draw->AddGeometry(o.name, o.model);
        } else {
            utility::LogWarning("Invalid object passed to Draw");
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }

    for (auto &act : actions) {
        draw->AddAction(act.name, act.callback);
    }

    draw->ResetCameraToDefault();

    gui::Application::GetInstance().AddWindow(draw);
    draw.reset();  // so we don't hold onto the pointer after Run() cleans up
    gui::Application::GetInstance().Run();
}

}  // namespace visualization
}  // namespace open3d
