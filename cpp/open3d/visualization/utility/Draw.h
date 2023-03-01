// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/visualizer/O3DVisualizer.h"

namespace open3d {
namespace visualization {

struct DrawObject {
    std::string name;
    std::shared_ptr<geometry::Geometry3D> geometry;
    std::shared_ptr<t::geometry::Geometry> tgeometry;
    std::shared_ptr<rendering::TriangleMeshModel> model;
    bool is_visible;

    DrawObject(const std::string &n,
               std::shared_ptr<geometry::Geometry3D> g,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<t::geometry::Geometry> tg,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<rendering::TriangleMeshModel> m,
               bool vis = true);
};

struct DrawAction {
    std::string name;
    std::function<void(visualizer::O3DVisualizer &)> callback;
};

void Draw(const std::vector<std::shared_ptr<geometry::Geometry3D>> &geometries,
          const std::string &window_name = "Open3D",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name = "Open3D",
        int width = 1024,
        int height = 768,
        const std::vector<DrawAction> &actions = {});

void Draw(const std::vector<std::shared_ptr<rendering::TriangleMeshModel>>
                  &models,
          const std::string &window_name = "Open3D",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name = "Open3D",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

}  // namespace visualization
}  // namespace open3d
