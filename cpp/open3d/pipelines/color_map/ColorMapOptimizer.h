// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <memory>
#include <vector>

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/color_map/NonRigidOptimizer.h"
#include "open3d/pipelines/color_map/RigidOptimizer.h"

namespace open3d {
namespace pipelines {
namespace color_map {

/// \brief Function for color mapping of reconstructed scenes via optimization.
///
/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014.
class ColorMapOptimizer {
public:
    /// \param mesh The input mesh geometry.
    /// \param imgs_rgbd A list of RGBDImages seen by cameras.
    /// \param camera_trajectory Cameras' parameters.
    ColorMapOptimizer(const geometry::TriangleMesh& mesh,
                      const std::vector<std::shared_ptr<geometry::RGBDImage>>&
                              images_rgbd,
                      const camera::PinholeCameraTrajectory& camera_trajectory);

    /// Run rigid optimization.
    void RunRigidOptimizer(const RigidOptimizerOption& option);

    /// Run non-rigid optimization.
    void RunNonRigidOptimizer(const NonRigidOptimizerOption& option);

    /// Get optimized mesh, should be used after running optimizations.
    std::shared_ptr<geometry::TriangleMesh> GetMesh() const { return mesh_; }

protected:
    std::shared_ptr<geometry::TriangleMesh> mesh_;
    std::vector<std::shared_ptr<geometry::RGBDImage>> images_rgbd_;
    std::shared_ptr<camera::PinholeCameraTrajectory> camera_trajectory_;
    std::vector<std::shared_ptr<geometry::Image>> images_gray_;
    std::vector<std::shared_ptr<geometry::Image>> images_dx_;
    std::vector<std::shared_ptr<geometry::Image>> images_dy_;
    std::vector<std::shared_ptr<geometry::Image>> images_color_;
    std::vector<std::shared_ptr<geometry::Image>> images_depth_;
};

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
