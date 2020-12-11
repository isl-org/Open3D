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

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/Registration.h"

using namespace open3d;

void VisualizeRegistration(const open3d::t::geometry::PointCloud &source,
                           const open3d::t::geometry::PointCloud &target,
                           const core::Tensor &Transformation) {
    auto pcd = source;
    pcd.Transform(Transformation);
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = pcd.ToLegacyPointCloud();
    *target_ptr = target.ToLegacyPointCloud();
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

void PrintHelp() {
    utility::LogInfo("Usage :");
    utility::LogInfo("  > TensorPointCloudTransform <src_file> <target_file>");
}

int main(int argc, char *argv[]) {
    // TODO: Add argument input options for users and developers
    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    auto device = core::Device("CPU:0");
    auto dtype = core::Dtype::Float64;

    t::geometry::PointCloud source(device);
    t::geometry::PointCloud source2(device);
    t::geometry::PointCloud target(device);
    t::io::ReadPointCloud(argv[1], source, {"auto", false, false, true});
    t::io::ReadPointCloud(argv[1], source2, {"auto", false, false, true});
    t::io::ReadPointCloud(argv[2], target, {"auto", false, false, true});

    // Manual Transformation
    std::vector<double> trans_init_vec{
            0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
            0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

    // Creating Tensor from manual transformation vector
    core::Tensor init_trans(trans_init_vec, {4, 4}, dtype, device);

    // core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);

    auto evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
            source, target, 0.02, init_trans);
    auto corres = evaluation.correspondence_set_;

    std::cout << " Iteration - 0 [Manual Init Transformation] " << std::endl
              << " [Tensor] Registration Results: " << std::endl
              << "   Fitness: " << evaluation.fitness_ << std::endl
              << "   Inlier RMSE: " << evaluation.inlier_rmse_ << std::endl;

    VisualizeRegistration(source, target, init_trans);

    auto reg_p2p = open3d::t::pipelines::registration::RegistrationICP(
            source, target, 0.02, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    1e-6, 1e-6, 30));

    std::cout << " [Point to Point] Max Iteration - " << 30 << std::endl
              << " [Tensor] Registration Results: " << std::endl
              << "   Fitness: " << reg_p2p.fitness_ << std::endl
              << "   Inlier RMSE: " << reg_p2p.inlier_rmse_ << std::endl;

    auto transformation_point2point = reg_p2p.transformation_;

    VisualizeRegistration(source, target, transformation_point2point);

    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source2, target, 0.02, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    1e-6, 1e-6, 30));

    std::cout << " [Point to Plane] Max Iteration - " << 30 << std::endl
              << " [Tensor] Registration Results: " << std::endl
              << "   Fitness: " << reg_p2plane.fitness_ << std::endl
              << "   Inlier RMSE: " << reg_p2plane.inlier_rmse_ << std::endl;

    auto transformation_point2plane = reg_p2plane.transformation_;

    VisualizeRegistration(source2, target, transformation_point2plane);

    return 0;
}