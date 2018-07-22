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

#include "UnitTest.h"
#include "../Core/Geometry/PointCloud.h"
#include <random>
#include <Eigen/Geometry>

using namespace three;

TEST(EstimateNormals, Default)
{
    NotImplemented();
}

TEST(EstimateNormals, OrientNormalsUsingMST)
{
    std::default_random_engine gen(0);
    std::uniform_int_distribution<int> u(0,1);
    auto sample = [&u,&gen]() { return (u(gen) ? -1 : 1); };

    // Test planar surfaces
    PointCloud plane_xy;
    for(int y = 0; y < 10; ++y) {
        for(int x = 0; x < 20; ++x) {
            plane_xy.points_.emplace_back(x, y, 0);
            plane_xy.normals_.emplace_back(0, 0, sample());
        }
    }
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    auto T_rand = Eigen::Translation3d(Eigen::Vector3d::Random()) *
                  Eigen::AngleAxisd(axis.norm(), axis/axis.norm());
    auto plane_rand = plane_xy;
    plane_rand.Transform(T_rand.matrix());

    ASSERT_TRUE(OrientNormalsUsingMST(plane_xy));
    ASSERT_TRUE(OrientNormalsUsingMST(plane_rand));

    const auto& n_ref_xy = plane_xy.normals_[0];
    for(const auto& n : plane_xy.normals_) {
        EXPECT_NEAR(n_ref_xy.dot(n), 1.0, 1e-7);
    }

    const auto& n_ref_rand = plane_rand.normals_[0];
    for(const auto& n : plane_rand.normals_) {
        EXPECT_NEAR(n_ref_rand.dot(n), 1.0, 1e-7);
    }

    // Test sphere surface
    PointCloud sphere;
    constexpr double r = 1.0;
    for(int i_az = 0; i_az < 360; i_az = i_az + 30) {
        for(int i_elev = -89; i_elev <= 89; i_elev = i_elev + 30) {
            double az = i_az*M_PI/180.0;
            double elev = i_elev*M_PI/180.0;
            Eigen::Vector3d n(cos(elev)*cos(az), cos(elev)*sin(az), sin(elev));
            sphere.points_.emplace_back(r*n[0], r*n[1], r*n[2]);
            sphere.normals_.emplace_back(n*sample());
        }
    }
    if(sphere.points_[0].dot(sphere.normals_[0]) < 0) {
        sphere.normals_[0] *= -1.0;
    }
    ASSERT_TRUE(OrientNormalsUsingMST(sphere));

    for(int i =0; i < sphere.points_.size(); ++i) {
        EXPECT_TRUE(sphere.points_[i].dot(sphere.normals_[i]) > 0);
    }
}
