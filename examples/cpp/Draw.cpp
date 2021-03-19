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

#include <cstdlib>

#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"

using namespace open3d;

// TODO: remove hard-coded path.
const std::string TEST_DIR =
        utility::filesystem::GetUnixHome() + "/repo/Open3D/examples/test_data";

void EmptyBox() {
    const double pc_rad = 1.0;
    const double r = 0.4;
    auto sphere_unlit = geometry::TriangleMesh::CreateSphere(r);
    sphere_unlit->Translate({0.0, 1.0, 0.0});
    auto sphere_colored_unlit = geometry::TriangleMesh::CreateSphere(r);
    sphere_colored_unlit->PaintUniformColor({1.0, 0.0, 0.0});
    sphere_colored_unlit->Translate({2.0, 1.0, 0.0});
    auto sphere_lit = geometry::TriangleMesh::CreateSphere(r);
    sphere_lit->ComputeVertexNormals();
    sphere_lit->Translate({4, 1, 0});
    auto sphere_colored_lit = geometry::TriangleMesh::CreateSphere(r);
    sphere_colored_lit->ComputeVertexNormals();
    sphere_colored_lit->PaintUniformColor({0.0, 1.0, 0.0});
    sphere_colored_lit->Translate({6, 1, 0});
    auto big_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>(
            Eigen::Vector3d{-pc_rad, -3, -pc_rad},
            Eigen::Vector3d{6.0 + r, 1.0 + r, pc_rad});
    auto bbox = sphere_unlit->GetAxisAlignedBoundingBox();
    auto sphere_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>(
            bbox.min_bound_, bbox.max_bound_);
    sphere_bbox->color_ = {1.0, 0.5, 0.0};
    auto lines = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_lit->GetAxisAlignedBoundingBox());
    auto lines_colored = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_colored_lit->GetAxisAlignedBoundingBox());
    lines_colored->PaintUniformColor({0.0, 0.0, 1.0});

    visualization::Draw(
            {sphere_unlit, sphere_colored_unlit, sphere_lit, sphere_colored_lit,
             big_bbox, sphere_bbox, lines, lines_colored},
            "Open3D", 640, 480);
}

void BoxWithOjects() {
    const double pc_rad = 1.0;
    const double r = 0.4;
    auto sphere_unlit = geometry::TriangleMesh::CreateSphere(r);
    sphere_unlit->Translate({0.0, 1.0, 0.0});
    auto sphere_colored_unlit = geometry::TriangleMesh::CreateSphere(r);
    sphere_colored_unlit->PaintUniformColor({1.0, 0.0, 0.0});
    sphere_colored_unlit->Translate({2.0, 1.0, 0.0});
    auto sphere_lit = geometry::TriangleMesh::CreateSphere(r);
    sphere_lit->ComputeVertexNormals();
    sphere_lit->Translate({4, 1, 0});
    auto sphere_colored_lit = geometry::TriangleMesh::CreateSphere(r);
    sphere_colored_lit->ComputeVertexNormals();
    sphere_colored_lit->PaintUniformColor({0.0, 1.0, 0.0});
    sphere_colored_lit->Translate({6, 1, 0});
    auto big_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>(
            Eigen::Vector3d{-pc_rad, -3, -pc_rad},
            Eigen::Vector3d{6.0 + r, 1.0 + r, pc_rad});
    auto bbox = sphere_unlit->GetAxisAlignedBoundingBox();
    auto sphere_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>(
            bbox.min_bound_, bbox.max_bound_);
    sphere_bbox->color_ = {1.0, 0.5, 0.0};
    auto lines = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_lit->GetAxisAlignedBoundingBox());
    auto lines_colored = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_colored_lit->GetAxisAlignedBoundingBox());
    lines_colored->PaintUniformColor({0.0, 0.0, 1.0});

    visualization::Draw(
            {sphere_unlit, sphere_colored_unlit, sphere_lit, sphere_colored_lit,
             big_bbox, sphere_bbox, lines, lines_colored},
            "Open3D", 640, 480);
}

int main(int argc, char **argv) {
    if (!utility::filesystem::DirectoryExists(TEST_DIR)) {
        utility::LogError(
                "This example needs to be run from the <build>/bin/examples "
                "directory, test_dir: {}",
                TEST_DIR);
    }
    open3d::visualization::gui::Application::GetInstance().EnableWebRTC();

    BoxWithOjects();
}
