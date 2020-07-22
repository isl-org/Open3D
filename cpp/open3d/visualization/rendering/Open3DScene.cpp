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

#include "open3d/visualization/rendering/Open3DScene.h"

#include <algorithm>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
std::shared_ptr<geometry::TriangleMesh> CreateAxisGeometry(double axis_length) {
    const double sphere_radius = 0.005 * axis_length;
    const double cyl_radius = 0.0025 * axis_length;
    const double cone_radius = 0.0075 * axis_length;
    const double cyl_height = 0.975 * axis_length;
    const double cone_height = 0.025 * axis_length;

    auto mesh_frame = geometry::TriangleMesh::CreateSphere(sphere_radius);
    mesh_frame->ComputeVertexNormals();
    mesh_frame->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<geometry::TriangleMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    // Add UVs because material shader for axes expects them
    mesh_frame->triangle_uvs_.resize(mesh_frame->triangles_.size() * 3,
                                     {0.0, 0.0});

    return mesh_frame;
}

GeometryHandle RecreateAxis(Scene* scene,
                            GeometryHandle old_id,
                            const geometry::AxisAlignedBoundingBox& bounds,
                            bool enabled) {
    if (old_id) {
        scene->RemoveGeometry(old_id);
    }

    // Axes length should be the longer of the bounds extent or 25% of the
    // distance from the origin. The latter is necessary so that the axis is
    // big enough to be visible even if the object is far from the origin.
    // See caterpillar.ply from Tanks & Temples.
    auto axis_length = bounds.GetMaxExtent();
    if (axis_length < 0.001) {  // avoid div by zero errors in CreateAxes()
        axis_length = 1.0;
    }
    axis_length = std::max(axis_length, 0.25 * bounds.GetCenter().norm());
    auto mesh = CreateAxisGeometry(axis_length);
    auto axis = scene->AddGeometry(*mesh);
    scene->SetGeometryShadows(axis, false, false);
    scene->SetEntityEnabled(axis, enabled);
    return axis;
}

}  // namespace

Open3DScene::Open3DScene(Renderer& renderer) : renderer_(renderer) {
    scene_ = renderer_.CreateScene();
    auto scene = renderer_.GetScene(scene_);

    axis_ = RecreateAxis(scene, axis_, bounds_, false);

    LightDescription desc;
    desc.intensity = 45000;
    desc.direction = {0.577f, -0.577f, -0.577f};
    desc.cast_shadows = true;
    desc.custom_attributes["custom_type"] = "SUN";
    sun_ = scene->AddLight(desc);
}

Open3DScene::~Open3DScene() {
    ClearGeometry();
    auto scene = renderer_.GetScene(scene_);
    scene->RemoveGeometry(axis_);
    scene->SetIndirectLight(IndirectLightHandle());
    scene->SetSkybox(SkyboxHandle());

    if (sun_) {
        scene->RemoveLight(sun_);
    }
}

ViewHandle Open3DScene::CreateView() {
    auto scene = renderer_.GetScene(scene_);
    view_ = scene->AddView(0, 0, 1, 1);
    auto view = scene->GetView(view_);
    view->SetClearColor({1.0f, 1.0f, 1.0f});

    return view_;
}

void Open3DScene::DestroyView(ViewHandle view) {
    auto scene = renderer_.GetScene(scene_);
    scene->RemoveView(view);
}

View* Open3DScene::GetView(ViewHandle view) const {
    auto scene = renderer_.GetScene(scene_);
    return scene->GetView(view);
}

void Open3DScene::SetIndirectLight(IndirectLightHandle ibl) {
    ibl_ = ibl;
    auto scene = renderer_.GetScene(scene_);
    scene->SetIndirectLight(ibl_);
}

void Open3DScene::SetSkybox(SkyboxHandle skybox) {
    skybox_ = skybox;
    auto scene = renderer_.GetScene(scene_);
    scene->SetSkybox(skybox_);
}

void Open3DScene::ClearGeometry() {
    auto scene = renderer_.GetScene(scene_);
    for (auto hgeom : model_) {
        scene->RemoveGeometry(hgeom);
    }
    for (auto hgeom : fast_model_) {
        scene->RemoveGeometry(hgeom);
    }
    model_.clear();
    fast_model_.clear();
    bounds_ = geometry::AxisAlignedBoundingBox();
    axis_ = RecreateAxis(scene, axis_, bounds_, scene->GetEntityEnabled(axis_));
}

GeometryHandle Open3DScene::AddGeometry(
        std::shared_ptr<const geometry::Geometry3D> geom,
        const MaterialInstanceHandle& material_id,
        bool add_downsampled_copy_for_fast_rendering /*= true*/) {
    auto scene = renderer_.GetScene(scene_);
    auto hgeom = scene->AddGeometry(*geom, material_id);
    bounds_ += scene->GetEntityBoundingBox(hgeom);
    model_.push_back(hgeom);
    scene->SetEntityEnabled(hgeom, (lod_ == LOD::HIGH_DETAIL));

    if (add_downsampled_copy_for_fast_rendering) {
        const std::size_t kMinPointsForDecimation = 6000000;
        auto pcd = std::dynamic_pointer_cast<const geometry::PointCloud>(geom);
        if (pcd && pcd->points_.size() > kMinPointsForDecimation) {
            int sample_rate =
                    pcd->points_.size() / (kMinPointsForDecimation / 2);
            auto small = pcd->UniformDownSample(sample_rate);
            auto hsmall = scene->AddGeometry(*small, material_id);
            fast_model_.push_back(hsmall);
            scene->SetEntityEnabled(hsmall, (lod_ == LOD::FAST));
        } else {
            fast_model_.push_back(hgeom);
        }
    } else {
        fast_model_.push_back(hgeom);
    }

    // Bounding box may have changed, force recreation of axes
    axis_ = RecreateAxis(scene, axis_, bounds_, scene->GetEntityEnabled(axis_));

    return hgeom;
}

void Open3DScene::SetLOD(LOD lod) {
    if (lod != lod_) {
        auto scene = renderer_.GetScene(scene_);
        // Disable all the geometry
        for (auto hgeom : model_) {
            scene->SetEntityEnabled(hgeom, false);
        }
        for (auto hgeom : fast_model_) {
            scene->SetEntityEnabled(hgeom, false);
        }
        // Enable the appropriate geometry for the LOD
        lod_ = lod;
        if (lod_ == LOD::HIGH_DETAIL) {
            for (auto hgeom : model_) {
                scene->SetEntityEnabled(hgeom, (lod_ == LOD::HIGH_DETAIL));
            }
        } else {
            for (auto hgeom : fast_model_) {
                scene->SetEntityEnabled(hgeom, (lod_ == LOD::FAST));
            }
        }
    }
}

Open3DScene::LOD Open3DScene::GetLOD() const { return lod_; }

Scene* Open3DScene::GetScene() const { return renderer_.GetScene(scene_); }

Camera* Open3DScene::GetCamera() const {
    auto scene = renderer_.GetScene(scene_);
    auto view = scene->GetView(view_);
    return view->GetCamera();
}

const std::vector<GeometryHandle>& Open3DScene::GetModel(
        LOD lod /*= LOD::HIGH_DETAIL*/) const {
    if (lod == LOD::FAST) {
        return fast_model_;
    } else {
        return model_;
    }
}

GeometryHandle Open3DScene::GetAxis() const { return axis_; }
SkyboxHandle Open3DScene::GetSkybox() const { return skybox_; }
IndirectLightHandle Open3DScene::GetIndirectLight() const { return ibl_; }
LightHandle Open3DScene::GetSun() const { return sun_; }

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
