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

#include "open3d/visualization/rendering/LightDirectionInteractorLogic.h"

#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

void CreateCircle(const Eigen::Vector3d& center,
                  const Eigen::Vector3d& u,
                  const Eigen::Vector3d& v,
                  double radius,
                  int n_segs,
                  std::vector<Eigen::Vector3d>& points,
                  std::vector<Eigen::Vector3d>& normals) {
    for (int i = 0; i <= n_segs; ++i) {
        double theta = 2.0 * M_PI * double(i) / double(n_segs);
        double cosT = std::cos(theta);
        double sinT = std::sin(theta);
        Eigen::Vector3d p = center + radius * cosT * u + radius * sinT * v;
        Eigen::Vector3d n = (cosT * u + sinT * v).normalized();
        points.push_back(p);
        normals.push_back(n);
    }
}

std::shared_ptr<geometry::TriangleMesh> CreateArrow(const Eigen::Vector3d& dir,
                                                    double radius,
                                                    double length,
                                                    double head_length,
                                                    int n_segs = 20) {
    Eigen::Vector3d tmp(dir.y(), dir.z(), dir.x());
    Eigen::Vector3d u = dir.cross(tmp).normalized();
    Eigen::Vector3d v = dir.cross(u);

    Eigen::Vector3d start(0, 0, 0);
    Eigen::Vector3d head_start((length - head_length) * dir.x(),
                               (length - head_length) * dir.y(),
                               (length - head_length) * dir.z());
    Eigen::Vector3d end(length * dir.x(), length * dir.y(), length * dir.z());
    auto arrow = std::make_shared<geometry::TriangleMesh>();

    // Cylinder
    CreateCircle(start, u, v, radius, n_segs, arrow->vertices_,
                 arrow->vertex_normals_);
    int n_verts_in_circle = n_segs + 1;
    CreateCircle(head_start, u, v, radius, n_segs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = 0; i < n_segs; ++i) {
        arrow->triangles_.push_back({i, i + 1, n_verts_in_circle + i + 1});
        arrow->triangles_.push_back(
                {n_verts_in_circle + i + 1, n_verts_in_circle + i, i});
    }

    // End of cone
    int start_idx = int(arrow->vertices_.size());
    CreateCircle(head_start, u, v, 2.0 * radius, n_segs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = start_idx; i < int(arrow->vertices_.size()); ++i) {
        arrow->vertex_normals_.push_back(-dir);
    }
    int center_idx = int(arrow->vertices_.size());
    arrow->vertices_.push_back(head_start);
    arrow->vertex_normals_.push_back(-dir);
    for (int i = 0; i < n_segs; ++i) {
        arrow->triangles_.push_back(
                {start_idx + i, start_idx + i + 1, center_idx});
    }

    // Cone
    start_idx = int(arrow->vertices_.size());
    CreateCircle(head_start, u, v, 2.0 * radius, n_segs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = 0; i < n_segs; ++i) {
        int pointIdx = int(arrow->vertices_.size());
        arrow->vertices_.push_back(end);
        arrow->vertex_normals_.push_back(arrow->vertex_normals_[start_idx + i]);
        arrow->triangles_.push_back(
                {start_idx + i, start_idx + i + 1, pointIdx});
    }

    return arrow;
}

}  // namespace

static const Eigen::Vector3d kSkyColor(0.0f, 0.0f, 1.0f);
static const Eigen::Vector3d kSunColor(1.0f, 0.9f, 0.0f);

LightDirectionInteractorLogic::LightDirectionInteractorLogic(Scene* scene,
                                                             Camera* camera)
    : scene_(scene), camera_(camera) {}

void LightDirectionInteractorLogic::Rotate(int dx, int dy) {
    Eigen::Vector3f up = camera_->GetUpVector();
    Eigen::Vector3f right = -camera_->GetLeftVector();
    RotateWorld(-dx, -dy, up, right);
    UpdateMouseDragUI();
}

void LightDirectionInteractorLogic::StartMouseDrag() {
    light_dir_at_mouse_down_ = scene_->GetSunLightDirection();
    auto identity = Camera::Transform::Identity();
    Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

    ClearUI();

    Eigen::Vector3f dir = scene_->GetSunLightDirection();

    double size = model_size_;
    if (size <= 0.001) {
        size = 10;
    }
    double sphere_size = 0.5 * size;  // size is a diameter
    auto sphere_tris = geometry::TriangleMesh::CreateSphere(sphere_size, 20);
    // NOTE: Line set doesn't support UVs. With defaultUnlit shader which
    // requires UVs, filament will print out a warning about the missing vertex
    // attribute. If/when we have a shader specifically for line sets we can use
    // it to avoid the warning.
    auto sphere = geometry::LineSet::CreateFromTriangleMesh(*sphere_tris);
    sphere->PaintUniformColor(kSkyColor);
    auto t0 = Camera::Transform::Identity();
    MaterialRecord mat;
    mat.shader = "defaultUnlit";
    scene_->AddGeometry("__suncagesphere__", *sphere, mat);
    scene_->SetGeometryTransform("__suncagesphere__", t0);
    scene_->GeometryShadows("__suncagesphere__", false, false);
    ui_objs_.push_back({"__suncagesphere__", t0});

    auto sun_radius = 0.05 * size;
    auto sun = geometry::TriangleMesh::CreateSphere(sun_radius, 20);
    sun->PaintUniformColor(kSunColor);
    sun->ComputeVertexNormals();
    sun->triangle_uvs_.resize(sun->triangles_.size() * 3, {0.f, 0.f});
    auto t1 = Camera::Transform::Identity();
    t1.translate(-sphere_size * dir);
    scene_->AddGeometry("__sunsphere__", *sun, mat);
    scene_->SetGeometryTransform("__sunsphere__", t1);
    scene_->GeometryShadows("__sunsphere__", false, false);
    ui_objs_.push_back({"__sunsphere__", t1});

    const double arrow_radius = 0.075 * sun_radius;
    const double arrow_length = 0.333 * size;
    auto sun_dir = CreateArrow(dir.cast<double>(), arrow_radius, arrow_length,
                               0.1 * arrow_length, 20);
    sun_dir->PaintUniformColor(kSunColor);
    sun_dir->ComputeVertexNormals();
    sun_dir->triangle_uvs_.resize(sun_dir->triangles_.size() * 3, {0.f, 0.f});
    auto t2 = Camera::Transform::Identity();
    t2.translate(-sphere_size * dir);
    scene_->AddGeometry("__sunarrow__", *sun_dir, mat);
    scene_->SetGeometryTransform("__sunarrow__", t2);
    scene_->GeometryShadows("__sunarrow__", false, false);
    ui_objs_.push_back({"__sunarrow__", t2});

    UpdateMouseDragUI();
}

void LightDirectionInteractorLogic::UpdateMouseDragUI() {
    Eigen::Vector3f model_center = model_bounds_.GetCenter().cast<float>();
    for (auto& o : ui_objs_) {
        Camera::Transform t = GetMatrix() * o.transform;
        t.pretranslate(model_center);
        scene_->SetGeometryTransform(o.name, t);
    }
}

void LightDirectionInteractorLogic::EndMouseDrag() { ClearUI(); }

void LightDirectionInteractorLogic::ClearUI() {
    for (auto& o : ui_objs_) {
        scene_->RemoveGeometry(o.name);
    }
    ui_objs_.clear();
}

Eigen::Vector3f LightDirectionInteractorLogic::GetCurrentDirection() const {
    return GetMatrix() * light_dir_at_mouse_down_;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
