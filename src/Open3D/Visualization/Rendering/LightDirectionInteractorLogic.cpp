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

#include "LightDirectionInteractorLogic.h"

#include "Camera.h"
#include "Scene.h"

#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
namespace visualization {

namespace {

void CreateCircle(const Eigen::Vector3d& center,
                  const Eigen::Vector3d& u,
                  const Eigen::Vector3d& v,
                  double radius,
                  int nSegs,
                  std::vector<Eigen::Vector3d>& points,
                  std::vector<Eigen::Vector3d>& normals) {
    for (int i = 0; i <= nSegs; ++i) {
        double theta = 2.0 * M_PI * double(i) / double(nSegs);
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
                                                    double headLength,
                                                    int nSegs = 20) {
    Eigen::Vector3d tmp(dir.y(), dir.z(), dir.x());
    Eigen::Vector3d u = dir.cross(tmp).normalized();
    Eigen::Vector3d v = dir.cross(u);

    Eigen::Vector3d start(0, 0, 0);
    Eigen::Vector3d headStart((length - headLength) * dir.x(),
                              (length - headLength) * dir.y(),
                              (length - headLength) * dir.z());
    Eigen::Vector3d end(length * dir.x(), length * dir.y(), length * dir.z());
    auto arrow = std::make_shared<geometry::TriangleMesh>();

    // Cylinder
    CreateCircle(start, u, v, radius, nSegs, arrow->vertices_,
                 arrow->vertex_normals_);
    int nVertsInCircle = nSegs + 1;
    CreateCircle(headStart, u, v, radius, nSegs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = 0; i < nSegs; ++i) {
        arrow->triangles_.push_back({i, i + 1, nVertsInCircle + i + 1});
        arrow->triangles_.push_back(
                {nVertsInCircle + i + 1, nVertsInCircle + i, i});
    }

    // End of cone
    int startIdx = int(arrow->vertices_.size());
    CreateCircle(headStart, u, v, 2.0 * radius, nSegs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = startIdx; i < int(arrow->vertices_.size()); ++i) {
        arrow->vertex_normals_.push_back(-dir);
    }
    int centerIdx = int(arrow->vertices_.size());
    arrow->vertices_.push_back(headStart);
    arrow->vertex_normals_.push_back(-dir);
    for (int i = 0; i < nSegs; ++i) {
        arrow->triangles_.push_back(
                {startIdx + i, startIdx + i + 1, centerIdx});
    }

    // Cone
    startIdx = int(arrow->vertices_.size());
    CreateCircle(headStart, u, v, 2.0 * radius, nSegs, arrow->vertices_,
                 arrow->vertex_normals_);
    for (int i = 0; i < nSegs; ++i) {
        int pointIdx = int(arrow->vertices_.size());
        arrow->vertices_.push_back(end);
        arrow->vertex_normals_.push_back(arrow->vertex_normals_[startIdx + i]);
        arrow->triangles_.push_back({startIdx + i, startIdx + i + 1, pointIdx});
    }

    return arrow;
}

}  // namespace

static const Eigen::Vector3d SKY_COLOR(0.0f, 0.0f, 1.0f);
static const Eigen::Vector3d SUN_COLOR(1.0f, 0.9f, 0.0f);

LightDirectionInteractorLogic::LightDirectionInteractorLogic(Scene* scene,
                                                             Camera* camera)
    : scene_(scene), camera_(camera) {}

void LightDirectionInteractorLogic::SetDirectionalLight(LightHandle dirLight) {
    dirLight_ = dirLight;
}

void LightDirectionInteractorLogic::Rotate(int dx, int dy) {
    Eigen::Vector3f up = camera_->GetUpVector();
    Eigen::Vector3f right = -camera_->GetLeftVector();
    RotateWorld(-dx, -dy, up, right);
    UpdateMouseDragUI();
}

void LightDirectionInteractorLogic::StartMouseDrag() {
    lightDirAtMouseDown_ = scene_->GetLightDirection(dirLight_);
    auto identity = Camera::Transform::Identity();
    Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

    ClearUI();

    Eigen::Vector3f dir = scene_->GetLightDirection(dirLight_);

    double size = modelSize_;
    if (size <= 0.001) {
        size = 10;
    }
    double sphereSize = 0.5 * size;  // size is a diameter
    auto sphereTris = geometry::TriangleMesh::CreateSphere(sphereSize, 20);
    auto sphere = geometry::LineSet::CreateFromTriangleMesh(*sphereTris);
    sphere->PaintUniformColor(SKY_COLOR);
    auto t0 = Camera::Transform::Identity();
    uiObjs_.push_back({scene_->AddGeometry(*sphere), t0});
    scene_->SetEntityTransform(uiObjs_[0].handle, t0);
    scene_->SetGeometryShadows(uiObjs_[0].handle, false, false);

    auto sunRadius = 0.05 * size;
    auto sun = geometry::TriangleMesh::CreateSphere(sunRadius, 20);
    sun->PaintUniformColor(SUN_COLOR);
    auto t1 = Camera::Transform::Identity();
    t1.translate(-sphereSize * dir);
    uiObjs_.push_back({scene_->AddGeometry(*sun), t1});
    scene_->SetEntityTransform(uiObjs_[1].handle, t1);
    scene_->SetGeometryShadows(uiObjs_[1].handle, false, false);

    const double arrowRadius = 0.075 * sunRadius;
    const double arrowLength = 0.333 * size;
    auto sunDir = CreateArrow(dir.cast<double>(), arrowRadius, arrowLength,
                              0.1 * arrowLength, 20);
    sunDir->PaintUniformColor(SUN_COLOR);
    auto t2 = Camera::Transform::Identity();
    t2.translate(-sphereSize * dir);
    uiObjs_.push_back({scene_->AddGeometry(*sunDir), t2});
    scene_->SetEntityTransform(uiObjs_[2].handle, t2);
    scene_->SetGeometryShadows(uiObjs_[2].handle, false, false);

    UpdateMouseDragUI();
}

void LightDirectionInteractorLogic::UpdateMouseDragUI() {
    Eigen::Vector3f modelCenter = modelBounds_.GetCenter().cast<float>();
    for (auto& o : uiObjs_) {
        Camera::Transform t = GetMatrix() * o.transform;
        t.pretranslate(modelCenter);
        scene_->SetEntityTransform(o.handle, t);
    }
}

void LightDirectionInteractorLogic::EndMouseDrag() { ClearUI(); }

void LightDirectionInteractorLogic::ClearUI() {
    for (auto& o : uiObjs_) {
        scene_->RemoveGeometry(o.handle);
    }
    uiObjs_.clear();
}

Eigen::Vector3f LightDirectionInteractorLogic::GetCurrentDirection() const {
    return GetMatrix() * lightDirAtMouseDown_;
}

}  // namespace visualization
}  // namespace open3d
