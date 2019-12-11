// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "FilamentCamera.h"

#include <cstddef>

#include <filament/Camera.h>
#include <filament/Engine.h>

namespace open3d
{
namespace visualization
{

FilamentCamera::FilamentCamera(filament::Engine& aEngine)
    : engine(aEngine)
{
    camera = engine.createCamera();
}

FilamentCamera::~FilamentCamera()
{
    engine.destroy(camera);
}

void FilamentCamera::SetProjection(double fov, double aspect, double near, double far, eFovType fovType)
{
    if (aspect > 0.0) {
        filament::Camera::Fov dir = (fovType == eFovType::HORIZONTAL_FOV)
                ? filament::Camera::Fov::HORIZONTAL
                : filament::Camera::Fov::VERTICAL;

        camera->setProjection(fov, aspect, near, far, dir);
    }
}

void FilamentCamera::SetProjection(eProjection projection, double left, double right, double bottom, double top, double near, double far)
{
    filament::Camera::Projection proj = (projection == eProjection::ORTHO)
            ? filament::Camera::Projection::ORTHO
            : filament::Camera::Projection::PERSPECTIVE;

    camera->setProjection(proj, left, right, bottom, top, near, far);
}

void FilamentCamera::LookAt(const Eigen::Vector3f& center, const Eigen::Vector3f& eye, const Eigen::Vector3f& up)
{
    camera->lookAt({ eye.x(), eye.y(), eye.z() },
                   { center.x(), center.y(), center.z()},
                   { up.x(), up.y(), up.z() });
}

Eigen::Vector3f FilamentCamera::GetPosition()
{
    auto camPos = camera->getPosition();
    return {camPos.x, camPos.y, camPos.z};
}

Eigen::Vector3f FilamentCamera::GetForwardVector()
{
    auto forward = camera->getForwardVector();
    return {forward.x, forward.y, forward.z};
}

Eigen::Vector3f FilamentCamera::GetLeftVector()
{
    auto left = camera->getLeftVector();
    return {left.x, left.y, left.z};
}

Eigen::Vector3f FilamentCamera::GetUpVector()
{
    auto up = camera->getUpVector();
    return {up.x, up.y, up.z};
}


}
}