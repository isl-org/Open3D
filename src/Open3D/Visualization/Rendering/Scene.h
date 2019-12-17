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

#pragma once

#include "RendererHandle.h"
#include "RendererStructs.h"

#include <Eigen/Geometry>

namespace open3d {

namespace geometry {
class Geometry3D;
}

namespace visualization {

class View;

// Contains renderable objects like geometry and lights
// Can have multiple views
class Scene {
public:
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;

    virtual ~Scene() {}

    virtual ViewHandle AddView(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) = 0;
    virtual View* GetView(const ViewHandle& viewId) const = 0;
    virtual void SetViewActive(const ViewHandle& viewId, bool isActive) = 0;
    virtual void RemoveView(const ViewHandle& viewId) = 0;

    virtual GeometryHandle AddGeometry(
            const geometry::Geometry3D& geometry,
            const MaterialInstanceHandle& materialId) = 0;
    virtual void RemoveGeometry(const GeometryHandle& geometryId) = 0;

    virtual LightHandle AddLight(const LightDescription& descr) = 0;
    // virtual LightFluentInterface ModifyLight(const REHandle<EntityType::Light>& id) = 0;
    virtual void RemoveLight(const LightHandle& id) = 0;

    virtual void SetEntityTransform(const REHandle_abstract& entityId, const Transform& transform) = 0;
    virtual Transform GetEntityTransform(const REHandle_abstract& entityId) const = 0;
};

}
}