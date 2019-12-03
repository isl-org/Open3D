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
#include "RendererEntitiesMods.h"

namespace open3d {

namespace geometry
{
    class Geometry3D;
}

namespace visualization {

class AbstractRenderInterface {
public:
    virtual ~AbstractRenderInterface() = default;

    virtual void Draw() = 0;

    // If MaterialInstanceHandle::kBad passed, then default material is used
    virtual GeometryHandle AddGeometry(const geometry::Geometry3D& geometry, const MaterialInstanceHandle& materialId) = 0;
    // High perfomance cost (probably), since Open3D geometry has no Transform concept
    // and entire geometry structure is modified (vertices and normals)
    // upon Translate/Rotate/Scale calls
    virtual void UpdateGeometry(const GeometryHandle& id, const geometry::Geometry3D& geometry) = 0;
    virtual void RemoveGeometry(const GeometryHandle& geometryId) = 0;

    virtual LightHandle AddLight(const LightDescription& descr) = 0;
    //virtual LightFluentInterface ModifyLight(const REHandle<eEntityType::Light>& id) = 0;
    virtual void RemoveLight(const LightHandle& id) = 0;

    virtual CameraHandle AddCamera(const CameraDescription& descr) = 0;
    // It is safe to cache interface
    //virtual CameraFluentInterface ModifyCamera(const ruid<eEntityType::Camera>& id) = 0;
    virtual void RemoveCamera(const CameraHandle& id) = 0;

    // Loads material from its data
    virtual MaterialHandle AddMaterial(const void* materialData, size_t dataSize) = 0;
    virtual MaterialModifier& ModifyMaterial(const MaterialHandle& id) = 0;
    virtual MaterialModifier& ModifyMaterial(const MaterialInstanceHandle& id) = 0;
    virtual void AssignMaterial(const GeometryHandle& geometryId, const MaterialInstanceHandle& materialId) = 0;
};

extern AbstractRenderInterface* TheRenderer;

}
}
