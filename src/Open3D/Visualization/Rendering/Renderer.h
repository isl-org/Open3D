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
#include "RendererEntitiesMods.h"

namespace open3d {
namespace visualization {

class Scene;
class Camera;

class MaterialLoadRequest {
public:
    using ErrorCallback = std::function<void(
            const MaterialLoadRequest&, const uint8_t, const std::string&)>;
    static ErrorCallback defaultErrorHandler;

    MaterialLoadRequest(const void* materialData,
                        size_t dataSize,
                        ErrorCallback errorCallback = defaultErrorHandler);
    explicit MaterialLoadRequest(
            const char* path,
            ErrorCallback errorCallback = defaultErrorHandler);

    const void* materialData;
    const size_t dataSize;
    const std::string path;
    ErrorCallback errorCallback;
};

class Renderer {
public:
    virtual ~Renderer() = default;

    virtual SceneHandle CreateScene() = 0;
    virtual Scene* GetScene(const SceneHandle& id) const = 0;
    virtual void DestroyScene(const SceneHandle& id) = 0;

    virtual void BeginFrame() = 0;
    virtual void Draw() = 0;
    virtual void EndFrame() = 0;

    // Loads material from its data
    virtual MaterialHandle AddMaterial(const void* materialData,
                                       size_t dataSize) = 0;
    virtual MaterialHandle AddMaterial(const MaterialLoadRequest& request) = 0;
    virtual MaterialModifier& ModifyMaterial(const MaterialHandle& id) = 0;
    virtual MaterialModifier& ModifyMaterial(
            const MaterialInstanceHandle& id) = 0;
};

}
}
