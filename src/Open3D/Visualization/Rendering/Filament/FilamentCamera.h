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

namespace filament
{
    class Camera;
    class Engine;
}

namespace open3d
{
namespace visualization
{

class FilamentCamera {
public:
    explicit FilamentCamera(filament::Engine& engine);
    ~FilamentCamera();

    // Updates the projection.
    void ChangeAspectRatio(float aspectRatio);

    // Sets projection parameters.  If ResizeProjection() has been called
    // before (that is, we know the size of our drawable), this will take
    // effect immediately, otherwise it will be at the next (i.e. first)
    // resize.
    void SetProjection(float near, float far, float verticalFoV);

    void LookAt(float centerX, float centerY, float centerZ,
                float eyeX, float eyeY, float eyeZ,
                float upX, float upY, float upZ);

    filament::Camera* GetNativeCamera() const { return camera; }

private:
    filament::Camera* camera = nullptr;
    filament::Engine& engine;

    float near = 0.01f;
    float far = 50.f;
    float verticalFoV = -0.0; // Invalid
    float aspectRatio = 90.f;
};

}
}