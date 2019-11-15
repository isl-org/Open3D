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

#pragma once

#include <memory>

namespace open3d {
namespace gui {

class Color;
struct Rect;
class Window;

struct BoundingBox {
    float xMin;  float xMax;
    float yMin;  float yMax;
    float zMin;  float zMax;

    BoundingBox();
    BoundingBox(float centerX, float centerY, float centerZ, float radius);
    BoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
};

/*class Transform {
public:
    Transform();  // identity
    explicit Transform(const Transform& t);

    void Translate(float x, float y, float z);
    void Rotate(float axis_x, float axis_y, float axis_z, float degrees);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
*/
class Transform;

class Renderer {
    friend class RendererScene;
    friend class RendererView;
    friend class SceneWidget;
public:
    using SceneId = int;
    using ViewId = int;
    using CameraId = int;
    using MaterialId = int;
    using VertexBufferId = int;
    using IndexBufferId = int;
    using GeometryId = int;
    using MeshId = int;
    using LightId = int;
    using IBLId = int;

    explicit Renderer(const Window& window);
    virtual ~Renderer();

    void UpdateFromDrawable();

    bool BeginFrame();
    void Render(Renderer::ViewId viewId);
    void EndFrame();

    ViewId CreateView();
    void DestroyView(ViewId viewId);
    SceneId CreateScene();
    void DestroyScene(SceneId sceneId);
    CameraId CreateCamera();
    void DestroyCamera(CameraId cameraId);

    MaterialId CreateMetal(const Color& baseColor,
                           float metallic,    // 0 (not metal) to 1 (metal)
                           float roughness,   // [0, 1]
                           float anisotropy); // [0, 1]
    MaterialId CreateNonMetal(const Color& baseColor,
                              float roughness,  // [0, 1]
                              float clearCoat,  // [0, 1]
                              float clearCoatRoughness); // [0, 1]

//    LightId CreateLight(...);
//    IBLId CreateIBL(...);

    // This is an expensive operation
    // Consumes vertices, normals, and indices.
    GeometryId CreateGeometry(std::vector<float> *vertices,
                              std::vector<float> *normals,
                              std::vector<uint32_t> *indices,
                              const BoundingBox& bbox);

    // This is (should be) a cheap operation
    MeshId CreateMesh(GeometryId geometryId, MaterialId materialId);

private:
    void* GetViewPointer(ViewId id);
    void* GetScenePointer(SceneId id);
    void* GetCameraPointer(CameraId id);
    void* GetMeshPointer(MeshId id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RendererCamera;
class RendererScene;

// Must be destroyed before Renderer (copies renderer ref)
class RendererView {
public:
    RendererView(Renderer& renderer, Renderer::ViewId id);
    ~RendererView();

    RendererScene& GetScene();
    RendererCamera& GetCamera();
    void SetClearColor(const Color& c);
    void SetViewport(const Rect& r);
    void Draw();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RendererCamera {
public:
    RendererCamera(Renderer& renderer);
    ~RendererCamera();

    Renderer::CameraId GetId() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Must be destroyed before Renderer (copies renderer ref)
class RendererScene {
public:
    RendererScene(Renderer& renderer);
    ~RendererScene();

    Renderer::SceneId GetId() const;

    void AddIBL(Renderer::IBLId iblId);
    void AddLight(Renderer::LightId lightId);
    void RemoveLight(Renderer::LightId lightId);
    void AddMesh(Renderer::MeshId meshId/*, const Transform& transform*/);
    void RemoveMesh(Renderer::MeshId meshId);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // gui
} // open3d
