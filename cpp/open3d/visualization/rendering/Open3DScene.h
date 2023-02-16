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

#pragma once

#include <map>
#include <vector>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {

namespace geometry {
class Geometry3D;
class Image;
}  // namespace geometry

namespace t {
namespace geometry {
class Geometry;
}
}  // namespace t

namespace visualization {
namespace rendering {

class Camera;
struct MaterialRecord;
struct TriangleMeshModel;

class Open3DScene {
public:
    Open3DScene(Renderer& renderer);
    ~Open3DScene();

    View* GetView() const;
    ViewHandle GetViewId() const { return view_; }
    void SetViewport(std::int32_t x,
                     std::int32_t y,
                     std::uint32_t width,
                     std::uint32_t height);

    void ShowSkybox(bool enable);
    void ShowAxes(bool enable);
    void SetBackground(const Eigen::Vector4f& color,
                       std::shared_ptr<geometry::Image> image = nullptr);
    const Eigen::Vector4f GetBackgroundColor() const;
    void ShowGroundPlane(bool enable, Scene::GroundPlane plane);

    enum class LightingProfile {
        HARD_SHADOWS,
        DARK_SHADOWS,
        MED_SHADOWS,
        SOFT_SHADOWS,
        NO_SHADOWS
    };

    void SetLighting(LightingProfile profile, const Eigen::Vector3f& sun_dir);

    /// Sets the maximum number of points before AddGeometry also adds a
    /// downsampled point cloud with number of points, used when rendering
    /// speed is important.
    void SetDownsampleThreshold(size_t n_points) {
        downsample_threshold_ = n_points;
    }
    size_t GetDownsampleThreshold() const { return downsample_threshold_; }

    void ClearGeometry();
    /// Adds a geometry with the specified name. Default visible is true.
    void AddGeometry(const std::string& name,
                     const geometry::Geometry3D* geom,
                     const MaterialRecord& mat,
                     bool add_downsampled_copy_for_fast_rendering = true);
    // Note: we can't use shared_ptr here, as we might be given something
    //       from Python, which is using unique_ptr. The pointer must live long
    //       enough to get copied to the GPU by the render thread.
    void AddGeometry(const std::string& name,
                     const t::geometry::Geometry* geom,
                     const MaterialRecord& mat,
                     bool add_downsampled_copy_for_fast_rendering = true);
    bool HasGeometry(const std::string& name) const;
    void RemoveGeometry(const std::string& name);
    /// Shows or hides the geometry with the specified name.
    void ShowGeometry(const std::string& name, bool show);
    bool GeometryIsVisible(const std::string& name);
    void SetGeometryTransform(const std::string& name,
                              const Eigen::Matrix4d& transform);
    Eigen::Matrix4d GetGeometryTransform(const std::string& name);

    void ModifyGeometryMaterial(const std::string& name,
                                const MaterialRecord& mat);
    void AddModel(const std::string& name, const TriangleMeshModel& model);

    /// Updates all geometries to use this material
    void UpdateMaterial(const MaterialRecord& mat);
    /// Updates the named model to use this material
    void UpdateModelMaterial(const std::string& name,
                             const TriangleMeshModel& model);
    std::vector<std::string> GetGeometries();

    const geometry::AxisAlignedBoundingBox& GetBoundingBox() { return bounds_; }

    enum class LOD {
        HIGH_DETAIL,  // used when rendering time is not as important
        FAST,         // used when rendering time is important, like rotating
    };
    void SetLOD(LOD lod);
    LOD GetLOD() const;

    Scene* GetScene() const;
    Camera* GetCamera() const;
    Renderer& GetRenderer() const;

private:
    struct GeometryData {
        std::string name;
        std::string fast_name;
        std::string low_name;
        bool visible;

        GeometryData() : visible(false) {}  // for STL containers
        GeometryData(const std::string& n, const std::string& fast)
            : name(n), fast_name(fast), visible(true) {}
    };

    void SetGeometryToLOD(const GeometryData&, LOD lod);

private:
    Renderer& renderer_;
    SceneHandle scene_;
    ViewHandle view_;

    Eigen::Vector4f background_color;
    LOD lod_ = LOD::HIGH_DETAIL;
    bool use_low_quality_if_available_ = false;
    bool axis_dirty_ = true;
    std::map<std::string, GeometryData> geometries_;  // name -> data
    geometry::AxisAlignedBoundingBox bounds_;
    size_t downsample_threshold_ = 6000000;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
