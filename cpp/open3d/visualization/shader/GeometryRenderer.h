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

#include "open3d/geometry/Geometry.h"
#include "open3d/visualization/shader/ImageMaskShader.h"
#include "open3d/visualization/shader/ImageShader.h"
#include "open3d/visualization/shader/NormalShader.h"
#include "open3d/visualization/shader/PhongShader.h"
#include "open3d/visualization/shader/PickingShader.h"
#include "open3d/visualization/shader/RGBDImageShader.h"
#include "open3d/visualization/shader/Simple2DShader.h"
#include "open3d/visualization/shader/SimpleBlackShader.h"
#include "open3d/visualization/shader/SimpleShader.h"
#include "open3d/visualization/shader/TexturePhongShader.h"
#include "open3d/visualization/shader/TextureSimpleShader.h"

namespace open3d {
namespace visualization {

namespace glsl {

class GeometryRenderer {
public:
    virtual ~GeometryRenderer() {}

public:
    virtual bool Render(const RenderOption &option,
                        const ViewControl &view) = 0;

    /// Function to add geometry to the renderer
    /// 1. After calling the function, the renderer owns the geometry object.
    /// 2. This function returns FALSE if the geometry type is not matched to
    /// the renderer.
    /// 3. If an added geometry is changed, programmer must call
    /// UpdateGeometry() to notify the renderer.
    virtual bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) = 0;

    /// Function to update geometry
    /// Programmer must call this function to notify a change of the geometry
    virtual bool UpdateGeometry() = 0;

    bool HasGeometry() const { return bool(geometry_ptr_); }
    std::shared_ptr<const geometry::Geometry> GetGeometry() const {
        return geometry_ptr_;
    }

    bool HasGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) const {
        return geometry_ptr_ == geometry_ptr;
    }

    bool IsVisible() const { return is_visible_; }
    void SetVisible(bool visible) { is_visible_ = visible; };

protected:
    std::shared_ptr<const geometry::Geometry> geometry_ptr_;
    bool is_visible_ = true;
};

class PointCloudRenderer : public GeometryRenderer {
public:
    ~PointCloudRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForPointCloud simple_point_shader_;
    PhongShaderForPointCloud phong_point_shader_;
    NormalShaderForPointCloud normal_point_shader_;
    SimpleBlackShaderForPointCloudNormal simpleblack_normal_shader_;
};

class PointCloudPickingRenderer : public GeometryRenderer {
public:
    ~PointCloudPickingRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    PickingShaderForPointCloud picking_shader_;
};

class LineSetRenderer : public GeometryRenderer {
public:
    ~LineSetRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForLineSet simple_lineset_shader_;
};

class TetraMeshRenderer : public GeometryRenderer {
public:
    ~TetraMeshRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForTetraMesh simple_tetramesh_shader_;
};

class OrientedBoundingBoxRenderer : public GeometryRenderer {
public:
    ~OrientedBoundingBoxRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForOrientedBoundingBox simple_oriented_bounding_box_shader_;
};

class AxisAlignedBoundingBoxRenderer : public GeometryRenderer {
public:
    ~AxisAlignedBoundingBoxRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForAxisAlignedBoundingBox
            simple_axis_aligned_bounding_box_shader_;
};

class TriangleMeshRenderer : public GeometryRenderer {
public:
    ~TriangleMeshRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForTriangleMesh simple_mesh_shader_;
    TextureSimpleShaderForTriangleMesh texture_simple_mesh_shader_;
    PhongShaderForTriangleMesh phong_mesh_shader_;
    TexturePhongShaderForTriangleMesh texture_phong_mesh_shader_;
    NormalShaderForTriangleMesh normal_mesh_shader_;
    SimpleBlackShaderForTriangleMeshWireFrame simpleblack_wireframe_shader_;
};

class VoxelGridRenderer : public GeometryRenderer {
public:
    ~VoxelGridRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForVoxelGridLine simple_shader_for_voxel_grid_line_;
    SimpleShaderForVoxelGridFace simple_shader_for_voxel_grid_face_;
};

class OctreeRenderer : public GeometryRenderer {
public:
    ~OctreeRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForOctreeLine simple_shader_for_octree_line_;
    SimpleShaderForOctreeFace simple_shader_for_octree_face_;
};

class PlanarPatchRenderer : public GeometryRenderer {
public:
    ~PlanarPatchRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForPlanarPatch simple_shader_for_planar_patch_;
    SimpleBlackShaderForPlanarPatchNormal simpleblack_normal_shader_;
};

class ImageRenderer : public GeometryRenderer {
public:
    ~ImageRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    ImageShaderForImage image_shader_;
};

class RGBDImageRenderer : public GeometryRenderer {
public:
    ~RGBDImageRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    RGBDImageShaderForImage rgbd_image_shader_;
};

class CoordinateFrameRenderer : public GeometryRenderer {
public:
    ~CoordinateFrameRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    PhongShaderForTriangleMesh phong_shader_;
};

class SelectionPolygonRenderer : public GeometryRenderer {
public:
    ~SelectionPolygonRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    Simple2DShaderForSelectionPolygon simple2d_shader_;
    ImageMaskShaderForImage image_mask_shader_;
};

class PointCloudPickerRenderer : public GeometryRenderer {
public:
    ~PointCloudPickerRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    PhongShaderForTriangleMesh phong_shader_;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
