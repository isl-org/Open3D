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

#include "open3d/visualization/shader/GeometryRenderer.h"

#include "open3d/geometry/Image.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/utility/PointCloudPicker.h"
#include "open3d/visualization/utility/SelectionPolygon.h"
#include "open3d/visualization/visualizer/RenderOptionWithEditing.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool PointCloudRenderer::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    const auto &pointcloud = (const geometry::PointCloud &)(*geometry_ptr_);
    bool success = true;
    if (pointcloud.HasNormals()) {
        if (option.point_color_option_ ==
            RenderOption::PointColorOption::Normal) {
            success &= normal_point_shader_.Render(pointcloud, option, view);
        } else {
            success &= phong_point_shader_.Render(pointcloud, option, view);
        }
        if (option.point_show_normal_) {
            success &=
                    simpleblack_normal_shader_.Render(pointcloud, option, view);
        }
    } else {
        success &= simple_point_shader_.Render(pointcloud, option, view);
    }
    return success;
}

bool PointCloudRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudRenderer::UpdateGeometry() {
    simple_point_shader_.InvalidateGeometry();
    phong_point_shader_.InvalidateGeometry();
    normal_point_shader_.InvalidateGeometry();
    simpleblack_normal_shader_.InvalidateGeometry();
    return true;
}

bool PointCloudPickingRenderer::Render(const RenderOption &option,
                                       const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    const auto &pointcloud = (const geometry::PointCloud &)(*geometry_ptr_);
    return picking_shader_.Render(pointcloud, option, view);
}

bool PointCloudPickingRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudPickingRenderer::UpdateGeometry() {
    picking_shader_.InvalidateGeometry();
    return true;
}

bool VoxelGridRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    if (option.mesh_show_wireframe_) {
        return simple_shader_for_voxel_grid_line_.Render(*geometry_ptr_, option,
                                                         view);
    } else {
        return simple_shader_for_voxel_grid_face_.Render(*geometry_ptr_, option,
                                                         view);
    }
}

bool VoxelGridRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool VoxelGridRenderer::UpdateGeometry() {
    simple_shader_for_voxel_grid_line_.InvalidateGeometry();
    simple_shader_for_voxel_grid_face_.InvalidateGeometry();
    return true;
}

bool OctreeRenderer::Render(const RenderOption &option,
                            const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    if (option.mesh_show_wireframe_) {
        return simple_shader_for_octree_line_.Render(*geometry_ptr_, option,
                                                     view);
    } else {
        bool rc = simple_shader_for_octree_face_.Render(*geometry_ptr_, option,
                                                        view);
        rc &= simple_shader_for_octree_line_.Render(*geometry_ptr_, option,
                                                    view);
        return rc;
    }
}

bool OctreeRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Octree) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool OctreeRenderer::UpdateGeometry() {
    simple_shader_for_octree_line_.InvalidateGeometry();
    simple_shader_for_octree_face_.InvalidateGeometry();
    return true;
}

bool LineSetRenderer::Render(const RenderOption &option,
                             const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return simple_lineset_shader_.Render(*geometry_ptr_, option, view);
}

bool LineSetRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool LineSetRenderer::UpdateGeometry() {
    simple_lineset_shader_.InvalidateGeometry();
    return true;
}

bool TetraMeshRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return simple_tetramesh_shader_.Render(*geometry_ptr_, option, view);
}

bool TetraMeshRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TetraMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TetraMeshRenderer::UpdateGeometry() {
    simple_tetramesh_shader_.InvalidateGeometry();
    return true;
}

bool OrientedBoundingBoxRenderer::Render(const RenderOption &option,
                                         const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return simple_oriented_bounding_box_shader_.Render(*geometry_ptr_, option,
                                                       view);
}

bool OrientedBoundingBoxRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::OrientedBoundingBox) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool OrientedBoundingBoxRenderer::UpdateGeometry() {
    simple_oriented_bounding_box_shader_.InvalidateGeometry();
    return true;
}

bool AxisAlignedBoundingBoxRenderer::Render(const RenderOption &option,
                                            const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return simple_axis_aligned_bounding_box_shader_.Render(*geometry_ptr_,
                                                           option, view);
}

bool AxisAlignedBoundingBoxRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool AxisAlignedBoundingBoxRenderer::UpdateGeometry() {
    simple_axis_aligned_bounding_box_shader_.InvalidateGeometry();
    return true;
}

bool PlanarPatchRenderer::Render(const RenderOption &option,
                                            const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    // const auto &patch = (const geometry::PlanarPatch &)(*geometry_ptr_);
    bool success = true;
    // if (pointcloud.HasNormals()) {
    //     if (option.point_color_option_ ==
    //         RenderOption::PointColorOption::Normal) {
    //         success &= normal_point_shader_.Render(pointcloud, option, view);
    //     } else {
    //         success &= phong_point_shader_.Render(pointcloud, option, view);
    //     }
        if (option.point_show_normal_) {
            success &=
                    simpleblack_normal_shader_.Render(*geometry_ptr_, option, view);
        }
    // } else {
        success &= simple_shader_for_planar_patch_.Render(*geometry_ptr_, option, view);
    // }
    return success;
}

bool PlanarPatchRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::PlanarPatch) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PlanarPatchRenderer::UpdateGeometry() {
    simple_shader_for_planar_patch_.InvalidateGeometry();
    return true;
}

bool TriangleMeshRenderer::Render(const RenderOption &option,
                                  const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    const auto &mesh = (const geometry::TriangleMesh &)(*geometry_ptr_);
    bool success = true;
    if (mesh.HasTriangleNormals() && mesh.HasVertexNormals()) {
        if (option.mesh_color_option_ ==
            RenderOption::MeshColorOption::Normal) {
            success &= normal_mesh_shader_.Render(mesh, option, view);
        } else if (option.mesh_color_option_ ==
                           RenderOption::MeshColorOption::Color &&
                   mesh.HasTriangleUvs() && mesh.HasTextures()) {
            success &= texture_phong_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= phong_mesh_shader_.Render(mesh, option, view);
        }
    } else {  // if normals are not ready
        if (option.mesh_color_option_ == RenderOption::MeshColorOption::Color &&
            mesh.HasTriangleUvs() && mesh.HasTextures()) {
            success &= texture_simple_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= simple_mesh_shader_.Render(mesh, option, view);
        }
    }
    if (option.mesh_show_wireframe_) {
        success &= simpleblack_wireframe_shader_.Render(mesh, option, view);
    }
    return success;
}

bool TriangleMeshRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry_ptr->GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshRenderer::UpdateGeometry() {
    simple_mesh_shader_.InvalidateGeometry();
    texture_simple_mesh_shader_.InvalidateGeometry();
    phong_mesh_shader_.InvalidateGeometry();
    texture_phong_mesh_shader_.InvalidateGeometry();
    normal_mesh_shader_.InvalidateGeometry();
    simpleblack_wireframe_shader_.InvalidateGeometry();
    return true;
}

bool ImageRenderer::Render(const RenderOption &option,
                           const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return image_shader_.Render(*geometry_ptr_, option, view);
}

bool ImageRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Image) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool ImageRenderer::UpdateGeometry() {
    image_shader_.InvalidateGeometry();
    return true;
}

bool RGBDImageRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    return rgbd_image_shader_.Render(*geometry_ptr_, option, view);
}

bool RGBDImageRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::RGBDImage) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool RGBDImageRenderer::UpdateGeometry() {
    rgbd_image_shader_.InvalidateGeometry();
    return true;
}

bool CoordinateFrameRenderer::Render(const RenderOption &option,
                                     const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    if (!option.show_coordinate_frame_) return true;
    const auto &mesh = (const geometry::TriangleMesh &)(*geometry_ptr_);
    return phong_shader_.Render(mesh, option, view);
}

bool CoordinateFrameRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry_ptr->GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool CoordinateFrameRenderer::UpdateGeometry() {
    phong_shader_.InvalidateGeometry();
    return true;
}

bool SelectionPolygonRenderer::Render(const RenderOption &option,
                                      const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    const auto &polygon = (const SelectionPolygon &)(*geometry_ptr_);
    if (polygon.IsEmpty()) return true;
    if (!simple2d_shader_.Render(polygon, option, view)) return false;
    if (polygon.polygon_interior_mask_.IsEmpty()) return true;
    return image_mask_shader_.Render(polygon.polygon_interior_mask_, option,
                                     view);
}

bool SelectionPolygonRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Unspecified) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool SelectionPolygonRenderer::UpdateGeometry() {
    simple2d_shader_.InvalidateGeometry();
    image_mask_shader_.InvalidateGeometry();
    return true;
}

bool PointCloudPickerRenderer::Render(const RenderOption &option,
                                      const ViewControl &view) {
    const int NUM_OF_COLOR_PALETTE = 5;
    Eigen::Vector3d color_palette[NUM_OF_COLOR_PALETTE] = {
            Eigen::Vector3d(255, 180, 0) / 255.0,
            Eigen::Vector3d(0, 166, 237) / 255.0,
            Eigen::Vector3d(246, 81, 29) / 255.0,
            Eigen::Vector3d(127, 184, 0) / 255.0,
            Eigen::Vector3d(13, 44, 84) / 255.0,
    };
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    const auto &picker = (const PointCloudPicker &)(*geometry_ptr_);
    const auto &pointcloud =
            (const geometry::PointCloud &)(*picker.pointcloud_ptr_);
    const auto &_option = (const RenderOptionWithEditing &)option;
    for (size_t i = 0; i < picker.picked_indices_.size(); i++) {
        size_t index = picker.picked_indices_[i];
        if (index < pointcloud.points_.size()) {
            auto sphere = geometry::TriangleMesh::CreateSphere(
                    view.GetBoundingBox().GetMaxExtent() *
                    _option.pointcloud_picker_sphere_size_);
            sphere->ComputeVertexNormals();
            sphere->vertex_colors_.clear();
            sphere->vertex_colors_.resize(
                    sphere->vertices_.size(),
                    color_palette[i % NUM_OF_COLOR_PALETTE]);
            Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
            trans.block<3, 1>(0, 3) = pointcloud.points_[index];
            sphere->Transform(trans);
            phong_shader_.InvalidateGeometry();
            if (!phong_shader_.Render(*sphere, option, view)) {
                return false;
            }
        }
    }
    return true;
}

bool PointCloudPickerRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Unspecified) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudPickerRenderer::UpdateGeometry() {
    // The geometry is updated on-the-fly
    // It is always in an invalidated status
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
