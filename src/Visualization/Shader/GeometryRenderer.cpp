// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "GeometryRenderer.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/LineSet.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Core/Geometry/Image.h>
#include <Visualization/Utility/SelectionPolygon.h>

namespace three{

namespace glsl {

bool PointCloudRenderer::Render(const RenderOption &option,
		const ViewControl &view)
{
	const auto &pointcloud = (const PointCloud &)(*geometry_ptr_);
	bool success = true;
	
	if (pointcloud.HasNormals()) {
		success &= phong_point_shader_.Render(pointcloud, option, view);
		if (option.point_show_normal_) {
			success &= simpleblack_normal_shader_.Render(pointcloud, option,
					view);
		}
	} else {
		success &= simple_point_shader_.Render(pointcloud, option, view);
	}
	return success;
}

bool PointCloudRenderer::AddGeometry(
		std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_POINTCLOUD) {
		return false;
	}
	const auto &pointcloud = (const PointCloud &)(*geometry_ptr);
	if (pointcloud.HasPoints() == false) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool PointCloudRenderer::UpdateGeometry()
{
	simple_point_shader_.InvalidateGeometry();
	phong_point_shader_.InvalidateGeometry();
	simpleblack_normal_shader_.InvalidateGeometry();
	return true;
}

bool LineSetRenderer::Render(const RenderOption &option,
		const ViewControl &view)
{
	return simple_lineset_shader_.Render(*geometry_ptr_, option, view);
}

bool LineSetRenderer::AddGeometry(std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_LINESET) {
		return false;
	}
	const auto &lineset = (const LineSet &)(*geometry_ptr);
	if (lineset.HasLines() == false) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool LineSetRenderer::UpdateGeometry()
{
	simple_lineset_shader_.InvalidateGeometry();
	return true;
}

bool TriangleMeshRenderer::Render(const RenderOption &option,
		const ViewControl &view)
{
	const auto &mesh = (const TriangleMesh &)(*geometry_ptr_);
	bool success = true;
	
	if (mesh.HasTriangleNormals() && mesh.HasVertexNormals()) {
		success &= phong_mesh_shader_.Render(mesh, option, view);
	} else {
		success &= simple_mesh_shader_.Render(mesh, option, view);
	}
	
	if (option.mesh_show_wireframe_) {
		success &= simpleblack_wireframe_shader_.Render(mesh, option, view);
	}
	return success;
}

bool TriangleMeshRenderer::AddGeometry(
		std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH) {
		return false;
	}
	const auto &mesh = (const TriangleMesh &)(*geometry_ptr);
	if (mesh.HasTriangles() == false) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool TriangleMeshRenderer::UpdateGeometry()
{
	simple_mesh_shader_.InvalidateGeometry();
	phong_mesh_shader_.InvalidateGeometry();
	simpleblack_wireframe_shader_.InvalidateGeometry();
	return true;
}

bool ImageRenderer::Render(const RenderOption &option, const ViewControl &view)
{
	return image_shader_.Render(*geometry_ptr_, option, view);
}

bool ImageRenderer::AddGeometry(std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_IMAGE) {
		return false;
	}
	const auto &image = (const Image &)(*geometry_ptr);
	if (image.HasData() == false) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool ImageRenderer::UpdateGeometry()
{
	image_shader_.InvalidateGeometry();
	return true;
}

bool CoordinateFrameRenderer::Render(const RenderOption &option,
		const ViewControl &view)
{
	if (option.show_coordinate_frame_ == false) {
		return true;
	}
	const auto &mesh = (const TriangleMesh &)(*geometry_ptr_);
	return phong_shader_.Render(mesh, option, view);
}

bool CoordinateFrameRenderer::AddGeometry(
		std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_TRIANGLEMESH) {
		return false;
	}
	const auto &mesh = (const TriangleMesh &)(*geometry_ptr);
	if (mesh.HasTriangles() == false) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool CoordinateFrameRenderer::UpdateGeometry()
{
	phong_shader_.InvalidateGeometry();
	return true;
}

bool SelectionPolygonRenderer::Render(const RenderOption &option,
		const ViewControl &view)
{
	const auto &polygon = (const SelectionPolygon &)(*geometry_ptr_);
	if (polygon.IsEmpty()) {
		return true;
	}
	if (simple2d_shader_.Render(polygon, option, view) == false) {
		return false;
	}
	if (polygon.polygon_interior_mask_.IsEmpty()) {
		return true;
	}
	return image_mask_shader_.Render(polygon.polygon_interior_mask_, option,
			view);
}

bool SelectionPolygonRenderer::AddGeometry(
		std::shared_ptr<const Geometry> geometry_ptr)
{
	if (geometry_ptr->GetGeometryType() != Geometry::GEOMETRY_UNSPECIFIED) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool SelectionPolygonRenderer::UpdateGeometry()
{
	simple2d_shader_.InvalidateGeometry();
	image_mask_shader_.InvalidateGeometry();
	return true;
}

}	// namespace three::glsl

}	// namespace three
