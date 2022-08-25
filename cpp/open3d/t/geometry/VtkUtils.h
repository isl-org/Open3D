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

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {
namespace vtkutils {

/// Returns the corresponding vtk data type for core::Dtype
/// Logs an error if no conversion exists.
int DtypeToVtkType(const core::Dtype& dtype);

/// Creates a vtkPolyData object from a point cloud or triangle mesh.
/// The returned vtkPolyData object may directly use the memory of the tensors
/// stored inside the Geometry object. Therefore, the Geometry object must be
/// kept alive until the returned vtkPolyData object is deleted.
/// \param geometry Open3D geometry object, e.g., a TriangleMesh.
/// \param copy If true always create a copy of the data.
/// \param point_attr_include A set of keys to select which point/vertex
///  attributes should be added. Note that the primary key may be included and
///  will silently be ignored.
/// \param face_attr_include A set of keys to select
///  which face attributes should be added. Note that the primary key may be
///  included and will silently be ignored.
/// \param point_attr_exclude A set of keys for which point/vertex attributes
///  will not be added to the vtkPolyData. The exclusion set has precedence over
///  the included keys.
/// \param face_attr_exclude A set of keys for which face attributes will not be
/// added
///  to the vtkPolyData. The exclusion set has precedence over the included
///  keys.
vtkSmartPointer<vtkPolyData> CreateVtkPolyDataFromGeometry(
        const Geometry& geometry,
        const std::unordered_set<std::string>& point_attr_include,
        const std::unordered_set<std::string>& face_attr_include,
        const std::unordered_set<std::string>& point_attr_exclude = {},
        const std::unordered_set<std::string>& face_attr_exclude = {},
        bool copy = false);

/// Creates a triangle mesh from a vtkPolyData object.
/// The returned TriangleMesh may directly use the memory of the data arrays in
/// the vtkPolyData object.
/// The returned TriangleMesh will hold references to the arrays and it is not
/// necessary to keep other references to the vtkPolyData object or its arrays
/// alive.
/// \param polydata Input polyData object.
/// \param copy If true always create a copy of the data.
TriangleMesh CreateTriangleMeshFromVtkPolyData(vtkPolyData* polydata,
                                               bool copy = false);

/// Creates a LineSet a vtkPolyData object.
/// The returned LineSet may directly use the memory of the data arrays in
/// the vtkPolyData object.
/// The returned LineSet will hold references to the arrays and it is not
/// necessary to keep other references to the vtkPolyData object or its arrays
/// alive.
/// \param polydata Input polyData object.
/// \param copy If true always create a copy of the data.
LineSet CreateLineSetFromVtkPolyData(vtkPolyData* polydata, bool copy = false);

/// Sweeps the geometry rotationally about an axis.
/// \param geometry Open3D geometry object, e.g., a TriangleMesh.
/// \param angle The rotation angle in degree.
/// \param axis The rotation axis.
/// \param resolution The resolution defines the number of intermediate
/// sweeps about the rotation axis.
/// \param translation The translation along the rotation axis.
/// \param capping If true adds caps to the mesh.
/// \return A triangle mesh with the result of the sweep operation.
TriangleMesh ExtrudeRotationTriangleMesh(const Geometry& geometry,
                                         double angle,
                                         const core::Tensor& axis,
                                         int resolution = 16,
                                         double translation = 0.0,
                                         bool capping = true);

/// Sweeps the geometry rotationally about an axis.
/// \param pointcloud A point cloud.
/// \param angle The rotation angle in degree.
/// \param axis The rotation axis.
/// \param resolution The resolution defines the number of intermediate
/// sweeps about the rotation axis.
/// \param translation The translation along the rotation axis.
/// \param capping If true adds caps to the mesh.
/// \return A line set with the result of the sweep operation.
LineSet ExtrudeRotationLineSet(const PointCloud& pointcloud,
                               double angle,
                               const core::Tensor& axis,
                               int resolution = 16,
                               double translation = 0.0,
                               bool capping = true);

/// Sweeps the geometry along a direction vector.
/// \param geometry Open3D geometry object, e.g., a TriangleMesh.
/// \param vector The direction vector.
/// \param scale Scalar factor which essentially scales the direction vector.
/// \param capping If true adds caps to the mesh.
/// \return A triangle mesh with the result of the sweep operation.
TriangleMesh ExtrudeLinearTriangleMesh(const Geometry& geometry,
                                       const core::Tensor& vector,
                                       double scale,
                                       bool capping);

/// Sweeps the geometry along a direction vector.
/// \param pointcloud A point cloud.
/// \param vector The direction vector.
/// \param scale Scalar factor which essentially scales the direction vector.
/// \param capping If true adds caps to the mesh.
/// \return A triangle mesh with the result of the sweep operation.
LineSet ExtrudeLinearLineSet(const PointCloud& pointcloud,
                             const core::Tensor& vector,
                             double scale,
                             bool capping);

}  // namespace vtkutils
}  // namespace geometry
}  // namespace t
}  // namespace open3d
