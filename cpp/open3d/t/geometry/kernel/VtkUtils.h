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
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace vtkutils {

/// Creates a vtkPolyData object from a point cloud or triangle mesh.
/// The returned vtkPolyData object may directly use the memory of the tensors
/// stored inside the Geometry object. Therefore, the Geometry object must be
/// kept alive until the returned vtkPolyData object is deleted.
/// \param geometry Open3D geometry object, e.g., a TriangleMesh.
/// \param copy If true always create a copy of the data.
vtkSmartPointer<vtkPolyData> CreateVtkPolyDataFromGeometry(
        const Geometry& geometry, bool copy=false);

/// Creates a triangle mesh from a vtkPolyData object.
/// The returned TriangleMesh may directly use the memory of the data arrays in
/// the vtkPolyData object.
/// The returned TriangleMesh will hold references to the arrays and it is not 
/// necessary to keep other references to the vtkPolyData object or its arrays
/// alive.
/// \param polyData Input polyData object.
/// \param copy If true always create a copy of the data.
TriangleMesh CreateTriangleMeshFromVtkPolyData(vtkPolyData* polyData, bool copy=false);

}  // namespace vtkutils
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
