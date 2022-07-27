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

#include "open3d/t/geometry/VtkUtils.h"

#include <vtkNew.h>
#include <vtkTriangle.h>

#include <vector>

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class VtkUtilsTest : public testing::TestWithParam<bool> {};

void Compare2DTensorVtkDataArrayContent(core::Tensor& tensor,
                                        vtkDataArray* data_array) {
    using namespace t::geometry;
    ASSERT_EQ(tensor.NumDims(), 2);

    EXPECT_EQ(tensor.GetShape(0), data_array->GetNumberOfTuples());
    EXPECT_EQ(tensor.GetShape(1), data_array->GetNumberOfComponents());
    EXPECT_EQ(tensor.GetDtype().ByteSize(), data_array->GetDataTypeSize());
    EXPECT_EQ(vtkutils::DtypeToVtkType(tensor.GetDtype()),
              data_array->GetDataType());

    const void* ptr_tensor = tensor.GetDataPtr();
    const void* ptr_vtk = data_array->GetVoidPointer(0);
    size_t nbytes = tensor.NumElements() * tensor.GetDtype().ByteSize();
    EXPECT_EQ(memcmp(ptr_tensor, ptr_vtk, nbytes), 0);
}

TEST_P(VtkUtilsTest, PointCloudToVtkPolyData) {
    using namespace t::geometry;
    bool copy = GetParam();

    auto pcd = PointCloud(
            core::Tensor::Init<float>({{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}));
    auto polydata =
            vtkutils::CreateVtkPolyDataFromGeometry(pcd, {}, {}, {}, {}, copy);

    auto tensor = pcd.GetPointPositions();
    auto data_array = polydata->GetPoints()->GetData();

    void* ptr_tensor = tensor.GetDataPtr();
    void* ptr_vtk = data_array->GetVoidPointer(0);

    if (copy) {
        EXPECT_NE(ptr_tensor, ptr_vtk);
    } else {
        EXPECT_EQ(ptr_tensor, ptr_vtk);
    }
    Compare2DTensorVtkDataArrayContent(tensor, data_array);
}

TEST_P(VtkUtilsTest, TriangleMeshToVtkPolyData) {
    using namespace t::geometry;
    bool copy = GetParam();

    auto legacy_box = geometry::TriangleMesh::CreateBox();
    auto box = TriangleMesh::FromLegacy(*legacy_box);
    auto polydata =
            vtkutils::CreateVtkPolyDataFromGeometry(box, {}, {}, {}, {}, copy);

    auto tensor = box.GetVertexPositions();
    auto data_array = polydata->GetPoints()->GetData();

    void* ptr_tensor = tensor.GetDataPtr();
    void* ptr_vtk = data_array->GetVoidPointer(0);

    if (copy) {
        EXPECT_NE(ptr_tensor, ptr_vtk);
    } else {
        EXPECT_EQ(ptr_tensor, ptr_vtk);
    }
    Compare2DTensorVtkDataArrayContent(tensor, data_array);
}

TEST_P(VtkUtilsTest, VtkPolyDataToO3D) {
    using namespace t::geometry;
    bool copy = GetParam();

    TriangleMesh tmesh;
    vtkDataArray* data_array;
    {
        vtkNew<vtkPoints> points;
        points->InsertNextPoint(0.0, 0.0, 0.0);
        points->InsertNextPoint(1.0, 0.0, 0.0);
        points->InsertNextPoint(0.0, 1.0, 0.0);

        vtkNew<vtkCellArray> triangles;
        vtkNew<vtkTriangle> triangle;
        triangle->GetPointIds()->SetId(0, 0);
        triangle->GetPointIds()->SetId(1, 1);
        triangle->GetPointIds()->SetId(2, 2);
        triangles->InsertNextCell(triangle);

        vtkNew<vtkPolyData> polydata;

        polydata->SetPoints(points);
        polydata->SetPolys(triangles);

        tmesh = vtkutils::CreateTriangleMeshFromVtkPolyData(polydata, copy);

        auto tensor = tmesh.GetVertexPositions();
        data_array = polydata->GetPoints()->GetData();

        void* ptr_tensor = tensor.GetDataPtr();
        void* ptr_vtk = data_array->GetVoidPointer(0);

        if (copy) {
            EXPECT_NE(ptr_tensor, ptr_vtk);
        } else {
            EXPECT_EQ(ptr_tensor, ptr_vtk);
            EXPECT_GE(data_array->GetReferenceCount(), 2);
        }
        Compare2DTensorVtkDataArrayContent(tensor, data_array);

        // connectivity is always a copy, just check the content
        auto tensor_connectivity = tmesh.GetTriangleIndices();
        tensor_connectivity =
                tensor_connectivity
                        .Reshape({tensor_connectivity.NumElements(), 1})
                        .Contiguous();
        Compare2DTensorVtkDataArrayContent(
                tensor_connectivity,
                polydata->GetPolys()->GetConnectivityArray());
    }
    if (!copy) {
        // Only tmesh holds a reference to the data array
        EXPECT_EQ(data_array->GetReferenceCount(), 1);
    }
}

INSTANTIATE_TEST_SUITE_P(CopyBool, VtkUtilsTest, testing::Values(false, true));

}  // namespace tests
}  // namespace open3d
