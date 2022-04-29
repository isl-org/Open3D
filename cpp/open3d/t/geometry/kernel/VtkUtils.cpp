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

#include "open3d/t/geometry/kernel/VtkUtils.h"

#include <vtkArrayDispatch.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace vtkutils {

vtkSmartPointer<vtkPoints> CreateVtkPointsFromTensor(
        const core::Tensor& tensor) {
    core::AssertTensorShape(tensor, {core::None, 3});
    core::AssertTensorDtypes(tensor, {core::Float32, core::Float64});

    auto tensor_cpu = tensor.To(core::Device()).Contiguous();
    const size_t num_points = tensor.GetLength();

    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();

    if (tensor.GetDtype() == core::Float32) {
        pts->SetDataTypeToFloat();
        pts->SetNumberOfPoints(num_points);
        const float* data = tensor_cpu.GetDataPtr<float>();
        for (size_t i = 0; i < num_points; ++i) {
            pts->SetPoint(i, data);
            data += 3;
        }
    } else {
        pts->SetDataTypeToDouble();
        pts->SetNumberOfPoints(num_points);
        const double* data = tensor_cpu.GetDataPtr<double>();
        for (size_t i = 0; i < num_points; ++i) {
            pts->SetPoint(i, data);
            data += 3;
        }
    }
    return pts;
}

vtkSmartPointer<vtkCellArray> CreateVtkCellArrayFromTensor(
        const core::Tensor& tensor) {
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    core::AssertTensorShape(tensor, {core::None, core::None});
    core::AssertTensorDtypes(tensor, {core::Int32, core::Int64});

    auto tensor_cpu = tensor.To(core::Device()).Contiguous();
    const size_t num_cells = tensor.GetLength();
    const int cell_size = tensor.GetShape()[1];

    if (tensor.GetDtype() == core::Int64) {
        const int64_t* data = tensor_cpu.GetDataPtr<int64_t>();
        for (size_t cell_i = 0; cell_i < num_cells; ++cell_i) {
            cells->InsertNextCell(cell_size);
            for (int j = 0; j < cell_size; ++j) cells->InsertCellPoint(data[j]);
            data += cell_size;
        }
    } else {
        const int32_t* data = tensor_cpu.GetDataPtr<int32_t>();
        for (size_t cell_i = 0; cell_i < num_cells; ++cell_i) {
            cells->InsertNextCell(cell_size);
            for (int j = 0; j < cell_size; ++j) cells->InsertCellPoint(data[j]);
            data += cell_size;
        }
    }

    return cells;
}

core::Tensor CreateTensorFromVtkPoints(vtkPoints* points) {
    core::Tensor result;
    int64_t num_points = points->GetNumberOfPoints();
    if (points->GetDataType() == VTK_FLOAT) {
        result = core::Tensor({num_points, 3}, core::Float32);
        float* data = result.GetDataPtr<float>();
        for (int64_t i = 0; i < num_points; ++i) {
            double p[3];
            points->GetPoint(i, p);
            data[0] = p[0];
            data[1] = p[1];
            data[2] = p[2];
            data += 3;
        }
    } else if (points->GetDataType() == VTK_DOUBLE) {
        result = core::Tensor({num_points, 3}, core::Float64);
        double* data = result.GetDataPtr<double>();
        for (int64_t i = 0; i < num_points; ++i) {
            points->GetPoint(i, data);
            data += 3;
        }
    } else {
        utility::LogError(
                "Unexpected data type for vtkPoints. Expected float or double "
                "but got {}",
                points->GetDataType());
    }
    return result;
}

// map types used by vtk to compatible Tensor types
template <class T>
struct VtkToTensorType {
    typedef T TensorType;
};

template <>
struct VtkToTensorType<long long> {
    typedef int64_t TensorType;
};

struct CreateTensorFromVtkDataArrayWorker {
    core::Tensor result;

    template <class TArray>
    void operator()(TArray* array) {
        typedef typename TArray::ValueType T;
        typedef typename VtkToTensorType<T>::TensorType TTensor;
        auto dtype = core::Dtype::FromType<TTensor>();
        int64_t length = array->GetNumberOfTuples();
        int64_t num_components = array->GetNumberOfComponents();

        result = core::Tensor({length, num_components}, dtype);

        TTensor* data = result.GetDataPtr<TTensor>();
        for (int64_t i = 0; i < length; ++i) {
            for (int64_t j = 0; j < num_components; ++j) {
                *data = array->GetTypedComponent(i, j);
                ++data;
            }
        }
    }
};

core::Tensor CreateTensorFromVtkDataArray(vtkDataArray* array) {
    CreateTensorFromVtkDataArrayWorker worker;
    typedef vtkTypeList_Create_7(float, double, int32_t, int64_t, uint32_t,
                                 uint64_t, long long) ArrayTypes;
    vtkArrayDispatch::DispatchByValueType<ArrayTypes>::Execute(array, worker);
    return worker.result;
}

core::Tensor CreateTensorFromVtkCellArray(vtkCellArray* cells) {
    auto num_cells = cells->GetNumberOfCells();
    int cell_size = 0;
    if (num_cells) {
        cell_size = cells->GetCellSize(0);
    }
    for (vtkIdType i = 1; i < num_cells; ++i) {
        if (cells->GetCellSize(i) != cell_size) {
            utility::LogError(
                    "Cannot convert to Tensor. All cells must have the same "
                    "size but first cell has size {} and cell {} has size {}",
                    cell_size, i, cells->GetCellSize(i));
        }
    }
    core::Tensor result =
            CreateTensorFromVtkDataArray(cells->GetConnectivityArray());
    if (num_cells * cell_size != result.NumElements()) {
        utility::LogError("Expected {}*{}={} elements but got {}", num_cells,
                          cell_size, num_cells * cell_size,
                          result.NumElements());
    }
    return result.Reshape({num_cells, cell_size});
}

vtkSmartPointer<vtkPolyData> CreateVtkPolyDataFromGeometry(
        const Geometry& geometry) {
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    if (geometry.GetGeometryType() == Geometry::GeometryType::PointCloud) {
        auto pcd = static_cast<const PointCloud&>(geometry);
        polyData->SetPoints(CreateVtkPointsFromTensor(pcd.GetPointPositions()));
        vtkSmartPointer<vtkCellArray> cells =
                vtkSmartPointer<vtkCellArray>::New();

        const size_t num_cells = pcd.GetPointPositions().GetLength();
        for (size_t i = 0; i < num_cells; ++i) {
            cells->InsertNextCell(1);
            cells->InsertCellPoint(i);
        }

        polyData->SetVerts(cells);
    } else if (geometry.GetGeometryType() ==
               Geometry::GeometryType::TriangleMesh) {
        auto mesh = static_cast<const TriangleMesh&>(geometry);
        polyData->SetPoints(
                CreateVtkPointsFromTensor(mesh.GetVertexPositions()));
        polyData->SetPolys(
                CreateVtkCellArrayFromTensor(mesh.GetTriangleIndices()));
    }
    // TODO convert other data like normals, colors, ...

    return polyData;
}

TriangleMesh CreateTriangleMeshFromVtkPolyData(vtkPolyData* polyData) {
    core::Tensor vertices =
            CreateTensorFromVtkDataArray(polyData->GetPoints()->GetData());
    core::Tensor triangles = CreateTensorFromVtkCellArray(polyData->GetPolys());
    TriangleMesh mesh(vertices, triangles);
    return mesh;
}

}  // namespace vtkutils
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
