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

/// Returns the corresponding vtk data type for core::Dtype
/// Logs an error if no conversion exists.
static int DtypeToVtkType(const core::Dtype& dtype) {
    if (dtype == core::Float32) {
        return VTK_FLOAT;
    } else if (dtype == core::Float64) {
        return VTK_DOUBLE;
    } else if (dtype == core::Int8) {
        return VTK_CHAR;
    } else if (dtype == core::Int16) {
        return VTK_SHORT;
    } else if (dtype == core::Int32) {
        return VTK_INT;
    } else if (dtype == core::Int64) {
        return VTK_LONG;
    } else if (dtype == core::UInt8) {
        return VTK_UNSIGNED_CHAR;
    } else if (dtype == core::UInt16) {
        return VTK_UNSIGNED_SHORT;
    } else if (dtype == core::UInt32) {
        return VTK_UNSIGNED_INT;
    } else if (dtype == core::UInt64) {
        return VTK_UNSIGNED_LONG;
    } else if (dtype == core::Bool) {
        // VTK_BIT arrays are compact and store 8 booleans per byte!
        return VTK_BIT;
    } else {
        utility::LogError("Type {} cannot be converted to a vtk data type!",
                          dtype.ToString());
    }
    return VTK_INT;
}

/// Creates a vtkDataArray from a Tensor.
/// The returned array may directly use the memory of the tensor and the tensor
/// must be kept alive until the returned vtkDataArray is deleted.
/// \param tensor The source tensor.
/// \param copy If true always create a copy of the data.
static vtkSmartPointer<vtkDataArray> CreateVtkDataArrayFromTensor(
        core::Tensor& tensor, bool copy) {
    core::AssertTensorShape(tensor, {core::None, core::None});
    if (tensor.GetDtype() == core::Bool) {
        utility::LogError(
                "Tensor conversion with type Bool is not implemented!");
    }

    int vtk_data_type = DtypeToVtkType(tensor.GetDtype());
    auto tensor_cpu = tensor.To(core::Device()).Contiguous();

    vtkSmartPointer<vtkDataArray> data_array;
    data_array.TakeReference(vtkDataArray::CreateDataArray(vtk_data_type));

    if (!copy && tensor.GetDataPtr() == tensor_cpu.GetDataPtr()) {
        // reuse tensor memory
        data_array->SetVoidArray(tensor.GetDataPtr(), tensor.NumElements(),
                                 1 /*dont delete*/);
        data_array->SetNumberOfComponents(tensor.GetShape(1));
        data_array->SetNumberOfTuples(tensor.GetShape(0));
    } else {
        // allocate new data array and copy
        data_array->SetNumberOfComponents(tensor.GetShape(1));
        data_array->SetNumberOfTuples(tensor.GetShape(0));
        memcpy(data_array->GetVoidPointer(0), tensor_cpu.GetDataPtr(),
               tensor_cpu.GetDtype().ByteSize() * tensor_cpu.NumElements());
    }

    return data_array;
}

/// Creates a vtkPoints object from a Tensor.
/// The returned array may directly use the memory of the tensor and the tensor
/// must be kept alive until the returned vtkPoints is deleted.
/// \param tensor The source tensor.
/// \param copy If true always create a copy of the data.
static vtkSmartPointer<vtkPoints> CreateVtkPointsFromTensor(
        core::Tensor& tensor, bool copy = false) {
    core::AssertTensorShape(tensor, {core::None, 3});
    core::AssertTensorDtypes(tensor, {core::Float32, core::Float64});

    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    auto data_array = CreateVtkDataArrayFromTensor(tensor, copy);
    pts->SetData(data_array);
    return pts;
}

// Helper for creating the offset array from Common/DataModel/vtkCellArray.cxx
struct GenerateOffsetsImpl {
    vtkIdType CellSize;
    vtkIdType ConnectivityArraySize;

    template <typename ArrayT>
    void operator()(ArrayT* offsets) {
        for (vtkIdType cc = 0, max = (offsets->GetNumberOfTuples() - 1);
             cc < max; ++cc) {
            offsets->SetTypedComponent(cc, 0, cc * this->CellSize);
        }
        offsets->SetTypedComponent(offsets->GetNumberOfTuples() - 1, 0,
                                   this->ConnectivityArraySize);
    }
};

/// Creates a vtkCellArray from a Tensor.
/// The returned array may directly use the memory of the tensor and the tensor
/// must be kept alive until the returned vtkPoints is deleted.
/// \param tensor The source tensor.
/// \param copy If true always create a copy of the data.
static vtkSmartPointer<vtkCellArray> CreateVtkCellArrayFromTensor(
        core::Tensor& tensor, bool copy = false) {
    core::AssertTensorShape(tensor, {core::None, core::None});
    core::AssertTensorDtypes(tensor, {core::Int32, core::Int64});

    const int cell_size = tensor.GetShape()[1];

    auto tensor_flat = tensor.Reshape({tensor.NumElements(), 1}).Contiguous();
    copy = copy && tensor.GetDataPtr() == tensor_flat.GetDataPtr();
    auto connectivity = CreateVtkDataArrayFromTensor(tensor_flat, copy);

    // vtk nightly build (9.1.20220520) has a function cells->SetData(cell_size,
    // connectivity) which allows to remove the code below
    vtkSmartPointer<vtkDataArray> offsets;
    {
        offsets.TakeReference(connectivity->NewInstance());
        offsets->SetNumberOfTuples(1 + connectivity->GetNumberOfTuples() /
                                               cell_size);

        GenerateOffsetsImpl worker{cell_size,
                                   connectivity->GetNumberOfTuples()};
        using SupportedArrays = vtkCellArray::InputArrayList;
        using Dispatch = vtkArrayDispatch::DispatchByArray<SupportedArrays>;
        Dispatch::Execute(offsets, worker);
    }
    //--

    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    // grr, this always makes a deep copy.
    // See ShallowCopy() in Common/Core/vtkDataArray.cxx for why.
    cells->SetData(offsets, connectivity);

    return cells;
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
    bool copy;
    vtkSmartPointer<vtkDataArray> data_array;
    core::Tensor result;

    template <class TArray>
    void operator()(TArray* array) {
        typedef typename TArray::ValueType T;
        typedef typename VtkToTensorType<T>::TensorType TTensor;
        auto dtype = core::Dtype::FromType<TTensor>();
        int64_t length = array->GetNumberOfTuples();
        int64_t num_components = array->GetNumberOfComponents();

        // copy if requested or the layout is not contiguous
        if (copy || !array->HasStandardMemoryLayout()) {
            result = core::Tensor({length, num_components}, dtype);
            if (array->HasStandardMemoryLayout()) {
                memcpy(array->GetVoidPointer(0), result.GetDataPtr(),
                       dtype.ByteSize() * result.NumElements());
            } else {
                TTensor* data = result.GetDataPtr<TTensor>();
                for (int64_t i = 0; i < length; ++i) {
                    for (int64_t j = 0; j < num_components; ++j) {
                        *data = array->GetTypedComponent(i, j);
                        ++data;
                    }
                }
            }
        } else {
            auto sp = data_array;
            auto blob = std::make_shared<core::Blob>(core::Device(),
                                                     array->GetVoidPointer(0),
                                                     [sp](void*) { (void)sp; });
            core::SizeVector shape{length, num_components};
            auto strides = core::shape_util::DefaultStrides(shape);
            result = core::Tensor(shape, strides, blob->GetDataPtr(), dtype,
                                  blob);
        }
    }
};

/// Creates a tensor from a vtkDataArray.
/// The returned Tensor may directly use the memory of the array if device (CPU)
/// and memory layout are compatible.
/// The returned Tensor will hold a reference to the array and it is not
/// necessary to keep other references to the array alive.
/// \param array The source array.
/// \param copy If true always create a copy of the data.
static core::Tensor CreateTensorFromVtkDataArray(vtkDataArray* array,
                                                 bool copy = false) {
    CreateTensorFromVtkDataArrayWorker worker;
    worker.copy = copy;
    worker.data_array =
            vtkSmartPointer<vtkDataArray>(array);  // inc the refcount
    typedef vtkTypeList_Create_7(float, double, int32_t, int64_t, uint32_t,
                                 uint64_t, long long) ArrayTypes;
    vtkArrayDispatch::DispatchByValueType<ArrayTypes>::Execute(array, worker);
    return worker.result;
}

/// Creates a tensor from a vtkCellArray.
/// The returned Tensor may directly use the memory of the array if device (CPU)
/// and memory layout are compatible.
/// The returned Tensor will hold a reference to the array and it is not
/// necessary to keep other references to the array alive.
/// Note that cell arrays with varying cell sizes cannot be converted and the
/// function will throw an exception.
/// \param array The source array.
/// \param copy If true always create a copy of the data.
static core::Tensor CreateTensorFromVtkCellArray(vtkCellArray* cells,
                                                 bool copy = false) {
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
            CreateTensorFromVtkDataArray(cells->GetConnectivityArray(), copy);
    if (num_cells * cell_size != result.NumElements()) {
        utility::LogError("Expected {}*{}={} elements but got {}", num_cells,
                          cell_size, num_cells * cell_size,
                          result.NumElements());
    }
    return result.Reshape({num_cells, cell_size});
}

vtkSmartPointer<vtkPolyData> CreateVtkPolyDataFromGeometry(
        const Geometry& geometry, bool copy) {
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

    if (geometry.GetGeometryType() == Geometry::GeometryType::PointCloud) {
        auto pcd = static_cast<const PointCloud&>(geometry);
        polydata->SetPoints(
                CreateVtkPointsFromTensor(pcd.GetPointPositions(), copy));
        vtkSmartPointer<vtkCellArray> cells =
                vtkSmartPointer<vtkCellArray>::New();

        const size_t num_cells = pcd.GetPointPositions().GetLength();
        for (size_t i = 0; i < num_cells; ++i) {
            cells->InsertNextCell(1);
            cells->InsertCellPoint(i);
        }

        polydata->SetVerts(cells);
    } else if (geometry.GetGeometryType() ==
               Geometry::GeometryType::TriangleMesh) {
        auto mesh = static_cast<const TriangleMesh&>(geometry);
        polydata->SetPoints(
                CreateVtkPointsFromTensor(mesh.GetVertexPositions(), copy));
        polydata->SetPolys(
                CreateVtkCellArrayFromTensor(mesh.GetTriangleIndices(), copy));
    } else {
        utility::LogError("Unsupported geometry type {}",
                          geometry.GetGeometryType());
    }
    // TODO convert other data like normals, colors, ...

    return polydata;
}

TriangleMesh CreateTriangleMeshFromVtkPolyData(vtkPolyData* polydata,
                                               bool copy) {
    core::Tensor vertices = CreateTensorFromVtkDataArray(
            polydata->GetPoints()->GetData(), copy);
    core::Tensor triangles =
            CreateTensorFromVtkCellArray(polydata->GetPolys(), copy);
    TriangleMesh mesh(vertices, triangles);
    return mesh;
}

}  // namespace vtkutils
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
