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

#include <vtkArrayDispatch.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkRotationalExtrusionFilter.h>
#include <vtkTriangleFilter.h>

namespace open3d {
namespace t {
namespace geometry {
namespace vtkutils {

int DtypeToVtkType(const core::Dtype& dtype) {
    if (dtype == core::Float32) {
        return VTK_FLOAT;
    } else if (dtype == core::Float64) {
        return VTK_DOUBLE;
    } else if (dtype == core::Int8) {
        return VTK_TYPE_INT8;
    } else if (dtype == core::Int16) {
        return VTK_TYPE_INT16;
    } else if (dtype == core::Int32) {
        return VTK_TYPE_INT32;
    } else if (dtype == core::Int64) {
        return VTK_TYPE_INT64;
    } else if (dtype == core::UInt8) {
        return VTK_TYPE_UINT8;
    } else if (dtype == core::UInt16) {
        return VTK_TYPE_UINT16;
    } else if (dtype == core::UInt32) {
        return VTK_TYPE_UINT32;
    } else if (dtype == core::UInt64) {
        return VTK_TYPE_UINT64;
    } else if (dtype == core::Bool) {
        // VTK_BIT arrays are compact and store 8 booleans per byte!
        return VTK_BIT;
    } else {
        utility::LogError("Type {} cannot be converted to a vtk data type!",
                          dtype.ToString());
    }
    return VTK_INT;
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

namespace {
struct CreateTensorFromVtkDataArrayWorker {
    bool copy;
    vtkDataArray* data_array;
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
            vtkSmartPointer<vtkDataArray> sp(data_array);  // inc the refcount
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
}  // namespace

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
    worker.data_array = array;
    typedef vtkTypeList_Create_7(float, double, int32_t, int64_t, uint32_t,
                                 uint64_t, long long) ArrayTypes;
    vtkArrayDispatch::DispatchByValueType<ArrayTypes>::Execute(array, worker);
    return worker.result;
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

    vtkSmartPointer<vtkDataArray> data_array;
    data_array.TakeReference(vtkDataArray::CreateDataArray(vtk_data_type));

    if (!copy && tensor.IsContiguous() &&
        tensor.GetDevice() == core::Device()) {
        // reuse tensor memory
        data_array->SetVoidArray(tensor.GetDataPtr(), tensor.NumElements(),
                                 1 /*dont delete*/);
        data_array->SetNumberOfComponents(tensor.GetShape(1));
        data_array->SetNumberOfTuples(tensor.GetShape(0));
    } else {
        // copy if requested or if data is not contiguous and/or is on different
        // devices
        data_array->SetNumberOfComponents(tensor.GetShape(1));
        data_array->SetNumberOfTuples(tensor.GetShape(0));
        auto dst_tensor = CreateTensorFromVtkDataArray(data_array, false);
        dst_tensor.CopyFrom(tensor);
    }

    return data_array;
}  // namespace vtkutils

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

namespace {
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
}  // namespace

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

/// Adds point or cell attribute arrays to a TensorMap.
/// \param tmap The destination TensorMap.
/// \param field_data The source vtkFieldData.
/// \param copy If true always create a copy for attribute arrays.
static void AddVtkFieldDataToTensorMap(TensorMap& tmap,
                                       vtkFieldData* field_data,
                                       bool copy) {
    for (int i = 0; i < field_data->GetNumberOfArrays(); ++i) {
        auto array = field_data->GetArray(i);
        char* array_name = array->GetName();
        if (array_name) {
            tmap[array_name] = CreateTensorFromVtkDataArray(array, copy);
        }
    }
}

/// Adds attribute tensors to vtkFieldData.
/// Primary key tensors will be ignored by this function.
/// \param field_data The destination vtkFieldData.
/// \param tmap The source TensorMap.
/// \param copy If true always create a copy for attribute arrays.
/// \param include A set of keys to select which attributes should be added.
/// \param exclude A set of keys for which attributes will not be added to the
/// vtkFieldData. The exclusion set has precedence over the included keys.
static void AddTensorMapToVtkFieldData(
        vtkFieldData* field_data,
        TensorMap& tmap,
        bool copy,
        std::unordered_set<std::string> include,
        std::unordered_set<std::string> exclude = {}) {
    for (auto key_tensor : tmap) {
        // we only want attributes and ignore the primary key here
        if (key_tensor.first == tmap.GetPrimaryKey()) {
            continue;
        }
        // we only support 2D tensors
        if (key_tensor.second.NumDims() != 2) {
            utility::LogWarning(
                    "Ignoring attribute '{}' for TensorMap with primary key "
                    "'{}' because of incompatible ndim={}",
                    key_tensor.first, tmap.GetPrimaryKey(),
                    key_tensor.second.NumDims());
            continue;
        }

        if (include.count(key_tensor.first) &&
            !exclude.count(key_tensor.first)) {
            auto array = CreateVtkDataArrayFromTensor(key_tensor.second, copy);
            array->SetName(key_tensor.first.c_str());
            field_data->AddArray(array);
        } else {
            utility::LogWarning(
                    "Ignoring attribute '{}' for TensorMap with primary key "
                    "'{}'",
                    key_tensor.first, tmap.GetPrimaryKey());
        }
    }
}

vtkSmartPointer<vtkPolyData> CreateVtkPolyDataFromGeometry(
        const Geometry& geometry,
        const std::unordered_set<std::string>& point_attr_include,
        const std::unordered_set<std::string>& face_attr_include,
        const std::unordered_set<std::string>& point_attr_exclude,
        const std::unordered_set<std::string>& face_attr_exclude,
        bool copy) {
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
        AddTensorMapToVtkFieldData(polydata->GetPointData(), pcd.GetPointAttr(),
                                   copy, point_attr_include,
                                   point_attr_exclude);

    } else if (geometry.GetGeometryType() == Geometry::GeometryType::LineSet) {
        auto lineset = static_cast<const LineSet&>(geometry);
        polydata->SetPoints(
                CreateVtkPointsFromTensor(lineset.GetPointPositions(), copy));
        polydata->SetLines(
                CreateVtkCellArrayFromTensor(lineset.GetLineIndices(), copy));

        AddTensorMapToVtkFieldData(polydata->GetPointData(),
                                   lineset.GetPointAttr(), copy,
                                   point_attr_include, point_attr_exclude);
        AddTensorMapToVtkFieldData(polydata->GetCellData(),
                                   lineset.GetLineAttr(), copy,
                                   face_attr_include, face_attr_exclude);
    } else if (geometry.GetGeometryType() ==
               Geometry::GeometryType::TriangleMesh) {
        auto mesh = static_cast<const TriangleMesh&>(geometry);
        polydata->SetPoints(
                CreateVtkPointsFromTensor(mesh.GetVertexPositions(), copy));
        polydata->SetPolys(
                CreateVtkCellArrayFromTensor(mesh.GetTriangleIndices(), copy));

        AddTensorMapToVtkFieldData(polydata->GetPointData(),
                                   mesh.GetVertexAttr(), copy,
                                   point_attr_include, point_attr_exclude);
        AddTensorMapToVtkFieldData(polydata->GetCellData(),
                                   mesh.GetTriangleAttr(), copy,
                                   face_attr_include, face_attr_exclude);
    } else {
        utility::LogError("Unsupported geometry type {}",
                          geometry.GetGeometryType());
    }

    return polydata;
}

TriangleMesh CreateTriangleMeshFromVtkPolyData(vtkPolyData* polydata,
                                               bool copy) {
    if (!polydata->GetPoints()) {
        return TriangleMesh();
    }
    core::Tensor vertices = CreateTensorFromVtkDataArray(
            polydata->GetPoints()->GetData(), copy);

    core::Tensor triangles =
            CreateTensorFromVtkCellArray(polydata->GetPolys(), copy);
    // Some algorithms return an empty tensor with shape (0,0).
    // Fix the last dim here.
    if (triangles.GetShape() == core::SizeVector{0, 0}) {
        triangles = triangles.Reshape({0, 3});
    }
    TriangleMesh mesh(vertices, triangles);

    AddVtkFieldDataToTensorMap(mesh.GetVertexAttr(), polydata->GetPointData(),
                               copy);
    AddVtkFieldDataToTensorMap(mesh.GetTriangleAttr(), polydata->GetCellData(),
                               copy);
    return mesh;
}

OPEN3D_LOCAL LineSet CreateLineSetFromVtkPolyData(vtkPolyData* polydata,
                                                  bool copy) {
    if (!polydata->GetPoints()) {
        return LineSet();
    }
    core::Tensor vertices = CreateTensorFromVtkDataArray(
            polydata->GetPoints()->GetData(), copy);

    core::Tensor lines =
            CreateTensorFromVtkCellArray(polydata->GetLines(), copy);
    // Some algorithms return an empty tensor with shape (0,0).
    // Fix the last dim here.
    if (lines.GetShape() == core::SizeVector{0, 0}) {
        lines = lines.Reshape({0, 2});
    }
    LineSet lineset(vertices, lines);

    AddVtkFieldDataToTensorMap(lineset.GetPointAttr(), polydata->GetPointData(),
                               copy);
    AddVtkFieldDataToTensorMap(lineset.GetLineAttr(), polydata->GetCellData(),
                               copy);
    return lineset;
}

static vtkSmartPointer<vtkPolyData> ExtrudeRotationPolyData(
        const Geometry& geometry,
        const double angle,
        const core::Tensor& axis,
        int resolution,
        double translation,
        bool capping) {
    core::AssertTensorShape(axis, {3});
    // allow int types for convenience
    core::AssertTensorDtypes(
            axis, {core::Float32, core::Float64, core::Int32, core::Int64});
    auto axis_ = axis.To(core::Device(), core::Float64).Contiguous();

    auto polydata =
            CreateVtkPolyDataFromGeometry(geometry, {}, {}, {}, {}, false);

    vtkNew<vtkRotationalExtrusionFilter> extrude;
    extrude->SetInputData(polydata);
    extrude->SetAngle(angle);
    extrude->SetRotationAxis(axis_.GetDataPtr<double>());
    extrude->SetResolution(resolution);
    extrude->SetTranslation(translation);
    extrude->SetCapping(capping);

    vtkNew<vtkTriangleFilter> triangulate;
    triangulate->SetInputConnection(extrude->GetOutputPort());
    triangulate->Update();
    vtkSmartPointer<vtkPolyData> swept_polydata = triangulate->GetOutput();
    return swept_polydata;
}

OPEN3D_LOCAL TriangleMesh ExtrudeRotationTriangleMesh(const Geometry& geometry,
                                                      const double angle,
                                                      const core::Tensor& axis,
                                                      int resolution,
                                                      double translation,
                                                      bool capping) {
    auto polydata = ExtrudeRotationPolyData(geometry, angle, axis, resolution,
                                            translation, capping);
    return CreateTriangleMeshFromVtkPolyData(polydata);
}

OPEN3D_LOCAL LineSet ExtrudeRotationLineSet(const PointCloud& pointcloud,
                                            const double angle,
                                            const core::Tensor& axis,
                                            int resolution,
                                            double translation,
                                            bool capping) {
    auto polydata = ExtrudeRotationPolyData(pointcloud, angle, axis, resolution,
                                            translation, capping);
    return CreateLineSetFromVtkPolyData(polydata);
}

static vtkSmartPointer<vtkPolyData> ExtrudeLinearPolyData(
        const Geometry& geometry,
        const core::Tensor& vector,
        double scale,
        bool capping) {
    core::AssertTensorShape(vector, {3});
    // allow int types for convenience
    core::AssertTensorDtypes(
            vector, {core::Float32, core::Float64, core::Int32, core::Int64});
    auto vector_ = vector.To(core::Device(), core::Float64).Contiguous();

    auto polydata =
            CreateVtkPolyDataFromGeometry(geometry, {}, {}, {}, {}, false);

    vtkNew<vtkLinearExtrusionFilter> extrude;
    extrude->SetInputData(polydata);
    extrude->SetExtrusionTypeToVectorExtrusion();
    extrude->SetVector(vector_.GetDataPtr<double>());
    extrude->SetScaleFactor(scale);
    extrude->SetCapping(capping);

    vtkNew<vtkTriangleFilter> triangulate;
    triangulate->SetInputConnection(extrude->GetOutputPort());
    triangulate->Update();
    vtkSmartPointer<vtkPolyData> swept_polydata = triangulate->GetOutput();
    return swept_polydata;
}

OPEN3D_LOCAL TriangleMesh ExtrudeLinearTriangleMesh(const Geometry& geometry,
                                                    const core::Tensor& vector,
                                                    double scale,
                                                    bool capping) {
    auto polydata = ExtrudeLinearPolyData(geometry, vector, scale, capping);
    return CreateTriangleMeshFromVtkPolyData(polydata);
}

OPEN3D_LOCAL LineSet ExtrudeLinearLineSet(const PointCloud& pointcloud,
                                          const core::Tensor& vector,
                                          double scale,
                                          bool capping) {
    auto polydata = ExtrudeLinearPolyData(pointcloud, vector, scale, capping);
    return CreateLineSetFromVtkPolyData(polydata);
}

}  // namespace vtkutils
}  // namespace geometry
}  // namespace t
}  // namespace open3d
