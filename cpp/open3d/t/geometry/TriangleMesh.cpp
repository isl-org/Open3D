// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/TriangleMesh.h"

#include <fmt/core.h>
#include <tbb/parallel_for_each.h>
#include <vtkBooleanOperationPolyDataFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkClipPolyData.h>
#include <vtkCutter.h>
#include <vtkFillHolesFilter.h>
#include <vtkPlane.h>
#include <vtkQuadricDecimation.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/TensorKey.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/t/geometry/VtkUtils.h"
#include "open3d/t/geometry/kernel/IPPImage.h"
#include "open3d/t/geometry/kernel/Metrics.h"
#include "open3d/t/geometry/kernel/PCAPartition.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/Transform.h"
#include "open3d/t/geometry/kernel/TriangleMesh.h"
#include "open3d/t/geometry/kernel/UVUnwrapping.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace t {
namespace geometry {

class PointCloud;  // forward declaration

TriangleMesh::TriangleMesh(const core::Device &device)
    : Geometry(Geometry::GeometryType::TriangleMesh, 3),
      device_(device),
      vertex_attr_(TensorMap("positions")),
      triangle_attr_(TensorMap("indices")) {}

TriangleMesh::TriangleMesh(const core::Tensor &vertex_positions,
                           const core::Tensor &triangle_indices)
    : TriangleMesh([&]() {
          if (vertex_positions.GetDevice() != triangle_indices.GetDevice()) {
              utility::LogError(
                      "vertex_positions' device {} does not match "
                      "triangle_indices' device {}.",
                      vertex_positions.GetDevice().ToString(),
                      triangle_indices.GetDevice().ToString());
          }
          return vertex_positions.GetDevice();
      }()) {
    SetVertexPositions(vertex_positions);
    SetTriangleIndices(triangle_indices);
}

std::string TriangleMesh::ToString() const {
    size_t num_vertices = 0;
    std::string vertex_dtype_str = "";
    size_t num_triangles = 0;
    std::string triangles_dtype_str = "";
    if (vertex_attr_.count(vertex_attr_.GetPrimaryKey())) {
        num_vertices = GetVertexPositions().GetLength();
        vertex_dtype_str = fmt::format(
                " ({})", GetVertexPositions().GetDtype().ToString());
    }
    if (triangle_attr_.count(triangle_attr_.GetPrimaryKey())) {
        num_triangles = GetTriangleIndices().GetLength();
        triangles_dtype_str = fmt::format(
                " ({})", GetTriangleIndices().GetDtype().ToString());
    }

    auto str = fmt::format(
            "TriangleMesh on {} [{} vertices{} and {} triangles{}].",
            GetDevice().ToString(), num_vertices, vertex_dtype_str,
            num_triangles, triangles_dtype_str);

    std::string vertices_attr_str = "\nVertex Attributes:";
    if ((vertex_attr_.size() -
         vertex_attr_.count(vertex_attr_.GetPrimaryKey())) == 0) {
        vertices_attr_str += " None.";
    } else {
        for (const auto &kv : vertex_attr_) {
            if (kv.first != "positions") {
                vertices_attr_str +=
                        fmt::format(" {} (dtype = {}, shape = {}),", kv.first,
                                    kv.second.GetDtype().ToString(),
                                    kv.second.GetShape().ToString());
            }
        }
        vertices_attr_str[vertices_attr_str.size() - 1] = '.';
    }

    std::string triangles_attr_str = "\nTriangle Attributes:";
    if ((triangle_attr_.size() -
         triangle_attr_.count(triangle_attr_.GetPrimaryKey())) == 0) {
        triangles_attr_str += " None.";
    } else {
        for (const auto &kv : triangle_attr_) {
            if (kv.first != "indices") {
                triangles_attr_str +=
                        fmt::format(" {} (dtype = {}, shape = {}),", kv.first,
                                    kv.second.GetDtype().ToString(),
                                    kv.second.GetShape().ToString());
            }
        }
        triangles_attr_str[triangles_attr_str.size() - 1] = '.';
    }

    return str + vertices_attr_str + triangles_attr_str;
}

TriangleMesh &TriangleMesh::Transform(const core::Tensor &transformation) {
    core::AssertTensorShape(transformation, {4, 4});

    kernel::transform::TransformPoints(transformation, GetVertexPositions());
    if (HasVertexNormals()) {
        kernel::transform::TransformNormals(transformation, GetVertexNormals());
    }
    if (HasTriangleNormals()) {
        kernel::transform::TransformNormals(transformation,
                                            GetTriangleNormals());
    }

    return *this;
}

TriangleMesh &TriangleMesh::Translate(const core::Tensor &translation,
                                      bool relative) {
    core::AssertTensorShape(translation, {3});

    core::Tensor transform =
            translation.To(GetDevice(), GetVertexPositions().GetDtype());

    if (!relative) {
        transform -= GetCenter();
    }
    GetVertexPositions() += transform;
    return *this;
}

TriangleMesh &TriangleMesh::Scale(double scale, const core::Tensor &center) {
    core::AssertTensorShape(center, {3});
    core::AssertTensorDevice(center, device_);

    const core::Tensor center_d =
            center.To(GetDevice(), GetVertexPositions().GetDtype());

    GetVertexPositions().Sub_(center_d).Mul_(scale).Add_(center_d);
    return *this;
}

TriangleMesh &TriangleMesh::Rotate(const core::Tensor &R,
                                   const core::Tensor &center) {
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(center, {3});

    kernel::transform::RotatePoints(R, GetVertexPositions(), center);
    if (HasVertexNormals()) {
        kernel::transform::RotateNormals(R, GetVertexNormals());
    }
    if (HasTriangleNormals()) {
        kernel::transform::RotateNormals(R, GetTriangleNormals());
    }
    return *this;
}

TriangleMesh &TriangleMesh::NormalizeNormals() {
    if (HasVertexNormals()) {
        SetVertexNormals(GetVertexNormals().Contiguous());
        core::Tensor &vertex_normals = GetVertexNormals();
        if (IsCPU()) {
            kernel::trianglemesh::NormalizeNormalsCPU(vertex_normals);
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::trianglemesh::NormalizeNormalsCUDA,
                      vertex_normals);
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogWarning("TriangleMesh has no vertex normals.");
    }

    if (HasTriangleNormals()) {
        SetTriangleNormals(GetTriangleNormals().Contiguous());
        core::Tensor &triangle_normals = GetTriangleNormals();
        if (IsCPU()) {
            kernel::trianglemesh::NormalizeNormalsCPU(triangle_normals);
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::trianglemesh::NormalizeNormalsCUDA,
                      triangle_normals);
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogWarning("TriangleMesh has no triangle normals.");
    }

    return *this;
}

TriangleMesh &TriangleMesh::ComputeTriangleNormals(bool normalized) {
    if (IsEmpty()) {
        utility::LogWarning("TriangleMesh is empty.");
        return *this;
    }

    if (!HasTriangleIndices()) {
        utility::LogWarning("TriangleMesh has no triangle indices.");
        return *this;
    }

    const int64_t triangle_num = GetTriangleIndices().GetLength();
    const core::Dtype dtype = GetVertexPositions().GetDtype();
    core::Tensor triangle_normals({triangle_num, 3}, dtype, GetDevice());
    SetVertexPositions(GetVertexPositions().Contiguous());
    SetTriangleIndices(GetTriangleIndices().Contiguous());

    if (IsCPU()) {
        kernel::trianglemesh::ComputeTriangleNormalsCPU(
                GetVertexPositions(), GetTriangleIndices(), triangle_normals);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::trianglemesh::ComputeTriangleNormalsCUDA,
                  GetVertexPositions(), GetTriangleIndices(), triangle_normals);
    } else {
        utility::LogError("Unimplemented device");
    }

    SetTriangleNormals(triangle_normals);

    if (normalized) {
        NormalizeNormals();
    }

    return *this;
}

TriangleMesh &TriangleMesh::ComputeVertexNormals(bool normalized) {
    if (IsEmpty()) {
        utility::LogWarning("TriangleMesh is empty.");
        return *this;
    }

    if (!HasTriangleIndices()) {
        utility::LogWarning("TriangleMesh has no triangle indices.");
        return *this;
    }

    ComputeTriangleNormals(false);

    const int64_t vertex_num = GetVertexPositions().GetLength();
    const core::Dtype dtype = GetVertexPositions().GetDtype();
    core::Tensor vertex_normals =
            core::Tensor::Zeros({vertex_num, 3}, dtype, GetDevice());

    SetTriangleNormals(GetTriangleNormals().Contiguous());
    SetTriangleIndices(GetTriangleIndices().Contiguous());

    if (IsCPU()) {
        kernel::trianglemesh::ComputeVertexNormalsCPU(
                GetTriangleIndices(), GetTriangleNormals(), vertex_normals);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::trianglemesh::ComputeVertexNormalsCUDA,
                  GetTriangleIndices(), GetTriangleNormals(), vertex_normals);
    } else {
        utility::LogError("Unimplemented device");
    }

    SetVertexNormals(vertex_normals);
    if (normalized) {
        NormalizeNormals();
    }

    return *this;
}

static core::Tensor ComputeTriangleAreasHelper(const TriangleMesh &mesh) {
    const int64_t triangle_num = mesh.GetTriangleIndices().GetLength();
    const core::Dtype dtype = mesh.GetVertexPositions().GetDtype();
    core::Tensor triangle_areas({triangle_num}, dtype, mesh.GetDevice());
    if (mesh.IsCPU()) {
        kernel::trianglemesh::ComputeTriangleAreasCPU(
                mesh.GetVertexPositions().Contiguous(),
                mesh.GetTriangleIndices().Contiguous(), triangle_areas);
    } else if (mesh.IsCUDA()) {
        CUDA_CALL(kernel::trianglemesh::ComputeTriangleAreasCUDA,
                  mesh.GetVertexPositions().Contiguous(),
                  mesh.GetTriangleIndices().Contiguous(), triangle_areas);
    } else {
        utility::LogError("Unimplemented device");
    }

    return triangle_areas;
}

TriangleMesh &TriangleMesh::ComputeTriangleAreas() {
    if (IsEmpty()) {
        utility::LogWarning("TriangleMesh is empty.");
        return *this;
    }

    if (!HasTriangleIndices()) {
        SetTriangleAttr("areas", core::Tensor::Empty(
                                         {0}, GetVertexPositions().GetDtype(),
                                         GetDevice()));
        utility::LogWarning("TriangleMesh has no triangle indices.");
        return *this;
    }
    if (HasTriangleAttr("areas")) {
        utility::LogWarning(
                "TriangleMesh already has triangle areas: remove "
                "'areas' triangle attribute if you'd like to update.");
        return *this;
    }

    core::Tensor triangle_areas = ComputeTriangleAreasHelper(*this);
    SetTriangleAttr("areas", triangle_areas);

    return *this;
}

double TriangleMesh::GetSurfaceArea() const {
    double surface_area = 0;
    if (IsEmpty()) {
        utility::LogWarning("TriangleMesh is empty.");
        return surface_area;
    }

    if (!HasTriangleIndices()) {
        utility::LogWarning("TriangleMesh has no triangle indices.");
        return surface_area;
    }

    core::Tensor triangle_areas = ComputeTriangleAreasHelper(*this);
    surface_area = triangle_areas.Sum({0}).To(core::Float64).Item<double>();

    return surface_area;
}

geometry::TriangleMesh TriangleMesh::FromLegacy(
        const open3d::geometry::TriangleMesh &mesh_legacy,
        core::Dtype float_dtype,
        core::Dtype int_dtype,
        const core::Device &device) {
    if (float_dtype != core::Float32 && float_dtype != core::Float64) {
        utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }
    if (int_dtype != core::Int32 && int_dtype != core::Int64) {
        utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    TriangleMesh mesh(device);
    if (mesh_legacy.HasVertices()) {
        mesh.SetVertexPositions(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.vertices_, float_dtype, device));
    } else {
        utility::LogWarning("Creating from empty legacy TriangleMesh.");
    }
    if (mesh_legacy.HasVertexColors()) {
        mesh.SetVertexColors(core::eigen_converter::EigenVector3dVectorToTensor(
                mesh_legacy.vertex_colors_, float_dtype, device));
    }
    if (mesh_legacy.HasVertexNormals()) {
        mesh.SetVertexNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.vertex_normals_, float_dtype, device));
    }
    if (mesh_legacy.HasTriangles()) {
        mesh.SetTriangleIndices(
                core::eigen_converter::EigenVector3iVectorToTensor(
                        mesh_legacy.triangles_, int_dtype, device));
    }
    if (mesh_legacy.HasTriangleNormals()) {
        mesh.SetTriangleNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.triangle_normals_, float_dtype, device));
    }
    if (mesh_legacy.HasTriangleUvs()) {
        mesh.SetTriangleAttr(
                "texture_uvs",
                core::eigen_converter::EigenVector2dVectorToTensor(
                        mesh_legacy.triangle_uvs_, float_dtype, device)
                        .Reshape({-1, 3, 2}));
    }

    // Convert first material only if one or more are present
    if (mesh_legacy.materials_.size() > 0) {
        const auto &mat = mesh_legacy.materials_.begin()->second;
        auto &tmat = mesh.GetMaterial();
        tmat.SetDefaultProperties();
        tmat.SetBaseColor(Eigen::Vector4f{mat.baseColor.f4});
        tmat.SetBaseRoughness(mat.baseRoughness);
        tmat.SetBaseMetallic(mat.baseMetallic);
        tmat.SetBaseReflectance(mat.baseReflectance);
        tmat.SetAnisotropy(mat.baseAnisotropy);
        tmat.SetBaseClearcoat(mat.baseClearCoat);
        tmat.SetBaseClearcoatRoughness(mat.baseClearCoatRoughness);
        // no emissive_color in legacy mesh material
        if (mat.albedo) tmat.SetAlbedoMap(Image::FromLegacy(*mat.albedo));
        if (mat.normalMap) tmat.SetNormalMap(Image::FromLegacy(*mat.normalMap));
        if (mat.roughness)
            tmat.SetRoughnessMap(Image::FromLegacy(*mat.roughness));
        if (mat.metallic) tmat.SetMetallicMap(Image::FromLegacy(*mat.metallic));
        if (mat.reflectance)
            tmat.SetReflectanceMap(Image::FromLegacy(*mat.reflectance));
        if (mat.ambientOcclusion)
            tmat.SetAOMap(Image::FromLegacy(*mat.ambientOcclusion));
        if (mat.clearCoat)
            tmat.SetClearcoatMap(Image::FromLegacy(*mat.clearCoat));
        if (mat.clearCoatRoughness)
            tmat.SetClearcoatRoughnessMap(
                    Image::FromLegacy(*mat.clearCoatRoughness));
        if (mat.anisotropy)
            tmat.SetAnisotropyMap(Image::FromLegacy(*mat.anisotropy));
    }
    if (mesh_legacy.materials_.size() > 1) {
        utility::LogWarning(
                "Legacy mesh has more than 1 material which is not supported "
                "by Tensor-based mesh. Only material {} was converted.",
                mesh_legacy.materials_.begin()->first);
    }
    return mesh;
}

open3d::geometry::TriangleMesh TriangleMesh::ToLegacy() const {
    open3d::geometry::TriangleMesh mesh_legacy;
    if (HasVertexPositions()) {
        mesh_legacy.vertices_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexPositions());
    }
    if (HasVertexColors()) {
        mesh_legacy.vertex_colors_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexColors());
    }
    if (HasVertexNormals()) {
        mesh_legacy.vertex_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetVertexNormals());
    }
    if (HasTriangleIndices()) {
        mesh_legacy.triangles_ =
                core::eigen_converter::TensorToEigenVector3iVector(
                        GetTriangleIndices());
    }
    if (HasTriangleNormals()) {
        mesh_legacy.triangle_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetTriangleNormals());
    }
    if (HasTriangleAttr("texture_uvs")) {
        mesh_legacy.triangle_uvs_ =
                core::eigen_converter::TensorToEigenVector2dVector(
                        GetTriangleAttr("texture_uvs").Reshape({-1, 2}));
    }
    if (HasVertexAttr("texture_uvs")) {
        utility::LogWarning("{}",
                            "texture_uvs as a vertex attribute is not "
                            "supported by legacy TriangleMesh. Ignored.");
    }

    // Convert material if the t geometry has a valid one
    auto &tmat = GetMaterial();
    if (tmat.IsValid()) {
        mesh_legacy.materials_.emplace_back();
        mesh_legacy.materials_.front().first = "Mat1";
        auto &legacy_mat = mesh_legacy.materials_.front().second;
        // Convert scalar properties
        if (tmat.HasBaseColor()) {
            legacy_mat.baseColor.f4[0] = tmat.GetBaseColor().x();
            legacy_mat.baseColor.f4[1] = tmat.GetBaseColor().y();
            legacy_mat.baseColor.f4[2] = tmat.GetBaseColor().z();
            legacy_mat.baseColor.f4[3] = tmat.GetBaseColor().w();
        }
        if (tmat.HasBaseRoughness()) {
            legacy_mat.baseRoughness = tmat.GetBaseRoughness();
        }
        if (tmat.HasBaseMetallic()) {
            legacy_mat.baseMetallic = tmat.GetBaseMetallic();
        }
        if (tmat.HasBaseReflectance()) {
            legacy_mat.baseReflectance = tmat.GetBaseReflectance();
        }
        if (tmat.HasBaseClearcoat()) {
            legacy_mat.baseClearCoat = tmat.GetBaseClearcoat();
        }
        if (tmat.HasBaseClearcoatRoughness()) {
            legacy_mat.baseClearCoatRoughness =
                    tmat.GetBaseClearcoatRoughness();
        }
        if (tmat.HasAnisotropy()) {
            legacy_mat.baseAnisotropy = tmat.GetAnisotropy();
        }
        // Convert maps
        if (tmat.HasAlbedoMap()) {
            legacy_mat.albedo = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.albedo = tmat.GetAlbedoMap().ToLegacy();
        }
        if (tmat.HasNormalMap()) {
            legacy_mat.normalMap = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.normalMap = tmat.GetNormalMap().ToLegacy();
        }
        if (tmat.HasAOMap()) {
            legacy_mat.ambientOcclusion =
                    std::make_shared<open3d::geometry::Image>();
            *legacy_mat.ambientOcclusion = tmat.GetAOMap().ToLegacy();
        }
        if (tmat.HasMetallicMap()) {
            legacy_mat.metallic = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.metallic = tmat.GetMetallicMap().ToLegacy();
        }
        if (tmat.HasRoughnessMap()) {
            legacy_mat.roughness = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.roughness = tmat.GetRoughnessMap().ToLegacy();
        }
        if (tmat.HasReflectanceMap()) {
            legacy_mat.reflectance =
                    std::make_shared<open3d::geometry::Image>();
            *legacy_mat.reflectance = tmat.GetReflectanceMap().ToLegacy();
        }
        if (tmat.HasClearcoatMap()) {
            legacy_mat.clearCoat = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.clearCoat = tmat.GetClearcoatMap().ToLegacy();
        }
        if (tmat.HasClearcoatRoughnessMap()) {
            legacy_mat.clearCoatRoughness =
                    std::make_shared<open3d::geometry::Image>();
            *legacy_mat.clearCoatRoughness =
                    tmat.GetClearcoatRoughnessMap().ToLegacy();
        }
        if (tmat.HasAnisotropyMap()) {
            legacy_mat.anisotropy = std::make_shared<open3d::geometry::Image>();
            *legacy_mat.anisotropy = tmat.GetAnisotropyMap().ToLegacy();
        }
    }

    return mesh_legacy;
}

std::unordered_map<std::string, geometry::TriangleMesh>
TriangleMesh::FromTriangleMeshModel(
        const open3d::visualization::rendering::TriangleMeshModel &model,
        core::Dtype float_dtype,
        core::Dtype int_dtype,
        const core::Device &device) {
    std::unordered_map<std::string, TriangleMesh> tmeshes;
    for (const auto &mobj : model.meshes_) {
        auto tmesh = TriangleMesh::FromLegacy(*mobj.mesh, float_dtype,
                                              int_dtype, device);
        // material textures will be on the CPU. GPU resident texture images is
        // not yet supported. See comment in Material.cpp
        tmesh.SetMaterial(
                visualization::rendering::Material::FromMaterialRecord(
                        model.materials_[mobj.material_idx]));
        tmeshes.emplace(mobj.mesh_name, tmesh);
    }
    return tmeshes;
}

TriangleMesh TriangleMesh::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    TriangleMesh mesh(device);
    for (const auto &kv : triangle_attr_) {
        mesh.SetTriangleAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    for (const auto &kv : vertex_attr_) {
        mesh.SetVertexAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    return mesh;
}

TriangleMesh TriangleMesh::ComputeConvexHull(bool joggle_inputs) const {
    PointCloud pcd(GetVertexPositions());
    return pcd.ComputeConvexHull();
}

TriangleMesh TriangleMesh::ClipPlane(const core::Tensor &point,
                                     const core::Tensor &normal) const {
    using namespace vtkutils;
    core::AssertTensorShape(point, {3});
    core::AssertTensorShape(normal, {3});
    // allow int types for convenience
    core::AssertTensorDtypes(
            point, {core::Float32, core::Float64, core::Int32, core::Int64});
    core::AssertTensorDtypes(
            normal, {core::Float32, core::Float64, core::Int32, core::Int64});

    auto point_ = point.To(core::Device(), core::Float64).Contiguous();
    auto normal_ = normal.To(core::Device(), core::Float64).Contiguous();

    auto polydata = CreateVtkPolyDataFromGeometry(
            *this, GetVertexAttr().GetKeySet(), GetTriangleAttr().GetKeySet(),
            {}, {}, false);

    vtkNew<vtkPlane> clipPlane;
    clipPlane->SetNormal(normal_.GetDataPtr<double>());
    clipPlane->SetOrigin(point_.GetDataPtr<double>());
    vtkNew<vtkClipPolyData> clipper;
    clipper->SetInputData(polydata);
    clipper->SetClipFunction(clipPlane);
    vtkNew<vtkCleanPolyData> cleaner;
    cleaner->SetInputConnection(clipper->GetOutputPort());
    cleaner->Update();
    auto clipped_polydata = cleaner->GetOutput();
    return CreateTriangleMeshFromVtkPolyData(clipped_polydata);
}

LineSet TriangleMesh::SlicePlane(
        const core::Tensor &point,
        const core::Tensor &normal,
        const std::vector<double> contour_values) const {
    using namespace vtkutils;
    core::AssertTensorShape(point, {3});
    core::AssertTensorShape(normal, {3});
    // allow int types for convenience
    core::AssertTensorDtypes(
            point, {core::Float32, core::Float64, core::Int32, core::Int64});
    core::AssertTensorDtypes(
            normal, {core::Float32, core::Float64, core::Int32, core::Int64});

    auto point_ = point.To(core::Device(), core::Float64).Contiguous();
    auto normal_ = normal.To(core::Device(), core::Float64).Contiguous();

    auto polydata = CreateVtkPolyDataFromGeometry(
            *this, GetVertexAttr().GetKeySet(), {}, {}, {}, false);

    vtkNew<vtkPlane> clipPlane;
    clipPlane->SetNormal(normal_.GetDataPtr<double>());
    clipPlane->SetOrigin(point_.GetDataPtr<double>());

    vtkNew<vtkCutter> cutter;
    cutter->SetInputData(polydata);
    cutter->SetCutFunction(clipPlane);
    cutter->GenerateTrianglesOff();
    cutter->SetNumberOfContours(contour_values.size());
    int i = 0;
    for (double value : contour_values) {
        cutter->SetValue(i++, value);
    }
    cutter->Update();
    auto slices_polydata = cutter->GetOutput();

    return CreateLineSetFromVtkPolyData(slices_polydata);
}

TriangleMesh TriangleMesh::SimplifyQuadricDecimation(
        double target_reduction, bool preserve_volume) const {
    using namespace vtkutils;
    if (target_reduction >= 1.0 || target_reduction < 0) {
        utility::LogError(
                "target_reduction must be in the range [0,1) but is {}",
                target_reduction);
    }

    // exclude attributes because they will not be preserved
    auto polydata = CreateVtkPolyDataFromGeometry(*this, {}, {}, {}, {}, false);

    vtkNew<vtkQuadricDecimation> decimate;
    decimate->SetInputData(polydata);
    decimate->SetTargetReduction(target_reduction);
    decimate->SetVolumePreservation(preserve_volume);
    decimate->Update();
    auto decimated_polydata = decimate->GetOutput();

    return CreateTriangleMeshFromVtkPolyData(decimated_polydata);
}

namespace {
TriangleMesh BooleanOperation(const TriangleMesh &mesh_A,
                              const TriangleMesh &mesh_B,
                              double tolerance,
                              int op) {
    using namespace vtkutils;
    // exclude triangle attributes because they will not be preserved
    auto polydata_A = CreateVtkPolyDataFromGeometry(
            mesh_A, mesh_A.GetVertexAttr().GetKeySet(), {}, {}, {}, false);
    auto polydata_B = CreateVtkPolyDataFromGeometry(
            mesh_B, mesh_B.GetVertexAttr().GetKeySet(), {}, {}, {}, false);

    // clean meshes before passing them to the boolean operation
    vtkNew<vtkCleanPolyData> cleaner_A;
    cleaner_A->SetInputData(polydata_A);

    vtkNew<vtkCleanPolyData> cleaner_B;
    cleaner_B->SetInputData(polydata_B);

    vtkNew<vtkBooleanOperationPolyDataFilter> boolean_filter;
    boolean_filter->SetOperation(op);
    boolean_filter->SetTolerance(tolerance);
    boolean_filter->SetInputConnection(0, cleaner_A->GetOutputPort());
    boolean_filter->SetInputConnection(1, cleaner_B->GetOutputPort());
    boolean_filter->Update();
    auto out_polydata = boolean_filter->GetOutput();

    return CreateTriangleMeshFromVtkPolyData(out_polydata);
}
}  // namespace

TriangleMesh TriangleMesh::BooleanUnion(const TriangleMesh &mesh,
                                        double tolerance) const {
    return BooleanOperation(*this, mesh, tolerance,
                            vtkBooleanOperationPolyDataFilter::VTK_UNION);
}

TriangleMesh TriangleMesh::BooleanIntersection(const TriangleMesh &mesh,
                                               double tolerance) const {
    return BooleanOperation(
            *this, mesh, tolerance,
            vtkBooleanOperationPolyDataFilter::VTK_INTERSECTION);
}

TriangleMesh TriangleMesh::BooleanDifference(const TriangleMesh &mesh,
                                             double tolerance) const {
    return BooleanOperation(*this, mesh, tolerance,
                            vtkBooleanOperationPolyDataFilter::VTK_DIFFERENCE);
}

AxisAlignedBoundingBox TriangleMesh::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetVertexPositions());
}

OrientedBoundingBox TriangleMesh::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(GetVertexPositions());
}

TriangleMesh TriangleMesh::FillHoles(double hole_size) const {
    using namespace vtkutils;
    // do not include triangle attributes because they will not be preserved by
    // the hole filling algorithm
    auto polydata = CreateVtkPolyDataFromGeometry(
            *this, GetVertexAttr().GetKeySet(), {}, {}, {}, false);
    vtkNew<vtkFillHolesFilter> fill_holes;
    fill_holes->SetInputData(polydata);
    fill_holes->SetHoleSize(hole_size);
    fill_holes->Update();
    auto result = fill_holes->GetOutput();
    return CreateTriangleMeshFromVtkPolyData(result);
}

std::tuple<float, int, int> TriangleMesh::ComputeUVAtlas(
        size_t size,
        float gutter,
        float max_stretch,
        int parallel_partitions,
        int nthreads) {
    return kernel::uvunwrapping::ComputeUVAtlas(*this, size, size, gutter,
                                                max_stretch,
                                                parallel_partitions, nthreads);
}

namespace {
/// Bakes vertex or triangle attributes to a texure.
///
/// \tparam TAttr The data type of the attribute.
/// \tparam TInt The data type for triangle indices.
/// \tparam VERTEX_ATTR If true bake vertex attributes with interpolation.
/// If false bake triangle attributes.
///
/// \param size The texture size.
/// \param margin The margin in pixels.
/// \param attr The vertex or triangle attribute tensor.
/// \param triangle_indices The triangle_indices of the TriangleMesh.
/// \param primitive_ids The primitive ids from ComputePrimitiveInfoTexture().
/// \param primitive_uvs The primitive uvs from ComputePrimitiveInfoTexture().
/// \param sqrdistance The squared distances from ComputePrimitiveInfoTexture().
/// \param fill_value Fill value for the generated textures.
template <class TAttr, class TInt, bool VERTEX_ATTR>
core::Tensor BakeAttribute(int size,
                           float margin,
                           const core::Tensor &attr,
                           const core::Tensor &triangle_indices,
                           const core::Tensor &primitive_ids,
                           const core::Tensor &primitive_uvs,
                           const core::Tensor &sqrdistance,
                           TAttr fill_value) {
    core::SizeVector tex_shape({size, size});
    tex_shape.insert(tex_shape.end(), attr.GetShapeRef().begin() + 1,
                     attr.GetShapeRef().end());
    core::SizeVector components_shape(attr.GetShapeRef().begin() + 1,
                                      attr.GetShapeRef().end());
    const int num_components =
            components_shape.NumElements();  // is 1 for empty shape
    core::Tensor tex = core::Tensor::Empty(tex_shape, attr.GetDtype());

    const float threshold = (margin / size) * (margin / size);
    Eigen::Map<const Eigen::MatrixXf> sqrdistance_map(
            sqrdistance.GetDataPtr<float>(), size, size);
    Eigen::Map<Eigen::Matrix<TAttr, Eigen::Dynamic, Eigen::Dynamic>> tex_map(
            tex.GetDataPtr<TAttr>(), num_components, size * size);
    Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>>
            tid_map(primitive_ids.GetDataPtr<uint32_t>(), size, size);
    Eigen::Map<const Eigen::MatrixXf> uv_map(primitive_uvs.GetDataPtr<float>(),
                                             2, size * size);
    Eigen::Map<const Eigen::Matrix<TAttr, Eigen::Dynamic, Eigen::Dynamic>>
            attr_map(attr.GetDataPtr<TAttr>(), num_components,
                     attr.GetLength());
    Eigen::Map<const Eigen::Matrix<TInt, 3, Eigen::Dynamic>>
            triangle_indices_map(triangle_indices.GetDataPtr<TInt>(), 3,
                                 triangle_indices.GetLength());

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            const int64_t linear_idx = i * size + j;
            if (sqrdistance_map(j, i) <= threshold) {
                const uint32_t tid = tid_map(j, i);
                if (VERTEX_ATTR) {
                    const auto &a = attr_map.col(triangle_indices_map(0, tid));
                    const auto &b = attr_map.col(triangle_indices_map(1, tid));
                    const auto &c = attr_map.col(triangle_indices_map(2, tid));
                    TAttr u = uv_map(0, linear_idx);
                    TAttr v = uv_map(1, linear_idx);
                    tex_map.col(linear_idx) =
                            std::max<TAttr>(0, 1 - u - v) * a + u * b + v * c;
                } else {
                    tex_map.col(linear_idx) = attr_map.col(tid);
                }
            } else {
                tex_map.col(linear_idx).setConstant(fill_value);
            }
        }
    }

    return tex;
}

/// Computes textures with the primitive ids, primitive uvs, and the squared
/// distance to the closest primitive.
///
/// This is a helper function for the texture baking functions.
///
/// \param size The texture size.
/// \param primitive_ids The output tensor for the primitive ids.
/// \param primitive_uvs The output tensor for the primitive uvs.
/// \param sqrdistance The output tensor for the squared distances.
/// \param texture_uvs Input tensor with the texture uvs.
void ComputePrimitiveInfoTexture(int size,
                                 core::Tensor &primitive_ids,
                                 core::Tensor &primitive_uvs,
                                 core::Tensor &sqrdistance,
                                 const core::Tensor &texture_uvs) {
    const int64_t num_triangles = texture_uvs.GetLength();

    // Generate vertices for each triangle using (u,v,0) as position.
    core::Tensor vertices({num_triangles * 3, 3}, core::Float32);
    {
        const float *uv_ptr = texture_uvs.GetDataPtr<float>();
        float *v_ptr = vertices.GetDataPtr<float>();
        for (int64_t i = 0; i < texture_uvs.GetLength(); ++i) {
            for (int64_t j = 0; j < 3; ++j) {
                v_ptr[i * 9 + j * 3 + 0] = uv_ptr[i * 6 + j * 2 + 0];
                v_ptr[i * 9 + j * 3 + 1] = uv_ptr[i * 6 + j * 2 + 1];
                v_ptr[i * 9 + j * 3 + 2] = 0;
            }
        }
    }
    core::Tensor triangle_indices =
            core::Tensor::Empty({num_triangles, 3}, core::UInt32);
    std::iota(triangle_indices.GetDataPtr<uint32_t>(),
              triangle_indices.GetDataPtr<uint32_t>() +
                      triangle_indices.NumElements(),
              0);

    RaycastingScene scene;
    scene.AddTriangles(vertices, triangle_indices);

    core::Tensor query_points =
            core::Tensor::Empty({size, size, 3}, core::Float32);
    float *ptr = query_points.GetDataPtr<float>();
    for (int i = 0; i < size; ++i) {
        float v = 1 - (i + 0.5f) / size;
        for (int j = 0; j < size; ++j) {
            float u = (j + 0.5f) / size;
            ptr[i * size * 3 + j * 3 + 0] = u;
            ptr[i * size * 3 + j * 3 + 1] = v;
            ptr[i * size * 3 + j * 3 + 2] = 0;
        }
    }

    auto ans = scene.ComputeClosestPoints(query_points);

    Eigen::Map<Eigen::MatrixXf> query_points_map(
            query_points.GetDataPtr<float>(), 3, size * size);
    Eigen::Map<Eigen::MatrixXf> closest_points_map(
            ans["points"].GetDataPtr<float>(), 3, size * size);
    sqrdistance = core::Tensor::Empty({size, size}, core::Float32);
    Eigen::Map<Eigen::VectorXf> sqrdistance_map(sqrdistance.GetDataPtr<float>(),
                                                size * size);
    sqrdistance_map =
            (closest_points_map - query_points_map).colwise().squaredNorm();
    primitive_ids = ans["primitive_ids"];
    primitive_uvs = ans["primitive_uvs"];
}
void UpdateMaterialTextures(
        std::unordered_map<std::string, core::Tensor> &textures,
        visualization::rendering::Material &material) {
    for (auto &tex : textures) {
        core::SizeVector element_shape(tex.second.GetShapeRef().begin() + 2,
                                       tex.second.GetShapeRef().end());
        core::SizeVector shape(tex.second.GetShapeRef().begin(),
                               tex.second.GetShapeRef().begin() + 2);
        if (tex.second.NumDims() > 2) {
            shape.push_back(element_shape.NumElements());
        }

        core::Tensor img_data = tex.second.Reshape(shape);
        material.SetTextureMap(tex.first, Image(img_data));
    }
}

}  // namespace
std::unordered_map<std::string, core::Tensor>
TriangleMesh::BakeVertexAttrTextures(
        int size,
        const std::unordered_set<std::string> &vertex_attr,
        double margin,
        double fill,
        bool update_material) {
    if (!vertex_attr.size()) {
        return std::unordered_map<std::string, core::Tensor>();
    }
    if (!triangle_attr_.Contains("texture_uvs")) {
        utility::LogError("Cannot find triangle attribute 'texture_uvs'");
    }

    core::Tensor texture_uvs =
            triangle_attr_.at("texture_uvs").To(core::Device()).Contiguous();
    core::AssertTensorShape(texture_uvs, {core::None, 3, 2});
    core::AssertTensorDtype(texture_uvs, {core::Float32});

    core::Tensor vertices({GetTriangleIndices().GetLength() * 3, 3},
                          core::Float32);
    {
        float *uv_ptr = texture_uvs.GetDataPtr<float>();
        float *v_ptr = vertices.GetDataPtr<float>();
        for (int64_t i = 0; i < texture_uvs.GetLength(); ++i) {
            for (int64_t j = 0; j < 3; ++j) {
                v_ptr[i * 9 + j * 3 + 0] = uv_ptr[i * 6 + j * 2 + 0];
                v_ptr[i * 9 + j * 3 + 1] = uv_ptr[i * 6 + j * 2 + 1];
                v_ptr[i * 9 + j * 3 + 2] = 0;
            }
        }
    }
    core::Tensor triangle_indices =
            core::Tensor::Empty(GetTriangleIndices().GetShape(), core::UInt32);
    std::iota(triangle_indices.GetDataPtr<uint32_t>(),
              triangle_indices.GetDataPtr<uint32_t>() +
                      triangle_indices.NumElements(),
              0);

    core::Tensor primitive_ids, primitive_uvs, sqrdistance;
    ComputePrimitiveInfoTexture(size, primitive_ids, primitive_uvs, sqrdistance,
                                texture_uvs);

    std::unordered_map<std::string, core::Tensor> result;
    for (auto attr : vertex_attr) {
        if (!vertex_attr_.Contains(attr)) {
            utility::LogError("Cannot find vertex attribute '{}'", attr);
        }
        core::Tensor tensor =
                vertex_attr_.at(attr).To(core::Device()).Contiguous();
        DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(
                tensor.GetDtype(), GetTriangleIndices().GetDtype(), [&]() {
                    core::Tensor tex = BakeAttribute<scalar_t, int_t, true>(
                            size, margin, tensor, GetTriangleIndices(),
                            primitive_ids, primitive_uvs, sqrdistance,
                            scalar_t(fill));
                    result[attr] = tex;
                });
    }
    if (update_material) {
        UpdateMaterialTextures(result, this->GetMaterial());
    }

    return result;
}

std::unordered_map<std::string, core::Tensor>
TriangleMesh::BakeTriangleAttrTextures(
        int size,
        const std::unordered_set<std::string> &triangle_attr,
        double margin,
        double fill,
        bool update_material) {
    if (!triangle_attr.size()) {
        return std::unordered_map<std::string, core::Tensor>();
    }
    if (!triangle_attr_.Contains("texture_uvs")) {
        utility::LogError("Cannot find triangle attribute 'texture_uvs'");
    }

    core::Tensor texture_uvs =
            triangle_attr_.at("texture_uvs").To(core::Device()).Contiguous();
    core::AssertTensorShape(texture_uvs, {core::None, 3, 2});
    core::AssertTensorDtype(texture_uvs, {core::Float32});

    core::Tensor primitive_ids, primitive_uvs, sqrdistance;
    ComputePrimitiveInfoTexture(size, primitive_ids, primitive_uvs, sqrdistance,
                                texture_uvs);

    std::unordered_map<std::string, core::Tensor> result;
    for (auto attr : triangle_attr) {
        if (!triangle_attr_.Contains(attr)) {
            utility::LogError("Cannot find triangle attribute '{}'", attr);
        }
        core::Tensor tensor =
                triangle_attr_.at(attr).To(core::Device()).Contiguous();
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(tensor.GetDtype(), [&]() {
            core::Tensor tex;
            if (GetTriangleIndices().GetDtype() == core::Int32) {
                tex = BakeAttribute<scalar_t, int32_t, false>(
                        size, margin, tensor, GetTriangleIndices(),
                        primitive_ids, primitive_uvs, sqrdistance,
                        scalar_t(fill));
            } else if (GetTriangleIndices().GetDtype() == core::Int64) {
                tex = BakeAttribute<scalar_t, int64_t, false>(
                        size, margin, tensor, GetTriangleIndices(),
                        primitive_ids, primitive_uvs, sqrdistance,
                        scalar_t(fill));
            } else {
                utility::LogError("Unsupported triangle indices data type.");
            }
            result[attr] = tex;
        });
    }
    if (update_material) {
        UpdateMaterialTextures(result, this->GetMaterial());
    }

    return result;
}

TriangleMesh TriangleMesh::ExtrudeRotation(double angle,
                                           const core::Tensor &axis,
                                           int resolution,
                                           double translation,
                                           bool capping) const {
    using namespace vtkutils;
    return ExtrudeRotationTriangleMesh(*this, angle, axis, resolution,
                                       translation, capping);
}

TriangleMesh TriangleMesh::ExtrudeLinear(const core::Tensor &vector,
                                         double scale,
                                         bool capping) const {
    using namespace vtkutils;
    return ExtrudeLinearTriangleMesh(*this, vector, scale, capping);
}

int TriangleMesh::PCAPartition(int max_faces) {
    core::Tensor verts = GetVertexPositions();
    core::Tensor tris = GetTriangleIndices();
    if (!tris.GetLength()) {
        utility::LogError("Mesh must have at least one face.");
    }
    core::Tensor tris_centers = verts.IndexGet({tris}).Mean({1});

    int num_parititions;
    core::Tensor partition_ids;
    std::tie(num_parititions, partition_ids) =
            kernel::pcapartition::PCAPartition(tris_centers, max_faces);
    SetTriangleAttr("partition_ids", partition_ids.To(GetDevice()));
    return num_parititions;
}

/// A helper to compute new vertex indices out of vertex mask.
/// \param tris_cpu CPU tensor with triangle indices to update.
/// \param vertex_mask CPU tensor with the mask for vertices.
template <typename T>
static void UpdateTriangleIndicesByVertexMask(core::Tensor &tris_cpu,
                                              const core::Tensor &vertex_mask) {
    int64_t num_verts = vertex_mask.GetLength();
    int64_t num_tris = tris_cpu.GetLength();
    const T *vertex_mask_ptr = vertex_mask.GetDataPtr<T>();
    std::vector<T> prefix_sum(num_verts + 1, 0);
    utility::InclusivePrefixSum(vertex_mask_ptr, vertex_mask_ptr + num_verts,
                                &prefix_sum[1]);

    // update triangle indices
    T *vert_idx_ptr = tris_cpu.GetDataPtr<T>();
    for (int64_t i = 0; i < num_tris * 3; ++i) {
        vert_idx_ptr[i] = prefix_sum[vert_idx_ptr[i]];
    }
}

/// A helper to copy mesh attributes.
/// \param dst destination mesh
/// \param src source mesh
/// \param vertex_mask vertex mask of the source mesh
/// \param tri_mask triangle mask of the source mesh
static void CopyAttributesByMasks(TriangleMesh &dst,
                                  const TriangleMesh &src,
                                  const core::Tensor &vertex_mask,
                                  const core::Tensor &tri_mask) {
    if (src.HasVertexPositions() && dst.HasVertexPositions()) {
        for (auto item : src.GetVertexAttr()) {
            if (!dst.HasVertexAttr(item.first)) {
                dst.SetVertexAttr(item.first,
                                  item.second.IndexGet({vertex_mask}));
            }
        }
    }

    if (src.HasTriangleIndices() && dst.HasTriangleIndices()) {
        for (auto item : src.GetTriangleAttr()) {
            if (!dst.HasTriangleAttr(item.first)) {
                dst.SetTriangleAttr(item.first,
                                    item.second.IndexGet({tri_mask}));
            }
        }
    }
}

TriangleMesh TriangleMesh::SelectFacesByMask(const core::Tensor &mask) const {
    if (!HasVertexPositions()) {
        utility::LogWarning(
                "[SelectFacesByMask] mesh has no vertex positions.");
        return {};
    }
    if (!HasTriangleIndices()) {
        utility::LogWarning(
                "[SelectFacesByMask] mesh has no triangle indices.");
        return {};
    }

    core::AssertTensorShape(mask, {GetTriangleIndices().GetLength()});
    core::AssertTensorDtype(mask, core::Bool);
    GetTriangleAttr().AssertSizeSynchronized();
    GetVertexAttr().AssertSizeSynchronized();

    // select triangles
    core::Tensor tris = GetTriangleIndices().IndexGet({mask});
    core::Tensor tris_cpu = tris.To(core::Device()).Contiguous();

    // create mask for vertices that are part of the selected faces
    const int64_t num_verts = GetVertexPositions().GetLength();
    // empty tensor to further construct the vertex mask
    core::Tensor vertex_mask;

    DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(tris_cpu.GetDtype(), tris, [&]() {
        vertex_mask = core::Tensor::Zeros(
                {num_verts}, core::Dtype::FromType<scalar_tris_t>());
        const int64_t num_tris = tris_cpu.GetLength();
        scalar_tris_t *vertex_mask_ptr =
                vertex_mask.GetDataPtr<scalar_tris_t>();
        scalar_tris_t *vert_idx_ptr = tris_cpu.GetDataPtr<scalar_tris_t>();
        // mask for the vertices, which are used in the triangles
        for (int64_t i = 0; i < num_tris * 3; ++i) {
            vertex_mask_ptr[vert_idx_ptr[i]] = 1;
        }
        UpdateTriangleIndicesByVertexMask<scalar_tris_t>(tris_cpu, vertex_mask);
    });

    tris = tris_cpu.To(GetDevice());
    vertex_mask = vertex_mask.To(GetDevice(), core::Bool);
    core::Tensor verts = GetVertexPositions().IndexGet({vertex_mask});
    TriangleMesh result(verts, tris);

    CopyAttributesByMasks(result, *this, vertex_mask, mask);

    return result;
}

/// brief Static negative checker for signed integer types
template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                          !std::is_same<T, bool>::value &&
                                          std::is_signed<T>::value,
                                  T>::type * = nullptr>
static bool IsNegative(T val) {
    return val < 0;
}

/// brief Overloaded static negative checker for unsigned integer types.
/// It unconditionally returns false, but we need it for template functions.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                          !std::is_same<T, bool>::value &&
                                          !std::is_signed<T>::value,
                                  T>::type * = nullptr>
static bool IsNegative(T val) {
    return false;
}

TriangleMesh TriangleMesh::SelectByIndex(const core::Tensor &indices,
                                         bool copy_attributes /*=true*/) const {
    core::AssertTensorShape(indices, {indices.GetLength()});
    if (indices.NumElements() == 0) {
        return {};
    }
    if (!HasVertexPositions()) {
        utility::LogWarning("[SelectByIndex] TriangleMesh has no vertices.");
        return {};
    }
    GetVertexAttr().AssertSizeSynchronized();

    // we allow indices of an integral type only
    core::Dtype::DtypeCode indices_dtype_code =
            indices.GetDtype().GetDtypeCode();
    if (indices_dtype_code != core::Dtype::DtypeCode::Int &&
        indices_dtype_code != core::Dtype::DtypeCode::UInt) {
        utility::LogError(
                "[SelectByIndex] indices are not of integral type {}.",
                indices.GetDtype().ToString());
    }
    core::Tensor indices_cpu = indices.To(core::Device()).Contiguous();
    core::Tensor tris_cpu, tri_mask;
    core::Dtype tri_dtype;
    if (HasTriangleIndices()) {
        GetTriangleAttr().AssertSizeSynchronized();
        tris_cpu = GetTriangleIndices().To(core::Device()).Contiguous();
        // bool mask for triangles.
        tri_mask = core::Tensor::Zeros({tris_cpu.GetLength()}, core::Bool);
        tri_dtype = tris_cpu.GetDtype();
    } else {
        utility::LogWarning("TriangleMesh has no triangle indices.");
        tri_dtype = core::Int64;
    }

    // int mask to select vertices for the new mesh.  We need it as int as we
    // will use its values to sum up and get the map of new indices
    core::Tensor vertex_mask =
            core::Tensor::Zeros({GetVertexPositions().GetLength()}, tri_dtype);

    DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(tri_dtype, tris, [&]() {
        DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(
                indices_cpu.GetDtype(), indices, [&]() {
                    const int64_t num_tris = tris_cpu.GetLength();
                    const int64_t num_verts = vertex_mask.GetLength();

                    // compute the vertices mask
                    scalar_tris_t *vertex_mask_ptr =
                            vertex_mask.GetDataPtr<scalar_tris_t>();
                    const scalar_indices_t *indices_ptr =
                            indices_cpu.GetDataPtr<scalar_indices_t>();
                    for (int64_t i = 0; i < indices.GetLength(); ++i) {
                        if (IsNegative(indices_ptr[i]) ||
                            indices_ptr[i] >=
                                    static_cast<scalar_indices_t>(num_verts)) {
                            utility::LogWarning(
                                    "[SelectByIndex] indices contains index {} "
                                    "out of range. "
                                    "It is ignored.",
                                    indices_ptr[i]);
                            continue;
                        }
                        vertex_mask_ptr[indices_ptr[i]] = 1;
                    }

                    if (tri_mask.GetDtype() == core::Undefined) {
                        // we don't need to compute triangles, if there are none
                        return;
                    }

                    // Build the triangle mask
                    scalar_tris_t *tris_cpu_ptr =
                            tris_cpu.GetDataPtr<scalar_tris_t>();
                    bool *tri_mask_ptr = tri_mask.GetDataPtr<bool>();
                    for (int64_t i = 0; i < num_tris; ++i) {
                        if (vertex_mask_ptr[tris_cpu_ptr[3 * i]] == 1 &&
                            vertex_mask_ptr[tris_cpu_ptr[3 * i + 1]] == 1 &&
                            vertex_mask_ptr[tris_cpu_ptr[3 * i + 2]] == 1) {
                            tri_mask_ptr[i] = true;
                        }
                    }
                    // select only needed triangles
                    tris_cpu = tris_cpu.IndexGet({tri_mask});
                    // update the triangle indices
                    UpdateTriangleIndicesByVertexMask<scalar_tris_t>(
                            tris_cpu, vertex_mask);
                });
    });

    // send the vertex mask and triangle mask to original device and apply to
    // vertices
    vertex_mask = vertex_mask.To(GetDevice(), core::Bool);
    if (tri_mask.NumElements() > 0) {  // To() needs non-empty tensor
        tri_mask = tri_mask.To(GetDevice());
    }
    core::Tensor new_vertices = GetVertexPositions().IndexGet({vertex_mask});
    TriangleMesh result(GetDevice());
    result.SetVertexPositions(new_vertices);
    if (tris_cpu.NumElements() > 0) {  // To() needs non-empty tensor
        result.SetTriangleIndices(tris_cpu.To(GetDevice()));
    }
    if (copy_attributes)
        CopyAttributesByMasks(result, *this, vertex_mask, tri_mask);

    return result;
}

TriangleMesh TriangleMesh::RemoveUnreferencedVertices() {
    if (!HasVertexPositions() || GetVertexPositions().GetLength() == 0) {
        utility::LogWarning(
                "[RemoveUnreferencedVertices] TriangleMesh has no vertices.");
        return *this;
    }
    GetVertexAttr().AssertSizeSynchronized();

    core::Dtype tri_dtype = HasTriangleIndices()
                                    ? GetTriangleIndices().GetDtype()
                                    : core::Int64;

    int64_t num_verts_old = GetVertexPositions().GetLength();
    // int mask for vertices as we need to remap indices.
    core::Tensor vertex_mask = core::Tensor::Zeros({num_verts_old}, tri_dtype);

    if (!HasTriangleIndices() || GetTriangleIndices().GetLength() == 0) {
        utility::LogWarning(
                "[RemoveUnreferencedVertices] TriangleMesh has no triangles. "
                "Removing all vertices.");
        // in this case we need to empty vertices and their attributes
    } else {
        GetTriangleAttr().AssertSizeSynchronized();
        core::Tensor tris_cpu =
                GetTriangleIndices().To(core::Device()).Contiguous();
        DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(tri_dtype, tris, [&]() {
            scalar_tris_t *tris_ptr = tris_cpu.GetDataPtr<scalar_tris_t>();
            scalar_tris_t *vertex_mask_ptr =
                    vertex_mask.GetDataPtr<scalar_tris_t>();
            for (int i = 0; i < tris_cpu.GetLength(); i++) {
                vertex_mask_ptr[tris_ptr[3 * i]] = 1;
                vertex_mask_ptr[tris_ptr[3 * i + 1]] = 1;
                vertex_mask_ptr[tris_ptr[3 * i + 2]] = 1;
            }

            UpdateTriangleIndicesByVertexMask<scalar_tris_t>(tris_cpu,
                                                             vertex_mask);
        });
        SetTriangleIndices(tris_cpu.To(GetDevice()));
    }

    // send the vertex mask to original device and apply to
    // vertices
    vertex_mask = vertex_mask.To(GetDevice(), core::Bool);
    for (auto item : GetVertexAttr()) {
        SetVertexAttr(item.first, item.second.IndexGet({vertex_mask}));
    }

    utility::LogDebug(
            "[RemoveUnreferencedVertices] {:d} vertices have been removed.",
            static_cast<int>(num_verts_old - GetVertexPositions().GetLength()));

    return *this;
}

namespace {

core::Tensor Project(const core::Tensor &t_xyz,  // contiguous {...,3}
                     const core::Tensor &t_intrinsic_matrix,  // {3,3}
                     const core::Tensor &t_extrinsic_matrix,  // {4,4}
                     core::Tensor &t_xy) {  // contiguous {...,2}
    auto xy_shape = t_xyz.GetShape();
    auto dt = t_xyz.GetDtype();
    auto t_K = t_intrinsic_matrix.To(dt).Contiguous(),
         t_T = t_extrinsic_matrix.To(dt).Contiguous();
    xy_shape[t_xyz.NumDims() - 1] = 2;
    if (t_xy.GetDtype() != dt || t_xy.GetShape() != xy_shape) {
        t_xy = core::Tensor(xy_shape, dt);
    }
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dt, [&]() {
        // Eigen is column major
        Eigen::Map<Eigen::MatrixX<scalar_t>> xy(t_xy.GetDataPtr<scalar_t>(), 2,
                                                t_xy.NumElements() / 2);
        Eigen::Map<const Eigen::MatrixX<scalar_t>> xyz(
                t_xyz.GetDataPtr<scalar_t>(), 3, t_xyz.NumElements() / 3);
        Eigen::Map<const Eigen::Matrix3<scalar_t>> KT(
                t_K.GetDataPtr<scalar_t>());
        Eigen::Map<const Eigen::Matrix4<scalar_t>> TT(
                t_T.GetDataPtr<scalar_t>());

        auto K = KT.transpose();
        auto T = TT.transpose();
        auto R = T.topLeftCorner<3, 3>();
        auto t = T.topRightCorner<3, 1>();
        auto pxyz = (K * ((R * xyz).colwise() + t)).array();
        xy = pxyz.topRows<2>().rowwise() / pxyz.bottomRows<1>();
    });
    return t_xy;  // contiguous {...,2}
}

/// Estimate minimum sqr distance from a set of points to a set of cameras.
float get_min_cam_sqrdistance(
        const core::Tensor &positions,
        const std::vector<core::Tensor> &extrinsic_matrices) {
    const size_t MAXPTS = 10000;
    core::Tensor cam_loc({int64_t(extrinsic_matrices.size()), 3},
                         core::Float32);
    for (size_t k = 0; k < extrinsic_matrices.size(); ++k) {
        const core::Tensor RT = extrinsic_matrices[k].Slice(0, 0, 3);
        cam_loc[k] =
                -RT.Slice(1, 0, 3).T().Matmul(RT.Slice(1, 3, 4)).Reshape({-1});
    }
    size_t npts = positions.GetShape(0);
    const core::Tensor pos_sample =
            npts > MAXPTS ? positions.Slice(0, 0, -1, npts / MAXPTS)
                          : positions;
    auto nns = core::nns::NearestNeighborSearch(pos_sample);
    nns.KnnIndex();
    float min_sqrdistance = nns.KnnSearch(cam_loc, 1)
                                    .second.Min({0, 1})
                                    .To(core::Device(), core::Float32)
                                    .Item<float>();
    return min_sqrdistance;
}

}  // namespace

Image TriangleMesh::ProjectImagesToAlbedo(
        const std::vector<Image> &images,
        const std::vector<core::Tensor> &intrinsic_matrices,
        const std::vector<core::Tensor> &extrinsic_matrices,
        int tex_size /*=1024*/,
        bool update_material /*=true*/) {
    if (!GetDevice().IsCPU() || !Image::HAVE_IPP) {
        utility::LogError(
                "ProjectImagesToAlbedo is only supported on x86_64 CPU "
                "devices.");
    }
    using core::None;
    using tk = core::TensorKey;
    constexpr float EPS = 1e-6;
    if (!HasTriangleAttr("texture_uvs")) {
        utility::LogError(
                "TriangleMesh does not contain 'texture_uvs'. Please compute "
                "it with ComputeUVAtlas() first.");
    }
    core::Tensor texture_uvs =
            GetTriangleAttr("texture_uvs").To(core::Device()).Contiguous();
    core::AssertTensorShape(texture_uvs, {core::None, 3, 2});
    core::AssertTensorDtype(texture_uvs, {core::Float32});

    if (images.size() != extrinsic_matrices.size() ||
        images.size() != intrinsic_matrices.size()) {
        utility::LogError(
                "Received {} images, but {} extrinsic matrices and {} "
                "intrinsic matrices.",
                images.size(), extrinsic_matrices.size(),
                intrinsic_matrices.size());
    }

    // softmax_shift is used to prevent overflow in the softmax function.
    // softmax_shift is set so that max value of weighting function is exp(64),
    // well within float range. (exp(89.f) is inf)
    float min_sqr_distance =
            get_min_cam_sqrdistance(GetVertexPositions(), extrinsic_matrices);
    const float softmax_scale = 20 * min_sqr_distance;
    constexpr float softmax_shift = 10.f, LOG_FLT_MAX = 88.f;
    // (u,v) -> (x,y,z) : {tex_size, tex_size, 3}
    // margin \propto tex_size (default 2px for 512 size texture as in
    // BakeVertexAttrTextures()). Should be at least 1px.
    core::Tensor position_map = BakeVertexAttrTextures(
            tex_size, {"positions"}, /*margin=*/(tex_size + 255) / 256,
            /*fill=*/0,
            /*update_material=*/false)["positions"];
    core::Tensor albedo =
            core::Tensor::Zeros({tex_size, tex_size, 4}, core::Float32);
    albedo.Slice(2, 3, 4).Fill(EPS);  // regularize weight
    std::mutex albedo_mutex;

    RaycastingScene rcs;
    rcs.AddTriangles(*this);

    // setup working data for each task.
    size_t max_workers = tbb::this_task_arena::max_concurrency();
    // Tensor copy ctor does shallow copies - OK for empty tensors.
    std::vector<core::Tensor> this_albedo(max_workers,
                                          core::Tensor({}, core::Float32)),
            weighted_image(max_workers, core::Tensor({}, core::Float32)),
            uv2xy(max_workers, core::Tensor({}, core::Float32)),
            uvrays(max_workers, core::Tensor({}, core::Float32));

    auto project_one_image = [&](size_t i, tbb::feeder<size_t> &feeder) {
        size_t widx = tbb::this_task_arena::current_thread_index();
        // initialize task variables
        if (!this_albedo[widx].GetShape().IsCompatible(
                    {tex_size, tex_size, 4})) {
            this_albedo[widx] =
                    core::Tensor::Empty({tex_size, tex_size, 4}, core::Float32);
            uvrays[widx] =
                    core::Tensor::Empty({tex_size, tex_size, 6}, core::Float32);
        }
        auto width = images[i].GetCols(), height = images[i].GetRows();
        if (!weighted_image[widx].GetShape().IsCompatible({height, width, 4})) {
            weighted_image[widx] =
                    core::Tensor({height, width, 4}, core::Float32);
        }
        core::AssertTensorShape(intrinsic_matrices[i], {3, 3});
        core::AssertTensorShape(extrinsic_matrices[i], {4, 4});

        // A. Get image space weight matrix, as inverse of pixel
        // footprint on the mesh.
        auto rays = RaycastingScene::CreateRaysPinhole(
                intrinsic_matrices[i], extrinsic_matrices[i], width, height);
        core::Tensor cam_loc =
                rays.GetItem({tk::Index(0), tk::Index(0), tk::Slice(0, 3, 1)});

        // A nested parallel_for's threads must be isolated from the threads
        // running this paralel_for, else we get BAD ACCESS errors.
        auto result = tbb::this_task_arena::isolate(
                [&rays, &rcs]() { return rcs.CastRays(rays); });
        // Eigen is column-major order
        Eigen::Map<Eigen::ArrayXXf> normals_e(
                result["primitive_normals"].GetDataPtr<float>(), 3,
                width * height);
        Eigen::Map<Eigen::ArrayXXf> rays_e(rays.GetDataPtr<float>(), 6,
                                           width * height);
        Eigen::Map<Eigen::ArrayXXf> t_hit(result["t_hit"].GetDataPtr<float>(),
                                          1, width * height);
        auto depth = t_hit * rays_e.bottomRows<3>().colwise().norm().array();
        // removing this eval() increase runtime a lot (?)
        auto rays_dir = rays_e.bottomRows<3>().colwise().normalized().eval();
        auto pixel_foreshortening = (normals_e * rays_dir)
                                            .colwise()
                                            .sum()
                                            .abs();  // ignore face orientation
        // fix for bad normals
        auto inv_footprint =
                pixel_foreshortening.isNaN().select(0, pixel_foreshortening) /
                (depth * depth);
        utility::LogDebug(
                "[ProjectImagesToAlbedo] Image {}, weight (inv_footprint) "
                "range: {}-{}",
                i, inv_footprint.minCoeff(), inv_footprint.maxCoeff());
        weighted_image[widx].Slice(2, 0, 3) =
                images[i].To(core::Float32).AsTensor();  // range: [0,1]
        Eigen::Map<Eigen::MatrixXf> weighted_image_e(
                weighted_image[widx].GetDataPtr<float>(), 4, width * height);
        weighted_image_e.bottomRows<1>() = inv_footprint;

        // B. Get texture space (u,v) -> (x,y) map and valid domain in
        // uv space.
        uvrays[widx].GetItem({tk::Slice(0, None, 1), tk::Slice(0, None, 1),
                              tk::Slice(0, 3, 1)}) = cam_loc;
        uvrays[widx].GetItem({tk::Slice(0, None, 1), tk::Slice(0, None, 1),
                              tk::Slice(3, 6, 1)}) = position_map - cam_loc;
        // A nested parallel_for's threads must be isolated from the threads
        // running this paralel_for, else we get BAD ACCESS errors.
        result = tbb::this_task_arena::isolate(
                [&rcs, &uvrays, widx]() { return rcs.CastRays(uvrays[widx]); });
        auto &t_hit_uv = result["t_hit"];

        Project(position_map, intrinsic_matrices[i], extrinsic_matrices[i],
                uv2xy[widx]);  // {ts, ts, 2}
        // Disable self-occluded points
        for (float *p_uv2xy = uv2xy[widx].GetDataPtr<float>(),
                   *p_t_hit = t_hit_uv.GetDataPtr<float>();
             p_uv2xy <
             uv2xy[widx].GetDataPtr<float>() + uv2xy[widx].NumElements();
             p_uv2xy += 2, ++p_t_hit) {
            if (*p_t_hit < 1 - EPS) *p_uv2xy = *(p_uv2xy + 1) = -1.f;
        }
        core::Tensor uv2xy2 =
                uv2xy[widx].Permute({2, 0, 1}).Contiguous();  // {2, ts, ts}

        // C. Interpolate weighted image to weighted texture
        // albedo[u,v] = image[ i[u,v], j[u,v] ]
        this_albedo[widx].Fill(0.f);
        IPP_CALL(ipp::Remap, weighted_image[widx], /*{height, width, 4} f32*/
                 uv2xy2[0],                        /* {texsz, texsz} f32*/
                 uv2xy2[1],                        /* {texsz, texsz} f32*/
                 this_albedo[widx],                /*{texsz, texsz, 4} f32*/
                 t::geometry::Image::InterpType::Linear);
        // Weights can become negative with higher order interpolation

        std::unique_lock<std::mutex> albedo_lock{albedo_mutex};
        // ^^^ released when lambda returns.
        for (auto p_albedo = albedo.GetDataPtr<float>(),
                  p_this_albedo = this_albedo[widx].GetDataPtr<float>();
             p_albedo < albedo.GetDataPtr<float>() + albedo.NumElements();
             p_albedo += 4, p_this_albedo += 4) {
            float softmax_weight =
                    exp(std::min(LOG_FLT_MAX, softmax_scale * p_this_albedo[3] -
                                                      softmax_shift));
            for (auto k = 0; k < 3; ++k)
                p_albedo[k] += p_this_albedo[k] * softmax_weight;
            p_albedo[3] += softmax_weight;
        }
    };

    std::vector<size_t> range(images.size(), 0);
    std::iota(range.begin(), range.end(), 0);
    tbb::parallel_for_each(range, project_one_image);
    albedo.Slice(2, 0, 3) /= albedo.Slice(2, 3, 4);

    Image albedo_texture{
            (albedo.Slice(2, 0, 3) * 255.f).Clip_(0.f, 255.f).To(core::UInt8)};
    if (update_material) {
        if (!HasMaterial()) {
            SetMaterial(visualization::rendering::Material());
            GetMaterial().SetDefaultProperties();  // defaultUnlit
        }
        GetMaterial().SetAlbedoMap(albedo_texture);
    }
    return albedo_texture;
}

namespace {
template <typename T,
          typename std::enable_if<std::is_integral<T>::value &&
                                          !std::is_same<T, bool>::value,
                                  T>::type * = nullptr>
using Edge = std::tuple<T, T>;
}

/// brief Helper function to get an edge with ordered vertex indices.
template <typename T>
static inline Edge<T> GetOrderedEdge(T vidx0, T vidx1) {
    return (vidx0 < vidx1) ? Edge<T>{vidx0, vidx1} : Edge<T>{vidx1, vidx0};
}

/// brief Helper
///
template <typename T>
static std::unordered_map<Edge<T>,
                          std::vector<size_t>,
                          utility::hash_tuple<Edge<T>>>
GetEdgeToTrianglesMap(const core::Tensor &tris_cpu) {
    std::unordered_map<Edge<T>, std::vector<size_t>,
                       utility::hash_tuple<Edge<T>>>
            tris_per_edge;
    auto AddEdge = [&](T vidx0, T vidx1, int64_t tidx) {
        tris_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(tidx);
    };
    const T *tris_ptr = tris_cpu.GetDataPtr<T>();
    for (int64_t tidx = 0; tidx < tris_cpu.GetLength(); ++tidx) {
        const T *triangle = &tris_ptr[3 * tidx];
        AddEdge(triangle[0], triangle[1], tidx);
        AddEdge(triangle[1], triangle[2], tidx);
        AddEdge(triangle[2], triangle[0], tidx);
    }
    return tris_per_edge;
}

TriangleMesh TriangleMesh::RemoveNonManifoldEdges() {
    if (!HasVertexPositions() || GetVertexPositions().GetLength() == 0) {
        utility::LogWarning(
                "[RemoveNonManifildEdges] TriangleMesh has no vertices.");
        return *this;
    }

    if (!HasTriangleIndices() || GetTriangleIndices().GetLength() == 0) {
        utility::LogWarning(
                "[RemoveNonManifoldEdges] TriangleMesh has no triangles.");
        return *this;
    }

    GetVertexAttr().AssertSizeSynchronized();
    GetTriangleAttr().AssertSizeSynchronized();

    core::Tensor tris_cpu =
            GetTriangleIndices().To(core::Device()).Contiguous();

    if (!HasTriangleAttr("areas")) {
        ComputeTriangleAreas();
    }
    core::Tensor tri_areas_cpu =
            GetTriangleAttr("areas").To(core::Device()).Contiguous();

    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(
            GetVertexPositions().GetDtype(), tris_cpu.GetDtype(), [&]() {
                scalar_t *tri_areas_ptr = tri_areas_cpu.GetDataPtr<scalar_t>();
                auto edges_to_tris = GetEdgeToTrianglesMap<int_t>(tris_cpu);

                // lambda to compare triangles areas by index
                auto area_greater_compare = [&tri_areas_ptr](size_t lhs,
                                                             size_t rhs) {
                    return tri_areas_ptr[lhs] > tri_areas_ptr[rhs];
                };

                // go through all edges and for those that have more than 2
                // triangles attached, remove the triangles with the minimal
                // area
                for (auto &kv : edges_to_tris) {
                    // remove all triangles which are already marked for removal
                    // (area < 0) note, the erasing of triangles happens
                    // afterwards
                    auto tris_end = std::remove_if(
                            kv.second.begin(), kv.second.end(),
                            [=](size_t t) { return tri_areas_ptr[t] < 0; });
                    // count non-removed triangles (with area > 0).
                    int n_tris = std::distance(kv.second.begin(), tris_end);

                    if (n_tris <= 2) {
                        // nothing to do here as either:
                        // - all triangles of the edge are already marked for
                        // deletion
                        // - the edge is manifold: it has 1 or 2 triangles with
                        //   a non-negative area
                        continue;
                    }

                    // now erase all triangle indices already marked for removal
                    kv.second.erase(tris_end, kv.second.end());

                    // find first to triangles with the maximal area
                    std::nth_element(kv.second.begin(), kv.second.begin() + 1,
                                     kv.second.end(), area_greater_compare);

                    // mark others for deletion
                    for (auto it = kv.second.begin() + 2; it < kv.second.end();
                         ++it) {
                        tri_areas_ptr[*it] = -1;
                    }
                }
            });

    // mask for triangles with positive area
    core::Tensor tri_mask = tri_areas_cpu.Gt(0.0).To(GetDevice());

    // pick up positive-area triangles (and their attributes)
    for (auto item : GetTriangleAttr()) {
        SetTriangleAttr(item.first, item.second.IndexGet({tri_mask}));
    }

    return *this;
}

core::Tensor TriangleMesh::GetNonManifoldEdges(
        bool allow_boundary_edges /* = true */) const {
    if (!HasVertexPositions()) {
        utility::LogWarning(
                "[GetNonManifoldEdges] TriangleMesh has no vertices.");
        return {};
    }

    if (!HasTriangleIndices()) {
        utility::LogWarning(
                "[GetNonManifoldEdges] TriangleMesh has no triangles.");
        return {};
    }

    core::Tensor result;
    core::Tensor tris_cpu =
            GetTriangleIndices().To(core::Device()).Contiguous();
    core::Dtype tri_dtype = tris_cpu.GetDtype();

    DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(tri_dtype, tris, [&]() {
        auto edges = GetEdgeToTrianglesMap<scalar_tris_t>(tris_cpu);
        std::vector<scalar_tris_t> non_manifold_edges;

        for (auto &kv : edges) {
            if ((allow_boundary_edges &&
                 (kv.second.size() < 1 || kv.second.size() > 2)) ||
                (!allow_boundary_edges && kv.second.size() != 2)) {
                non_manifold_edges.push_back(std::get<0>(kv.first));
                non_manifold_edges.push_back(std::get<1>(kv.first));
            }
        }

        result = core::Tensor(non_manifold_edges,
                              {(long int)non_manifold_edges.size() / 2, 2},
                              tri_dtype, GetTriangleIndices().GetDevice());
    });

    return result;
}

PointCloud TriangleMesh::SamplePointsUniformly(
        size_t number_of_points, bool use_triangle_normal /*=false*/) {
    if (number_of_points <= 0) {
        utility::LogError("number_of_points <= 0");
    }
    if (IsEmpty()) {
        utility::LogError("Input mesh is empty. Cannot sample points.");
    }
    if (!HasTriangleIndices()) {
        utility::LogError("Input mesh has no triangles. Cannot sample points.");
    }
    if (use_triangle_normal && !HasTriangleNormals()) {
        ComputeTriangleNormals(true);
    }
    if (!HasTriangleAttr("areas")) {
        ComputeTriangleAreas();  // Compute area of each triangle
    }
    if (!IsCPU()) {
        utility::LogWarning(
                "SamplePointsUniformly is implemented only on CPU. Computing "
                "on CPU.");
    }
    bool use_vert_normals = HasVertexNormals() && !use_triangle_normal;
    bool use_albedo =
            HasTriangleAttr("texture_uvs") && GetMaterial().HasAlbedoMap();
    bool use_vert_colors = HasVertexColors() && !use_albedo;

    auto cpu = core::Device();
    core::Tensor null_tensor({0}, core::Float32);  // zero size tensor
    auto triangles = GetTriangleIndices().To(cpu).Contiguous(),
         vertices = GetVertexPositions().To(cpu).Contiguous();
    auto float_dt = vertices.GetDtype();
    auto areas = GetTriangleAttr("areas").To(cpu, float_dt).Contiguous(),
         vertex_normals =
                 use_vert_normals
                         ? GetVertexNormals().To(cpu, float_dt).Contiguous()
                         : null_tensor,
         triangle_normals =
                 use_triangle_normal
                         ? GetTriangleNormals().To(cpu, float_dt).Contiguous()
                         : null_tensor,
         vertex_colors =
                 use_vert_colors
                         ? GetVertexColors().To(cpu, core::Float32).Contiguous()
                         : null_tensor,
         texture_uvs = use_albedo ? GetTriangleAttr("texture_uvs")
                                            .To(cpu, float_dt)
                                            .Contiguous()
                                  : null_tensor,
         // With correct range conversion [0,255] -> [0,1]
            albedo = use_albedo ? GetMaterial()
                                          .GetAlbedoMap()
                                          .To(core::Float32)
                                          .To(cpu)
                                          .AsTensor()
                                : null_tensor;
    if (use_vert_colors) {
        if (GetVertexColors().GetDtype() == core::UInt8) vertex_colors /= 255;
        if (GetVertexColors().GetDtype() == core::UInt16)
            vertex_colors /= 65535;
    }

    std::array<core::Tensor, 3> result =
            kernel::trianglemesh::SamplePointsUniformlyCPU(
                    triangles, vertices, areas, vertex_normals, vertex_colors,
                    triangle_normals, texture_uvs, albedo, number_of_points);

    PointCloud pcd(result[0]);
    if (use_vert_normals || use_triangle_normal) pcd.SetPointNormals(result[1]);
    if (use_albedo || use_vert_colors) pcd.SetPointColors(result[2]);
    return pcd.To(GetDevice());
}

core::Tensor TriangleMesh::ComputeMetrics(const TriangleMesh &mesh2,
                                          std::vector<Metric> metrics,
                                          MetricParameters params) const {
    if (IsEmpty() || mesh2.IsEmpty()) {
        utility::LogError("One or both input triangle meshes are empty!");
    }
    if (!IsCPU() || !mesh2.IsCPU()) {
        utility::LogWarning(
                "ComputeDistance is implemented only on CPU. Computing on "
                "CPU.");
    }
    auto cpu_mesh1 = To(core::Device("CPU:0")),
         cpu_mesh2 = mesh2.To(core::Device("CPU:0"));
    core::Tensor points1 =
            cpu_mesh1.SamplePointsUniformly(params.n_sampled_points)
                    .GetPointPositions();
    core::Tensor points2 =
            cpu_mesh2.SamplePointsUniformly(params.n_sampled_points)
                    .GetPointPositions();

    RaycastingScene scene1, scene2;
    scene1.AddTriangles(cpu_mesh1);
    scene2.AddTriangles(cpu_mesh2);

    core::Tensor distance21 = scene1.ComputeDistance(points2);
    core::Tensor distance12 = scene2.ComputeDistance(points1);

    return ComputeMetricsCommon(distance12, distance21, metrics, params);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
