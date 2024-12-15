// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/PointCloud.h"

#include <libqhullcpp/PointCoordinates.h>
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullVertexSet.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/core/hashmap/HashSet.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/geometry/VtkUtils.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/Metrics.h"
#include "open3d/t/geometry/kernel/PCAPartition.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/Transform.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace t {
namespace geometry {

PointCloud::PointCloud(const core::Device &device)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      device_(device),
      point_attr_(TensorMap("positions")) {}

PointCloud::PointCloud(const core::Tensor &points)
    : PointCloud(points.GetDevice()) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    SetPointPositions(points);
}

PointCloud::PointCloud(const std::unordered_map<std::string, core::Tensor>
                               &map_keys_to_tensors)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      point_attr_(TensorMap("positions")) {
    if (map_keys_to_tensors.count("positions") == 0) {
        utility::LogError("\"positions\" attribute must be specified.");
    }
    device_ = map_keys_to_tensors.at("positions").GetDevice();
    core::AssertTensorShape(map_keys_to_tensors.at("positions"),
                            {utility::nullopt, 3});
    point_attr_ = TensorMap("positions", map_keys_to_tensors.begin(),
                            map_keys_to_tensors.end());
}

std::string PointCloud::ToString() const {
    size_t num_points = 0;
    std::string points_dtype_str = "";
    if (point_attr_.count(point_attr_.GetPrimaryKey())) {
        num_points = GetPointPositions().GetLength();
        points_dtype_str =
                fmt::format(" ({})", GetPointPositions().GetDtype().ToString());
    }
    auto str =
            fmt::format("PointCloud on {} [{} points{}].\nAttributes:",
                        GetDevice().ToString(), num_points, points_dtype_str);

    if ((point_attr_.size() - point_attr_.count(point_attr_.GetPrimaryKey())) ==
        0)
        return str + " None.";
    for (const auto &keyval : point_attr_) {
        if (keyval.first != "positions") {
            str += fmt::format(" {} (dtype = {}, shape = {}),", keyval.first,
                               keyval.second.GetDtype().ToString(),
                               keyval.second.GetShape().ToString());
        }
    }
    str[str.size() - 1] = '.';
    return str;
}

core::Tensor PointCloud::GetMinBound() const {
    return GetPointPositions().Min({0});
}

core::Tensor PointCloud::GetMaxBound() const {
    return GetPointPositions().Max({0});
}

core::Tensor PointCloud::GetCenter() const {
    return GetPointPositions().Mean({0});
}

PointCloud PointCloud::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    PointCloud pcd(device);
    for (auto &kv : point_attr_) {
        pcd.SetPointAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    return pcd;
}

PointCloud PointCloud::Clone() const { return To(GetDevice(), /*copy=*/true); }

PointCloud PointCloud::Append(const PointCloud &other) const {
    PointCloud pcd(GetDevice());

    int64_t length = GetPointPositions().GetLength();

    for (auto &kv : point_attr_) {
        if (other.HasPointAttr(kv.first)) {
            auto other_attr = other.GetPointAttr(kv.first);
            core::AssertTensorDtype(other_attr, kv.second.GetDtype());
            core::AssertTensorDevice(other_attr, kv.second.GetDevice());

            // Checking shape compatibility.
            auto other_attr_shape = other_attr.GetShape();
            auto attr_shape = kv.second.GetShape();
            int64_t combined_length = other_attr_shape[0] + attr_shape[0];
            other_attr_shape[0] = combined_length;
            attr_shape[0] = combined_length;
            if (other_attr_shape != attr_shape) {
                utility::LogError(
                        "Shape mismatch. Attribute {}, shape {}, is not "
                        "compatible with {}.",
                        kv.first, other_attr.GetShape(), kv.second.GetShape());
            }

            core::Tensor combined_attr =
                    core::Tensor::Empty(other_attr_shape, kv.second.GetDtype(),
                                        kv.second.GetDevice());

            combined_attr.SetItem(core::TensorKey::Slice(0, length, 1),
                                  kv.second);
            combined_attr.SetItem(
                    core::TensorKey::Slice(length, combined_length, 1),
                    other_attr);

            pcd.SetPointAttr(kv.first, combined_attr.Clone());
        } else {
            utility::LogError(
                    "The pointcloud is missing attribute {}. The pointcloud "
                    "being appended, must have all the attributes present in "
                    "the pointcloud it is being appended to.",
                    kv.first);
        }
    }
    return pcd;
}

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    core::AssertTensorShape(transformation, {4, 4});

    kernel::transform::TransformPoints(transformation, GetPointPositions());
    if (HasPointNormals()) {
        kernel::transform::TransformNormals(transformation, GetPointNormals());
    }

    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    core::AssertTensorShape(translation, {3});

    core::Tensor transform =
            translation.To(GetDevice(), GetPointPositions().GetDtype());

    if (!relative) {
        transform -= GetCenter();
    }
    GetPointPositions() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    core::AssertTensorShape(center, {3});

    const core::Tensor center_d =
            center.To(GetDevice(), GetPointPositions().GetDtype());

    GetPointPositions().Sub_(center_d).Mul_(scale).Add_(center_d);
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(center, {3});

    kernel::transform::RotatePoints(R, GetPointPositions(), center);

    if (HasPointNormals()) {
        kernel::transform::RotateNormals(R, GetPointNormals());
    }
    return *this;
}

PointCloud PointCloud::SelectByMask(const core::Tensor &boolean_mask,
                                    bool invert /* = false */) const {
    const int64_t length = GetPointPositions().GetLength();
    core::AssertTensorDtype(boolean_mask, core::Dtype::Bool);
    core::AssertTensorShape(boolean_mask, {length});
    core::AssertTensorDevice(boolean_mask, GetDevice());

    core::Tensor indices_local;
    if (invert) {
        indices_local = boolean_mask.LogicalNot();
    } else {
        indices_local = boolean_mask;
    }

    PointCloud pcd(GetDevice());
    for (auto &kv : GetPointAttr()) {
        if (HasPointAttr(kv.first)) {
            pcd.SetPointAttr(kv.first, kv.second.IndexGet({indices_local}));
        }
    }

    utility::LogDebug("Pointcloud down sampled from {} points to {} points.",
                      length, pcd.GetPointPositions().GetLength());
    return pcd;
}

PointCloud PointCloud::SelectByIndex(
        const core::Tensor &indices,
        bool invert /* = false */,
        bool remove_duplicates /* = false */) const {
    const int64_t length = GetPointPositions().GetLength();
    core::AssertTensorDtype(indices, core::Int64);
    core::AssertTensorDevice(indices, GetDevice());

    PointCloud pcd(GetDevice());

    if (!remove_duplicates && !invert) {
        core::TensorKey key = core::TensorKey::IndexTensor(indices);
        for (auto &kv : GetPointAttr()) {
            if (HasPointAttr(kv.first)) {
                pcd.SetPointAttr(kv.first, kv.second.GetItem(key));
            }
        }
        utility::LogDebug(
                "Pointcloud down sampled from {} points to {} points.", length,
                pcd.GetPointPositions().GetLength());
    } else {
        // The indices may have duplicate index value and will result in
        // identity point cloud attributes. We convert indices Tensor into mask
        // Tensor and call SelectByMask to avoid this situation.
        core::Tensor mask =
                core::Tensor::Zeros({length}, core::Bool, GetDevice());
        mask.SetItem(core::TensorKey::IndexTensor(indices),
                     core::Tensor::Init<bool>(true, GetDevice()));

        pcd = SelectByMask(mask, invert);
    }

    return pcd;
}

PointCloud PointCloud::VoxelDownSample(double voxel_size,
                                       const std::string &reduction) const {
    if (voxel_size <= 0) {
        utility::LogError("voxel_size must be positive.");
    }
    if (reduction != "mean") {
        utility::LogError("Reduction can only be 'mean' for VoxelDownSample.");
    }

    // Discretize voxels.
    core::Tensor voxeld = GetPointPositions() / voxel_size;
    core::Tensor voxeli = voxeld.Floor().To(core::Int64);

    // Map discrete voxels to indices.
    core::HashSet voxeli_hashset(voxeli.GetLength(), core::Int64, {3}, device_);

    // Index map: (0, original_points) -> (0, unique_points).
    core::Tensor index_map_point2voxel, masks;
    voxeli_hashset.Insert(voxeli, index_map_point2voxel, masks);

    // Insert and find are two different passes.
    // In the insertion pass, -1/false is returned for already existing
    // downsampled corresponding points.
    // In the find pass, actual indices are returned corresponding downsampled
    // points.
    voxeli_hashset.Find(voxeli, index_map_point2voxel, masks);
    index_map_point2voxel = index_map_point2voxel.To(core::Int64);

    int64_t num_points = voxeli.GetLength();
    int64_t num_voxels = voxeli_hashset.Size();

    // Count the number of points in each voxel.
    auto voxel_num_points =
            core::Tensor::Zeros({num_voxels}, core::Float32, device_);
    voxel_num_points.IndexAdd_(
            /*dim*/ 0, index_map_point2voxel,
            core::Tensor::Ones({num_points}, core::Float32, device_));

    // Create a new point cloud.
    PointCloud pcd_down(device_);
    for (auto &kv : point_attr_) {
        auto point_attr = kv.second;

        std::string attr_string = kv.first;
        auto attr_dtype = point_attr.GetDtype();

        // Use float to avoid unsupported tensor types.
        core::SizeVector attr_shape = point_attr.GetShape();
        attr_shape[0] = num_voxels;
        auto voxel_attr =
                core::Tensor::Zeros(attr_shape, core::Float32, device_);
        if (reduction == "mean") {
            voxel_attr.IndexAdd_(0, index_map_point2voxel,
                                 point_attr.To(core::Float32));
            voxel_attr /= voxel_num_points.View({-1, 1});
            voxel_attr = voxel_attr.To(attr_dtype);
        } else {
            utility::LogError("Unsupported reduction type {}.", reduction);
        }
        pcd_down.SetPointAttr(attr_string, voxel_attr);
    }

    return pcd_down;
}

PointCloud PointCloud::UniformDownSample(size_t every_k_points) const {
    if (every_k_points == 0) {
        utility::LogError(
                "Illegal sample rate, every_k_points must be larger than 0.");
    }

    const int64_t length = GetPointPositions().GetLength();

    PointCloud pcd_down(GetDevice());
    for (auto &kv : GetPointAttr()) {
        pcd_down.SetPointAttr(
                kv.first,
                kv.second.Slice(0, 0, length, (int64_t)every_k_points));
    }

    return pcd_down;
}

PointCloud PointCloud::RandomDownSample(double sampling_ratio) const {
    if (sampling_ratio < 0 || sampling_ratio > 1) {
        utility::LogError(
                "Illegal sampling_ratio {}, sampling_ratio must be between 0 "
                "and 1.");
    }

    const int64_t length = GetPointPositions().GetLength();
    std::vector<int64_t> indices(length);
    std::iota(std::begin(indices), std::end(indices), 0);
    {
        std::lock_guard<std::mutex> lock(*utility::random::GetMutex());
        std::shuffle(indices.begin(), indices.end(),
                     *utility::random::GetEngine());
    }

    const int sample_size = sampling_ratio * length;
    indices.resize(sample_size);
    // TODO: Generate random indices in GPU using CUDA rng maybe more efficient
    // than copy indices data from CPU to GPU.
    return SelectByIndex(
            core::Tensor(indices, {sample_size}, core::Int64, GetDevice()),
            false, false);
}

PointCloud PointCloud::FarthestPointDownSample(const size_t num_samples,
                                               const size_t start_index) const {
    const core::Dtype dtype = GetPointPositions().GetDtype();
    const int64_t num_points = GetPointPositions().GetLength();
    if (num_samples == 0) {
        return PointCloud(GetDevice());
    } else if (num_samples == size_t(num_points)) {
        return Clone();
    } else if (num_samples > size_t(num_points)) {
        utility::LogError(
                "Illegal number of samples: {}, must <= point size: {}",
                num_samples, num_points);
    } else if (start_index >= size_t(num_points)) {
        utility::LogError("Illegal start index: {}, must <= point size: {}",
                          start_index, num_points);
    }
    core::Tensor selection_mask =
            core::Tensor::Zeros({num_points}, core::Bool, GetDevice());
    core::Tensor smallest_distances = core::Tensor::Full(
            {num_points}, std::numeric_limits<double>::infinity(), dtype,
            GetDevice());

    int64_t farthest_index = static_cast<int64_t>(start_index);

    for (size_t i = 0; i < num_samples; i++) {
        selection_mask[farthest_index] = true;
        core::Tensor selected = GetPointPositions()[farthest_index];

        core::Tensor diff = GetPointPositions() - selected;
        core::Tensor distances_to_selected = (diff * diff).Sum({1});
        smallest_distances = open3d::core::Minimum(distances_to_selected,
                                                   smallest_distances);

        farthest_index = smallest_distances.ArgMax({0}).Item<int64_t>();
    }
    return SelectByMask(selection_mask);
}

std::tuple<PointCloud, core::Tensor> PointCloud::RemoveRadiusOutliers(
        size_t nb_points, double search_radius) const {
    if (nb_points < 1 || search_radius <= 0) {
        utility::LogError(
                "Illegal input parameters, number of points and radius must be "
                "positive");
    }
    core::nns::NearestNeighborSearch target_nns(GetPointPositions());

    const bool check = target_nns.FixedRadiusIndex(search_radius);
    if (!check) {
        utility::LogError("Fixed radius search index is not set.");
    }

    core::Tensor indices, distance, row_splits;
    std::tie(indices, distance, row_splits) = target_nns.FixedRadiusSearch(
            GetPointPositions(), search_radius, false);
    row_splits = row_splits.To(GetDevice());

    const int64_t size = row_splits.GetLength();
    const core::Tensor num_neighbors =
            row_splits.Slice(0, 1, size) - row_splits.Slice(0, 0, size - 1);

    const core::Tensor valid =
            num_neighbors.Ge(static_cast<int64_t>(nb_points));
    return std::make_tuple(SelectByMask(valid), valid);
}

std::tuple<PointCloud, core::Tensor> PointCloud::RemoveStatisticalOutliers(
        size_t nb_neighbors, double std_ratio) const {
    if (nb_neighbors < 1 || std_ratio <= 0) {
        utility::LogError(
                "Illegal input parameters, the number of neighbors and "
                "standard deviation ratio must be positive.");
    }
    if (GetPointPositions().GetLength() == 0) {
        return std::make_tuple(PointCloud(GetDevice()),
                               core::Tensor({0}, core::Bool, GetDevice()));
    }

    core::nns::NearestNeighborSearch nns(GetPointPositions().Contiguous());
    const bool check = nns.KnnIndex();
    if (!check) {
        utility::LogError("Knn search index is not set.");
    }

    core::Tensor indices, distance2;
    std::tie(indices, distance2) =
            nns.KnnSearch(GetPointPositions(), nb_neighbors);

    core::Tensor avg_distances = distance2.Sqrt().Mean({1});
    const double cloud_mean =
            avg_distances.Mean({0}).To(core::Float64).Item<double>();
    const core::Tensor std_distances_centered = avg_distances - cloud_mean;
    const double sq_sum = (std_distances_centered * std_distances_centered)
                                  .Sum({0})
                                  .To(core::Float64)
                                  .Item<double>();
    const double std_dev =
            std::sqrt(sq_sum / (avg_distances.GetShape()[0] - 1));
    const double distance_threshold = cloud_mean + std_ratio * std_dev;
    const core::Tensor valid = avg_distances.Le(distance_threshold);

    return std::make_tuple(SelectByMask(valid), valid);
}

std::tuple<PointCloud, core::Tensor> PointCloud::RemoveNonFinitePoints(
        bool remove_nan, bool remove_inf) const {
    core::Tensor finite_indices_mask;
    const core::SizeVector dim = {1};
    if (remove_nan && remove_inf) {
        finite_indices_mask =
                this->GetPointPositions().IsFinite().All(dim, false);
    } else if (remove_nan) {
        finite_indices_mask =
                this->GetPointPositions().IsNan().LogicalNot().All(dim, false);
    } else if (remove_inf) {
        finite_indices_mask =
                this->GetPointPositions().IsInf().LogicalNot().All(dim, false);
    } else {
        finite_indices_mask = core::Tensor::Full(
                {this->GetPointPositions().GetLength()}, true, core::Bool,
                this->GetPointPositions().GetDevice());
    }

    utility::LogDebug("Removing non-finite points.");
    return std::make_tuple(SelectByMask(finite_indices_mask),
                           finite_indices_mask);
}

std::tuple<PointCloud, core::Tensor> PointCloud::RemoveDuplicatedPoints()
        const {
    core::Tensor points_voxeli;
    const core::Dtype dtype = GetPointPositions().GetDtype();
    if (dtype.ByteSize() == 4) {
        points_voxeli = GetPointPositions().ReinterpretCast(core::Int32);
    } else if (dtype.ByteSize() == 8) {
        points_voxeli = GetPointPositions().ReinterpretCast(core::Int64);
    } else {
        utility::LogError(
                "Unsupported point position data-type. Only support "
                "Int32, Int64, Float32 and Float64.");
    }

    core::HashSet points_voxeli_hashset(points_voxeli.GetLength(),
                                        points_voxeli.GetDtype(), {3}, device_);
    core::Tensor buf_indices, masks;
    points_voxeli_hashset.Insert(points_voxeli, buf_indices, masks);

    return std::make_tuple(SelectByMask(masks), masks);
}

PointCloud &PointCloud::NormalizeNormals() {
    if (!HasPointNormals()) {
        utility::LogWarning("PointCloud has no normals.");
        return *this;
    } else {
        SetPointNormals(GetPointNormals().Contiguous());
    }

    core::Tensor &normals = GetPointNormals();
    if (IsCPU()) {
        kernel::pointcloud::NormalizeNormalsCPU(normals);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::NormalizeNormalsCUDA, normals);
    } else {
        utility::LogError("Unimplemented device");
    }

    return *this;
}

PointCloud &PointCloud::PaintUniformColor(const core::Tensor &color) {
    core::AssertTensorShape(color, {3});
    core::Tensor clipped_color = color.To(GetDevice());
    if (color.GetDtype() == core::Float32 ||
        color.GetDtype() == core::Float64) {
        clipped_color = clipped_color.Clip(0.0f, 1.0f);
    }
    core::Tensor pcd_colors =
            core::Tensor::Empty({GetPointPositions().GetLength(), 3},
                                clipped_color.GetDtype(), GetDevice());
    pcd_colors.AsRvalue() = clipped_color;
    SetPointColors(pcd_colors);

    return *this;
}

std::tuple<PointCloud, core::Tensor> PointCloud::ComputeBoundaryPoints(
        double radius, int max_nn, double angle_threshold) const {
    core::AssertTensorDtypes(this->GetPointPositions(),
                             {core::Float32, core::Float64});
    if (!HasPointNormals()) {
        utility::LogError(
                "PointCloud must have normals attribute to compute boundary "
                "points.");
    }

    const core::Device device = GetDevice();
    const int64_t num_points = GetPointPositions().GetLength();

    const core::Tensor points_d = GetPointPositions().Contiguous();
    const core::Tensor normals_d = GetPointNormals().Contiguous();

    // Compute nearest neighbors.
    core::Tensor indices, distance2, counts;
    core::nns::NearestNeighborSearch tree(points_d, core::Int32);

    bool check = tree.HybridIndex(radius);
    if (!check) {
        utility::LogError("Building HybridIndex failed.");
    }
    std::tie(indices, distance2, counts) =
            tree.HybridSearch(points_d, radius, max_nn);
    utility::LogDebug(
            "Use HybridSearch [max_nn: {} | radius {}] for computing "
            "boundary points.",
            max_nn, radius);

    core::Tensor mask = core::Tensor::Zeros({num_points}, core::Bool, device);
    if (IsCPU()) {
        kernel::pointcloud::ComputeBoundaryPointsCPU(
                points_d, normals_d, indices, counts, mask, angle_threshold);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::ComputeBoundaryPointsCUDA, points_d,
                  normals_d, indices, counts, mask, angle_threshold);
    } else {
        utility::LogError("Unimplemented device");
    }

    return std::make_tuple(SelectByMask(mask), mask);
}

void PointCloud::EstimateNormals(
        const utility::optional<int> max_knn /* = 30*/,
        const utility::optional<double> radius /*= utility::nullopt*/) {
    core::AssertTensorDtypes(this->GetPointPositions(),
                             {core::Float32, core::Float64});

    const core::Dtype dtype = this->GetPointPositions().GetDtype();
    const core::Device device = GetDevice();

    const bool has_normals = HasPointNormals();

    if (!has_normals) {
        this->SetPointNormals(core::Tensor::Empty(
                {GetPointPositions().GetLength(), 3}, dtype, device));
    } else {
        core::AssertTensorDtype(this->GetPointNormals(), dtype);

        this->SetPointNormals(GetPointNormals().Contiguous());
    }

    this->SetPointAttr(
            "covariances",
            core::Tensor::Empty({GetPointPositions().GetLength(), 3, 3}, dtype,
                                device));

    if (radius.has_value() && max_knn.has_value()) {
        utility::LogDebug("Using Hybrid Search for computing covariances");
        // Computes and sets `covariances` attribute using Hybrid Search
        // method.
        if (IsCPU()) {
            kernel::pointcloud::EstimateCovariancesUsingHybridSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointAttr("covariances"), radius.value(),
                    max_knn.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateCovariancesUsingHybridSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointAttr("covariances"), radius.value(),
                      max_knn.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else if (max_knn.has_value() && !radius.has_value()) {
        utility::LogDebug("Using KNN Search for computing covariances");
        // Computes and sets `covariances` attribute using KNN Search method.
        if (IsCPU()) {
            kernel::pointcloud::EstimateCovariancesUsingKNNSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointAttr("covariances"), max_knn.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::EstimateCovariancesUsingKNNSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointAttr("covariances"), max_knn.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else if (!max_knn.has_value() && radius.has_value()) {
        utility::LogDebug("Using Radius Search for computing covariances");
        // Computes and sets `covariances` attribute using KNN Search method.
        if (IsCPU()) {
            kernel::pointcloud::EstimateCovariancesUsingRadiusSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointAttr("covariances"), radius.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateCovariancesUsingRadiusSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointAttr("covariances"), radius.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogError("Both max_nn and radius are none.");
    }

    // Estimate `normal` of each point using its `covariance` matrix.
    if (IsCPU()) {
        kernel::pointcloud::EstimateNormalsFromCovariancesCPU(
                this->GetPointAttr("covariances"), this->GetPointNormals(),
                has_normals);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::EstimateNormalsFromCovariancesCUDA,
                  this->GetPointAttr("covariances"), this->GetPointNormals(),
                  has_normals);
    } else {
        utility::LogError("Unimplemented device");
    }

    // TODO (@rishabh): Don't remove covariances attribute, when
    // EstimateCovariance functionality is exposed.
    RemovePointAttr("covariances");
}

void PointCloud::OrientNormalsToAlignWithDirection(
        const core::Tensor &orientation_reference) {
    core::AssertTensorDevice(orientation_reference, GetDevice());
    core::AssertTensorShape(orientation_reference, {3});

    if (!HasPointNormals()) {
        utility::LogError(
                "No normals in the PointCloud. Call EstimateNormals() first.");
    } else {
        SetPointNormals(GetPointNormals().Contiguous());
    }

    core::Tensor reference =
            orientation_reference.To(GetPointPositions().GetDtype());

    core::Tensor &normals = GetPointNormals();
    if (IsCPU()) {
        kernel::pointcloud::OrientNormalsToAlignWithDirectionCPU(normals,
                                                                 reference);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::OrientNormalsToAlignWithDirectionCUDA,
                  normals, reference);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void PointCloud::OrientNormalsTowardsCameraLocation(
        const core::Tensor &camera_location) {
    core::AssertTensorDevice(camera_location, GetDevice());
    core::AssertTensorShape(camera_location, {3});

    if (!HasPointNormals()) {
        utility::LogError(
                "No normals in the PointCloud. Call EstimateNormals() first.");
    } else {
        SetPointNormals(GetPointNormals().Contiguous());
    }

    core::Tensor reference = camera_location.To(GetPointPositions().GetDtype());

    core::Tensor &normals = GetPointNormals();
    if (IsCPU()) {
        kernel::pointcloud::OrientNormalsTowardsCameraLocationCPU(
                GetPointPositions().Contiguous(), normals, reference);
    } else if (IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::OrientNormalsTowardsCameraLocationCUDA,
                  GetPointPositions().Contiguous(), normals, reference);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void PointCloud::OrientNormalsConsistentTangentPlane(
        size_t k,
        const double lambda /* = 0.0*/,
        const double cos_alpha_tol /* = 1.0*/) {
    PointCloud tpcd(GetPointPositions());
    tpcd.SetPointNormals(GetPointNormals());

    open3d::geometry::PointCloud lpcd = tpcd.ToLegacy();
    lpcd.OrientNormalsConsistentTangentPlane(k, lambda, cos_alpha_tol);

    SetPointNormals(core::eigen_converter::EigenVector3dVectorToTensor(
            lpcd.normals_, GetPointPositions().GetDtype(), GetDevice()));
}

void PointCloud::EstimateColorGradients(
        const utility::optional<int> max_knn /* = 30*/,
        const utility::optional<double> radius /*= utility::nullopt*/) {
    if (!HasPointColors() || !HasPointNormals()) {
        utility::LogError(
                "PointCloud must have colors and normals attribute "
                "to compute color gradients.");
    }
    core::AssertTensorDtypes(this->GetPointColors(),
                             {core::Float32, core::Float64});

    const core::Dtype dtype = this->GetPointColors().GetDtype();
    const core::Device device = GetDevice();

    if (!this->HasPointAttr("color_gradients")) {
        this->SetPointAttr(
                "color_gradients",
                core::Tensor::Empty({GetPointPositions().GetLength(), 3}, dtype,
                                    device));
    } else {
        if (this->GetPointAttr("color_gradients").GetDtype() != dtype) {
            utility::LogError(
                    "color_gradients attribute must be of same dtype as "
                    "points.");
        }
        // If color_gradients attribute is already present, do not re-compute.
        return;
    }

    // Compute and set `color_gradients` attribute.
    if (radius.has_value() && max_knn.has_value()) {
        utility::LogDebug("Using Hybrid Search for computing color_gradients");
        if (IsCPU()) {
            kernel::pointcloud::EstimateColorGradientsUsingHybridSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointNormals().Contiguous(),
                    this->GetPointColors().Contiguous(),
                    this->GetPointAttr("color_gradients"), radius.value(),
                    max_knn.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateColorGradientsUsingHybridSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointNormals().Contiguous(),
                      this->GetPointColors().Contiguous(),
                      this->GetPointAttr("color_gradients"), radius.value(),
                      max_knn.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else if (max_knn.has_value() && !radius.has_value()) {
        utility::LogDebug("Using KNN Search for computing color_gradients");
        if (IsCPU()) {
            kernel::pointcloud::EstimateColorGradientsUsingKNNSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointNormals().Contiguous(),
                    this->GetPointColors().Contiguous(),
                    this->GetPointAttr("color_gradients"), max_knn.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateColorGradientsUsingKNNSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointNormals().Contiguous(),
                      this->GetPointColors().Contiguous(),
                      this->GetPointAttr("color_gradients"), max_knn.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else if (!max_knn.has_value() && radius.has_value()) {
        utility::LogDebug("Using Radius Search for computing color_gradients");
        if (IsCPU()) {
            kernel::pointcloud::EstimateColorGradientsUsingRadiusSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointNormals().Contiguous(),
                    this->GetPointColors().Contiguous(),
                    this->GetPointAttr("color_gradients"), radius.value());
        } else if (IsCUDA()) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateColorGradientsUsingRadiusSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointNormals().Contiguous(),
                      this->GetPointColors().Contiguous(),
                      this->GetPointAttr("color_gradients"), radius.value());
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogError("Both max_nn and radius are none.");
    }
}

static PointCloud CreatePointCloudWithNormals(
        const Image &depth_in, /* UInt16 or Float32 */
        const Image &color_in, /* Float32 */
        const core::Tensor &intrinsics_in,
        const core::Tensor &extrinsics,
        float depth_scale,
        float depth_max,
        int stride) {
    using core::None;
    using core::Tensor;
    using core::TensorKey;
    const float invalid_fill = NAN;
    // Filter defaults for depth processing
    const int bilateral_kernel_size =  // bilateral filter defaults for backends
            depth_in.IsCUDA() ? 3 : 5;
    const float depth_diff_threshold = 0.14f;
    const float bilateral_value_sigma = 10.f;
    const float bilateral_distance_sigma = 10.f;
    if ((stride & (stride - 1)) != 0) {
        utility::LogError(
                "Only powers of 2 are supported for stride when "
                "with_normals=true.");
    }
    if (color_in.GetRows() > 0 && (depth_in.GetRows() != color_in.GetRows() ||
                                   depth_in.GetCols() != color_in.GetCols())) {
        utility::LogError("Depth and color images have different sizes.");
    }
    auto depth =
            depth_in.ClipTransform(depth_scale, 0.01f, depth_max, invalid_fill);
    auto color = color_in;
    auto intrinsics = intrinsics_in / stride;
    intrinsics[-1][-1] = 1.f;
    if (stride == 1) {
        depth = depth.FilterBilateral(bilateral_kernel_size,
                                      bilateral_value_sigma,
                                      bilateral_distance_sigma);
    } else {
        for (; stride > 1; stride /= 2) {
            depth = depth.PyrDownDepth(depth_diff_threshold, invalid_fill);
            color = color.PyrDown();
        }
    }
    const int64_t im_size = depth.GetRows() * depth.GetCols();
    auto vertex_map = depth.CreateVertexMap(intrinsics, invalid_fill);
    auto vertex_list = vertex_map.AsTensor().View({im_size, 3});
    if (!extrinsics.AllClose(Tensor::Eye(4, extrinsics.GetDtype(),
                                         extrinsics.GetDevice()))) {
        auto cam_to_world = extrinsics.Inverse();
        vertex_list = vertex_list
                              .Matmul(cam_to_world.Slice(0, 0, 3, 1)
                                              .Slice(1, 0, 3, 1)
                                              .T())
                              .Add_(cam_to_world.Slice(0, 0, 3, 1)
                                            .Slice(1, 3, 4, 1)
                                            .T());
        vertex_map = Image(vertex_list.View(vertex_map.AsTensor().GetShape())
                                   .Contiguous());
    }
    auto normal_map_t = vertex_map.CreateNormalMap(invalid_fill)
                                .AsTensor()
                                .View({im_size, 3});
    // all columns are the same
    auto valid_idx =
            normal_map_t.Slice(1, 0, 1, 1)
                    .IsFinite()
                    .LogicalAnd(vertex_list.Slice(1, 0, 1, 1).IsFinite())
                    .Reshape({im_size});
    PointCloud pcd(
            {{"positions",
              vertex_list.GetItem({TensorKey::IndexTensor(valid_idx),
                                   TensorKey::Slice(None, None, None)})},
             {"normals",
              normal_map_t.GetItem({TensorKey::IndexTensor(valid_idx),
                                    TensorKey::Slice(None, None, None)})}});
    if (color.GetRows() > 0) {
        pcd.SetPointColors(
                color.AsTensor()
                        .View({im_size, 3})
                        .GetItem({TensorKey::IndexTensor(valid_idx),
                                  TensorKey::Slice(None, None, None)}));
    }
    return pcd;
}

PointCloud PointCloud::CreateFromDepthImage(const Image &depth,
                                            const core::Tensor &intrinsics,
                                            const core::Tensor &extrinsics,
                                            float depth_scale,
                                            float depth_max,
                                            int stride,
                                            bool with_normals) {
    core::AssertTensorDtypes(depth.AsTensor(), {core::UInt16, core::Float32});
    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    core::Tensor intrinsics_host =
            intrinsics.To(core::Device("CPU:0"), core::Float64);
    core::Tensor extrinsics_host =
            extrinsics.To(core::Device("CPU:0"), core::Float64);

    if (with_normals) {
        return CreatePointCloudWithNormals(depth, Image(), intrinsics_host,
                                           extrinsics_host, depth_scale,
                                           depth_max, stride);
    } else {
        core::Tensor points;
        kernel::pointcloud::Unproject(depth.AsTensor(), utility::nullopt,
                                      points, utility::nullopt, intrinsics_host,
                                      extrinsics_host, depth_scale, depth_max,
                                      stride);
        return PointCloud(points);
    }
}

PointCloud PointCloud::CreateFromRGBDImage(const RGBDImage &rgbd_image,
                                           const core::Tensor &intrinsics,
                                           const core::Tensor &extrinsics,
                                           float depth_scale,
                                           float depth_max,
                                           int stride,
                                           bool with_normals) {
    core::AssertTensorDtypes(rgbd_image.depth_.AsTensor(),
                             {core::UInt16, core::Float32});
    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    Image image_colors = rgbd_image.color_.To(core::Float32, /*copy=*/false);

    if (with_normals) {
        return CreatePointCloudWithNormals(rgbd_image.depth_, image_colors,
                                           intrinsics, extrinsics, depth_scale,
                                           depth_max, stride);
    } else {
        core::Tensor points, colors, image_colors_t = image_colors.AsTensor();
        kernel::pointcloud::Unproject(
                rgbd_image.depth_.AsTensor(), image_colors_t, points, colors,
                intrinsics, extrinsics, depth_scale, depth_max, stride);
        return PointCloud({{"positions", points}, {"colors", colors}});
    }
}

geometry::Image PointCloud::ProjectToDepthImage(int width,
                                                int height,
                                                const core::Tensor &intrinsics,
                                                const core::Tensor &extrinsics,
                                                float depth_scale,
                                                float depth_max) {
    if (!HasPointPositions()) {
        utility::LogWarning(
                "Called ProjectToDepthImage on a point cloud with no Positions "
                "attribute. Returning empty image.");
        return geometry::Image();
    }
    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    core::Tensor depth =
            core::Tensor::Zeros({height, width, 1}, core::Float32, device_);
    kernel::pointcloud::Project(depth, utility::nullopt, GetPointPositions(),
                                utility::nullopt, intrinsics, extrinsics,
                                depth_scale, depth_max);
    return geometry::Image(depth);
}

geometry::RGBDImage PointCloud::ProjectToRGBDImage(
        int width,
        int height,
        const core::Tensor &intrinsics,
        const core::Tensor &extrinsics,
        float depth_scale,
        float depth_max) {
    if (!HasPointPositions()) {
        utility::LogWarning(
                "Called ProjectToRGBDImage on a point cloud with no Positions "
                "attribute. Returning empty image.");
        return geometry::RGBDImage();
    }
    if (!HasPointColors()) {
        utility::LogError(
                "Unable to project to RGBD without the Color attribute in the "
                "point cloud.");
    }
    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    core::Tensor depth =
            core::Tensor::Zeros({height, width, 1}, core::Float32, device_);
    core::Tensor color =
            core::Tensor::Zeros({height, width, 3}, core::Float32, device_);

    // Assume point colors are Float32 for kernel dispatch
    core::Tensor point_colors = GetPointColors();
    if (point_colors.GetDtype() == core::Dtype::UInt8) {
        point_colors = point_colors.To(core::Dtype::Float32) / 255.0;
    }

    kernel::pointcloud::Project(depth, color, GetPointPositions(), point_colors,
                                intrinsics, extrinsics, depth_scale, depth_max);

    return geometry::RGBDImage(color, depth);
}

PointCloud PointCloud::FromLegacy(
        const open3d::geometry::PointCloud &pcd_legacy,
        core::Dtype dtype,
        const core::Device &device) {
    geometry::PointCloud pcd(device);
    if (pcd_legacy.HasPoints()) {
        pcd.SetPointPositions(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        pcd_legacy.points_, dtype, device));
    } else {
        utility::LogWarning("Creating from an empty legacy PointCloud.");
    }
    if (pcd_legacy.HasColors()) {
        pcd.SetPointColors(core::eigen_converter::EigenVector3dVectorToTensor(
                pcd_legacy.colors_, dtype, device));
    }
    if (pcd_legacy.HasNormals()) {
        pcd.SetPointNormals(core::eigen_converter::EigenVector3dVectorToTensor(
                pcd_legacy.normals_, dtype, device));
    }
    return pcd;
}

open3d::geometry::PointCloud PointCloud::ToLegacy() const {
    open3d::geometry::PointCloud pcd_legacy;
    if (HasPointPositions()) {
        pcd_legacy.points_ = core::eigen_converter::TensorToEigenVector3dVector(
                GetPointPositions());
    }
    if (HasPointColors()) {
        bool dtype_is_supported_for_conversion = true;
        double normalization_factor = 1.0;
        core::Dtype point_color_dtype = GetPointColors().GetDtype();

        if (point_color_dtype == core::UInt8) {
            normalization_factor =
                    1.0 /
                    static_cast<double>(std::numeric_limits<uint8_t>::max());
        } else if (point_color_dtype == core::UInt16) {
            normalization_factor =
                    1.0 /
                    static_cast<double>(std::numeric_limits<uint16_t>::max());
        } else if (point_color_dtype != core::Float32 &&
                   point_color_dtype != core::Float64) {
            utility::LogWarning(
                    "Dtype {} of color attribute is not supported for "
                    "conversion to LegacyPointCloud and will be skipped. "
                    "Supported dtypes include UInt8, UIn16, Float32, and "
                    "Float64",
                    point_color_dtype.ToString());
            dtype_is_supported_for_conversion = false;
        }

        if (dtype_is_supported_for_conversion) {
            if (normalization_factor != 1.0) {
                core::Tensor rescaled_colors =
                        GetPointColors().To(core::Float64) *
                        normalization_factor;
                pcd_legacy.colors_ =
                        core::eigen_converter::TensorToEigenVector3dVector(
                                rescaled_colors);
            } else {
                pcd_legacy.colors_ =
                        core::eigen_converter::TensorToEigenVector3dVector(
                                GetPointColors());
            }
        }
    }
    if (HasPointNormals()) {
        pcd_legacy.normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        GetPointNormals());
    }
    return pcd_legacy;
}

std::tuple<TriangleMesh, core::Tensor> PointCloud::HiddenPointRemoval(
        const core::Tensor &camera_location, double radius) const {
    core::AssertTensorShape(camera_location, {3});
    core::AssertTensorDevice(camera_location, GetDevice());

    // The HiddenPointRemoval only need positions attribute.
    PointCloud tpcd(GetPointPositions());
    const open3d::geometry::PointCloud lpcd = tpcd.ToLegacy();
    const Eigen::Vector3d camera_location_eigen =
            core::eigen_converter::TensorToEigenMatrixXd(
                    camera_location.Reshape({3, 1}));

    std::shared_ptr<open3d::geometry::TriangleMesh> lmesh;
    std::vector<size_t> pt_map;
    std::tie(lmesh, pt_map) =
            lpcd.HiddenPointRemoval(camera_location_eigen, radius);

    // Convert pt_map into Int64 Tensor.
    std::vector<int64_t> indices(pt_map.begin(), pt_map.end());

    return std::make_tuple(
            TriangleMesh::FromLegacy(*lmesh, GetPointPositions().GetDtype(),
                                     core::Int64, GetDevice()),
            core::Tensor(std::move(indices)).To(GetDevice()));
}

core::Tensor PointCloud::ClusterDBSCAN(double eps,
                                       size_t min_points,
                                       bool print_progress) const {
    // Create a legacy point cloud with only points, no attributes to reduce
    // copying.
    PointCloud tpcd(GetPointPositions());
    open3d::geometry::PointCloud lpcd = tpcd.ToLegacy();
    std::vector<int> labels =
            lpcd.ClusterDBSCAN(eps, min_points, print_progress);
    return core::Tensor(std::move(labels)).To(GetDevice());
}

std::tuple<core::Tensor, core::Tensor> PointCloud::SegmentPlane(
        const double distance_threshold,
        const int ransac_n,
        const int num_iterations,
        const double probability) const {
    // The RANSAC plane fitting only need positions attribute.
    PointCloud tpcd(GetPointPositions());
    const open3d::geometry::PointCloud lpcd = tpcd.ToLegacy();
    std::vector<size_t> inliers;
    Eigen::Vector4d plane;
    std::tie(plane, inliers) = lpcd.SegmentPlane(distance_threshold, ransac_n,
                                                 num_iterations, probability);

    // Convert inliers into Int64 Tensor.
    std::vector<int64_t> indices(inliers.begin(), inliers.end());

    return std::make_tuple(
            core::eigen_converter::EigenMatrixToTensor(plane).Flatten().To(
                    GetDevice()),
            core::Tensor(std::move(indices)).To(GetDevice()));
}

TriangleMesh PointCloud::ComputeConvexHull(bool joggle_inputs) const {
    // QHull needs double dtype on the CPU.
    static_assert(std::is_same<realT, double>::value,
                  "Qhull realT is not double. Update code!");
    using int_t = int32_t;
    const auto int_dtype = core::Int32;
    core::Tensor coordinates(
            GetPointPositions().To(core::Float64).To(core::Device("CPU:0")));

    orgQhull::Qhull qhull;
    std::string options = "Qt";  // triangulated output
    if (joggle_inputs) {
        options = "QJ";  // joggle input to avoid precision problems
    }
    qhull.runQhull("", 3, coordinates.GetLength(),
                   coordinates.GetDataPtr<double>(), options.c_str());
    orgQhull::QhullFacetList facets = qhull.facetList();

    core::Tensor vertices({qhull.vertexCount(), 3}, core::Float64),
            triangles({qhull.facetCount(), 3}, int_dtype),
            point_indices({qhull.vertexCount()}, int_dtype);
    std::unordered_map<int_t, int_t> vertex_map;  // pcd -> conv hull
    int_t tidx = 0, next_vtx = 0;
    auto p_vertices = vertices.GetDataPtr<double>();
    auto p_triangle = triangles.GetDataPtr<int_t>();
    auto p_point_indices = point_indices.GetDataPtr<int_t>();
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullVertexSet vSet = it->vertices();
        int_t triangle_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt, ++triangle_subscript) {
            orgQhull::QhullPoint p = (*vIt).point();
            int_t vidx = p.id();

            auto inserted = vertex_map.insert({vidx, next_vtx});
            if (inserted.second) {
                p_triangle[triangle_subscript] = next_vtx;  // hull vertex idx
                double *coords = p.coordinates();
                std::copy(coords, coords + 3, p_vertices);
                p_vertices += 3;
                p_point_indices[next_vtx++] = vidx;
            } else {
                p_triangle[triangle_subscript] =
                        inserted.first->second;  // hull vertex idx
            }
        }
        if ((*it).isTopOrient()) {
            std::swap(p_triangle[0], p_triangle[1]);
        }
        tidx++;
        p_triangle += 3;
    }
    if (tidx < triangles.GetShape(0)) {
        triangles = triangles.Slice(0, 0, tidx);
    }
    if (next_vtx != vertices.GetShape(0)) {
        utility::LogError(
                "Qhull output has incorrect number of vertices {} instead of "
                "reported {}",
                next_vtx, vertices.GetShape(0));
    }

    TriangleMesh convex_hull(vertices, triangles);
    convex_hull.SetVertexAttr("point_indices", point_indices);
    return convex_hull.To(GetPointPositions().GetDevice());
}

AxisAlignedBoundingBox PointCloud::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetPointPositions());
}

OrientedBoundingBox PointCloud::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(GetPointPositions());
}

LineSet PointCloud::ExtrudeRotation(double angle,
                                    const core::Tensor &axis,
                                    int resolution,
                                    double translation,
                                    bool capping) const {
    using namespace vtkutils;
    return ExtrudeRotationLineSet(*this, angle, axis, resolution, translation,
                                  capping);
}

LineSet PointCloud::ExtrudeLinear(const core::Tensor &vector,
                                  double scale,
                                  bool capping) const {
    using namespace vtkutils;
    return ExtrudeLinearLineSet(*this, vector, scale, capping);
}

PointCloud PointCloud::Crop(const AxisAlignedBoundingBox &aabb,
                            bool invert) const {
    core::AssertTensorDevice(GetPointPositions(), aabb.GetDevice());
    if (aabb.IsEmpty()) {
        utility::LogWarning(
                "Bounding box is empty. Returning empty point cloud if "
                "invert is false, or the original point cloud if "
                "invert is true.");
        return invert ? Clone() : PointCloud(GetDevice());
    }
    return SelectByIndex(
            aabb.GetPointIndicesWithinBoundingBox(GetPointPositions()), invert);
}

PointCloud PointCloud::Crop(const OrientedBoundingBox &obb, bool invert) const {
    core::AssertTensorDevice(GetPointPositions(), obb.GetDevice());
    if (obb.IsEmpty()) {
        utility::LogWarning(
                "Bounding box is empty. Returning empty point cloud if "
                "invert is false, or the original point cloud if "
                "invert is true.");
        return invert ? Clone() : PointCloud(GetDevice());
    }
    return SelectByIndex(
            obb.GetPointIndicesWithinBoundingBox(GetPointPositions()), invert);
}

int PointCloud::PCAPartition(int max_points) {
    int num_partitions;
    core::Tensor partition_ids;
    std::tie(num_partitions, partition_ids) =
            kernel::pcapartition::PCAPartition(GetPointPositions(), max_points);
    SetPointAttr("partition_ids", partition_ids.To(GetDevice()));
    return num_partitions;
}

core::Tensor PointCloud::ComputeMetrics(const PointCloud &pcd2,
                                        std::vector<Metric> metrics,
                                        MetricParameters params) const {
    if (IsEmpty() || pcd2.IsEmpty()) {
        utility::LogError("One or both input point clouds are empty!");
    }
    if (!IsCPU() || !pcd2.IsCPU()) {
        utility::LogWarning(
                "ComputeDistance is implemented only on CPU. Computing on "
                "CPU.");
    }
    core::Tensor points1 = GetPointPositions().To(core::Device("CPU:0")),
                 points2 = pcd2.GetPointPositions().To(core::Device("CPU:0"));
    [[maybe_unused]] core::Tensor indices12, indices21;
    core::Tensor sqr_distance12, sqr_distance21;

    core::nns::NearestNeighborSearch tree1(points1);
    core::nns::NearestNeighborSearch tree2(points2);

    if (!tree2.KnnIndex()) {
        utility::LogError("[ComputeDistance] Building KNN-Index failed!");
    }
    if (!tree1.KnnIndex()) {
        utility::LogError("[ComputeDistance] Building KNN-Index failed!");
    }

    std::tie(indices12, sqr_distance12) = tree2.KnnSearch(points1, 1);
    std::tie(indices21, sqr_distance21) = tree1.KnnSearch(points2, 1);

    return ComputeMetricsCommon(sqr_distance12.Sqrt_(), sqr_distance21.Sqrt_(),
                                metrics, params);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
