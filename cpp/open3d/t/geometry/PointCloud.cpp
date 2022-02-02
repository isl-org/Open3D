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

#include "open3d/t/geometry/PointCloud.h"

#include <Eigen/Core>
#include <limits>
#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/hashmap/HashSet.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/t/geometry/kernel/Transform.h"

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
    if (point_attr_.size() == 0)
        return fmt::format("PointCloud on {} [0 points ()] Attributes: None.",
                           GetDevice().ToString());
    auto str =
            fmt::format("PointCloud on {} [{} points ({})] Attributes:",
                        GetDevice().ToString(), GetPointPositions().GetShape(0),
                        GetPointPositions().GetDtype().ToString());

    if (point_attr_.size() == 1) return str + " None.";
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
                        "Shape mismatch. Attribure {}, shape {}, is not "
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

PointCloud PointCloud::VoxelDownSample(
        double voxel_size, const core::HashBackendType &backend) const {
    if (voxel_size <= 0) {
        utility::LogError("voxel_size must be positive.");
    }
    core::Tensor points_voxeld = GetPointPositions() / voxel_size;
    core::Tensor points_voxeli = points_voxeld.Floor().To(core::Int64);

    core::HashSet points_voxeli_hashset(points_voxeli.GetLength(), core::Int64,
                                        {3}, device_, backend);

    core::Tensor buf_indices, masks;
    points_voxeli_hashset.Insert(points_voxeli, buf_indices, masks);

    PointCloud pcd_down(GetPointPositions().GetDevice());
    for (auto &kv : point_attr_) {
        if (kv.first == "positions") {
            pcd_down.SetPointAttr(kv.first,
                                  points_voxeli.IndexGet({masks}).To(
                                          GetPointPositions().GetDtype()) *
                                          voxel_size);
        } else {
            pcd_down.SetPointAttr(kv.first, kv.second.IndexGet({masks}));
        }
    }

    return pcd_down;
}

void PointCloud::EstimateNormals(
        const int max_knn /* = 30*/,
        const utility::optional<double> radius /*= utility::nullopt*/) {
    core::AssertTensorDtypes(this->GetPointPositions(),
                             {core::Float32, core::Float64});

    const core::Dtype dtype = this->GetPointPositions().GetDtype();
    const core::Device device = GetDevice();
    const core::Device::DeviceType device_type = device.GetType();
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

    if (radius.has_value()) {
        utility::LogDebug("Using Hybrid Search for computing covariances");
        // Computes and sets `covariances` attribute using Hybrid Search
        // mehtod.
        if (device_type == core::Device::DeviceType::CPU) {
            kernel::pointcloud::EstimateCovariancesUsingHybridSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointAttr("covariances"), radius.value(), max_knn);
        } else if (device_type == core::Device::DeviceType::CUDA) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateCovariancesUsingHybridSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointAttr("covariances"), radius.value(),
                      max_knn);
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogDebug("Using KNN Search for computing covariances");
        // Computes and sets `covariances` attribute using KNN Search method.
        if (device_type == core::Device::DeviceType::CPU) {
            kernel::pointcloud::EstimateCovariancesUsingKNNSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointAttr("covariances"), max_knn);
        } else if (device_type == core::Device::DeviceType::CUDA) {
            CUDA_CALL(kernel::pointcloud::EstimateCovariancesUsingKNNSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointAttr("covariances"), max_knn);
        } else {
            utility::LogError("Unimplemented device");
        }
    }

    // Estimate `normal` of each point using its `covariance` matrix.
    if (device_type == core::Device::DeviceType::CPU) {
        kernel::pointcloud::EstimateNormalsFromCovariancesCPU(
                this->GetPointAttr("covariances"), this->GetPointNormals(),
                has_normals);
    } else if (device_type == core::Device::DeviceType::CUDA) {
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

void PointCloud::EstimateColorGradients(
        const int max_knn /* = 30*/,
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
    const core::Device::DeviceType device_type = device.GetType();

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
    if (radius.has_value()) {
        utility::LogDebug("Using Hybrid Search for computing color_gradients");
        if (device_type == core::Device::DeviceType::CPU) {
            kernel::pointcloud::EstimateColorGradientsUsingHybridSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointNormals().Contiguous(),
                    this->GetPointColors().Contiguous(),
                    this->GetPointAttr("color_gradients"), radius.value(),
                    max_knn);
        } else if (device_type == core::Device::DeviceType::CUDA) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateColorGradientsUsingHybridSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointNormals().Contiguous(),
                      this->GetPointColors().Contiguous(),
                      this->GetPointAttr("color_gradients"), radius.value(),
                      max_knn);
        } else {
            utility::LogError("Unimplemented device");
        }
    } else {
        utility::LogDebug("Using KNN Search for computing color_gradients");
        if (device_type == core::Device::DeviceType::CPU) {
            kernel::pointcloud::EstimateColorGradientsUsingKNNSearchCPU(
                    this->GetPointPositions().Contiguous(),
                    this->GetPointNormals().Contiguous(),
                    this->GetPointColors().Contiguous(),
                    this->GetPointAttr("color_gradients"), max_knn);
        } else if (device_type == core::Device::DeviceType::CUDA) {
            CUDA_CALL(kernel::pointcloud::
                              EstimateColorGradientsUsingKNNSearchCUDA,
                      this->GetPointPositions().Contiguous(),
                      this->GetPointNormals().Contiguous(),
                      this->GetPointColors().Contiguous(),
                      this->GetPointAttr("color_gradients"), max_knn);
        } else {
            utility::LogError("Unimplemented device");
        }
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
            depth_in.GetDevice().GetType() == core::Device::DeviceType::CUDA
                    ? 3
                    : 5;
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

}  // namespace geometry
}  // namespace t
}  // namespace open3d
