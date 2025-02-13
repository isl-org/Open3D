// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <bits/stdint-intn.h>
#include <paddle/phi/common/place.h>

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"

template <class T>
void BuildSpatialHashTableCPU(const paddle::Tensor& points,
                              double radius,
                              const paddle::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              paddle::Tensor& hash_table_index,
                              paddle::Tensor& hash_table_cell_splits);
#ifdef BUILD_CUDA_MODULE
template <class T>
void BuildSpatialHashTableCUDA(const paddle::Tensor& points,
                               double radius,
                               const paddle::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               paddle::Tensor& hash_table_index,
                               paddle::Tensor& hash_table_cell_splits);
#endif

std::vector<paddle::Tensor> BuildSpatialHashTable(
        paddle::Tensor& points,
        paddle::Tensor& points_row_splits,
        double radius,
        double hash_table_size_factor,
        int64_t max_hash_table_size) {
    points_row_splits = points_row_splits.copy_to(phi::CPUPlace(), false);
    CHECK_TYPE(points_row_splits, paddle::DataType::INT64);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim batch_size("batch_size");

    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);

    const auto& point_type = points.dtype();

    std::vector<uint32_t> hash_table_splits(batch_size.value() + 1, 0);
    for (int i = 0; i < batch_size.value(); ++i) {
        int64_t num_points_i = points_row_splits.data<int64_t>()[i + 1] -
                               points_row_splits.data<int64_t>()[i];
        int64_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
                max_hash_table_size);
        hash_table_splits[i + 1] = hash_table_splits[i] + hash_table_size;
    }

    auto place = points.place();
    paddle::Tensor hash_table_index;
    if (points.shape()[0] != 0) {
        hash_table_index =
                paddle::empty({points.shape()[0]},
                              paddle::DataType(ToPaddleDtype<int>()), place);
    } else {
        hash_table_index = InitializedEmptyTensor<int>({0}, place);
    }
    paddle::Tensor hash_table_cell_splits =
            paddle::empty({hash_table_splits.back() + 1},
                          paddle::DataType(ToPaddleDtype<int32_t>()), place);
    paddle::Tensor out_hash_table_splits = paddle::empty(
            {batch_size.value() + 1},
            paddle::DataType(ToPaddleDtype<int32_t>()), phi::CPUPlace());
    for (size_t i = 0; i < hash_table_splits.size(); ++i) {
        out_hash_table_splits.data<int32_t>()[i] = hash_table_splits[i];
    }
#define FN_PARAMETERS                                                       \
    points, radius, points_row_splits, hash_table_splits, hash_table_index, \
            hash_table_cell_splits
#define CALL(type, fn)                                    \
    if (ComparePaddleDtype<type>(point_type)) {           \
        fn<type>(FN_PARAMETERS);                          \
        return {hash_table_index, hash_table_cell_splits, \
                out_hash_table_splits};                   \
    }
    if (points.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, BuildSpatialHashTableCUDA)
#else
        PD_CHECK(false,
                 "BuildSpatialHashTable was not compiled with CUDA support");
#endif
    } else {
        CALL(float, BuildSpatialHashTableCPU)
        CALL(double, BuildSpatialHashTableCPU)
    }
    PD_CHECK(false, "BuildSpatialHashTable does not support " +
                            phi::DataTypeToString(points.dtype()) +
                            " as input for "
                            "points");

    return std::vector<paddle::Tensor>();
}

std::vector<paddle::DataType> BuildSpatialHashTableInferDtype() {
    auto dtype = paddle::DataType::INT32;
    return {dtype, dtype, dtype};
}

PD_BUILD_OP(open3d_build_spatial_hash_table)
        .Inputs({"points", "points_row_splits"})
        .Outputs({"hash_table_index", "hash_table_cell_splits",
                  "hash_table_splits"})
        .Attrs({"radius: double", "hash_table_size_factor: double",
                "max_hash_table_size: int64_t"})
        .SetKernelFn(PD_KERNEL(BuildSpatialHashTable))
        .SetInferDtypeFn(PD_INFER_DTYPE(BuildSpatialHashTableInferDtype));
