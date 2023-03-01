// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "FixedRadiusSearchOpKernel.h"

#include "open3d/core/nns/FixedRadiusSearchImpl.h"

using namespace fixed_radius_search_opkernel;
using namespace tensorflow;

template <class T, class TIndex>
class FixedRadiusSearchOpKernelCPU : public FixedRadiusSearchOpKernel {
public:
    explicit FixedRadiusSearchOpKernelCPU(OpKernelConstruction* construction)
        : FixedRadiusSearchOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& queries,
                const tensorflow::Tensor& radius,
                const tensorflow::Tensor& points_row_splits,
                const tensorflow::Tensor& queries_row_splits,
                const tensorflow::Tensor& hash_table_splits,
                const tensorflow::Tensor& hash_table_index,
                const tensorflow::Tensor& hash_table_cell_splits,
                tensorflow::Tensor& query_neighbors_row_splits) {
        OutputAllocator<T, TIndex> output_allocator(context);

        open3d::core::nns::impl::FixedRadiusSearchCPU<T, TIndex>(
                (int64_t*)query_neighbors_row_splits.flat<int64>().data(),
                points.shape().dim_size(0), points.flat<T>().data(),
                queries.shape().dim_size(0), queries.flat<T>().data(),
                radius.scalar<T>()(), points_row_splits.shape().dim_size(0),
                (int64_t*)points_row_splits.flat<int64>().data(),
                queries_row_splits.shape().dim_size(0),
                (int64_t*)queries_row_splits.flat<int64>().data(),
                hash_table_splits.flat<uint32_t>().data(),
                hash_table_cell_splits.shape().dim_size(0),
                hash_table_cell_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data(), metric,
                ignore_query_point, return_distances, output_allocator);
    }
};

#define REG_KB(type, itype)                                                \
    REGISTER_KERNEL_BUILDER(Name("Open3DFixedRadiusSearch")                \
                                    .Device(DEVICE_CPU)                    \
                                    .TypeConstraint<type>("T")             \
                                    .TypeConstraint<itype>("index_dtype"), \
                            FixedRadiusSearchOpKernelCPU<type, itype>);
REG_KB(float, int)
REG_KB(float, long)
REG_KB(double, int)
REG_KB(double, long)
#undef REG_KB
