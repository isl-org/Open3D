// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
//

#define EIGEN_USE_GPU
#include "FixedRadiusSearchOpKernel.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/ML/Misc/Detail/FixedRadiusSearch.cuh"

using namespace open3d;
using namespace open3d::ml::detail;
using namespace fixed_radius_search_opkernel;
using namespace tensorflow;

template <class T>
class FixedRadiusSearchOpKernelCUDA : public FixedRadiusSearchOpKernel {
public:
    explicit FixedRadiusSearchOpKernelCUDA(OpKernelConstruction* construction)
        : FixedRadiusSearchOpKernel(construction) {
        texture_alignment = GetCUDACurrentDeviceTextureAlignment();
    }

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& queries,
                const tensorflow::Tensor& radius,
                const size_t hash_table_size,
                tensorflow::Tensor& query_neighbors_row_splits) {
        auto device = context->eigen_gpu_device();

        OutputAllocator<T> output_allocator(context);

        Tensor hash_table_row_splits;
        TensorShape hash_table_row_splits_shape({ssize_t(hash_table_size + 1)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint32_t>::v(),
                                              hash_table_row_splits_shape,
                                              &hash_table_row_splits));

        Tensor hash_table_index;
        TensorShape hash_table_index_shape({points.shape().dim_size(0)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint32_t>::v(),
                                              hash_table_index_shape,
                                              &hash_table_index));

        void* temp_ptr = nullptr;
        size_t temp_size = 0;
        size_t temp_size_hash_table = 0;

        // determine temp_size
        BuildSpatialHashTableCUDA(device.stream(), temp_ptr,
                                  temp_size_hash_table, texture_alignment,
                                  points.shape().dim_size(0),
                                  points.flat<T>().data(), radius.scalar<T>()(),
                                  hash_table_size + 1,
                                  hash_table_row_splits.flat<uint32_t>().data(),
                                  hash_table_index.flat<uint32_t>().data());

        FixedRadiusSearchCUDA(
                device.stream(), temp_ptr, temp_size, texture_alignment,
                (int64_t*)query_neighbors_row_splits.flat<int64>().data(),
                points.shape().dim_size(0), points.flat<T>().data(),
                queries.shape().dim_size(0), queries.flat<T>().data(),
                radius.scalar<T>()(), hash_table_row_splits.shape().dim_size(0),
                hash_table_row_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data(), metric,
                ignore_query_point, return_distances, output_allocator);

        temp_size = std::max(temp_size, temp_size_hash_table);
        Tensor temp_tensor;
        TensorShape temp_shape({ssize_t(temp_size)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint8_t>::v(),
                                              temp_shape, &temp_tensor));
        temp_ptr = temp_tensor.flat<uint8_t>().data();

        // actually run the search
        BuildSpatialHashTableCUDA(device.stream(), temp_ptr,
                                  temp_size_hash_table, texture_alignment,
                                  points.shape().dim_size(0),
                                  points.flat<T>().data(), radius.scalar<T>()(),
                                  hash_table_size + 1,
                                  hash_table_row_splits.flat<uint32_t>().data(),
                                  hash_table_index.flat<uint32_t>().data());

        FixedRadiusSearchCUDA(
                device.stream(), temp_ptr, temp_size, texture_alignment,
                (int64_t*)query_neighbors_row_splits.flat<int64>().data(),
                points.shape().dim_size(0), points.flat<T>().data(),
                queries.shape().dim_size(0), queries.flat<T>().data(),
                radius.scalar<T>()(), hash_table_row_splits.shape().dim_size(0),
                hash_table_row_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data(), metric,
                ignore_query_point, return_distances, output_allocator);
    }

private:
    int texture_alignment;
};

#define REG_KB(type)                                                       \
    REGISTER_KERNEL_BUILDER(Name("Open3DFixedRadiusSearch")                \
                                    .Device(DEVICE_GPU)                    \
                                    .TypeConstraint<type>("T")             \
                                    .HostMemory("radius")                  \
                                    .HostMemory("hash_table_size_factor"), \
                            FixedRadiusSearchOpKernelCUDA<type>);
REG_KB(float)
#undef REG_KB
