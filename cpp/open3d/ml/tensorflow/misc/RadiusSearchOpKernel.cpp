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

#include "RadiusSearchOpKernel.h"

#include "open3d/core/nns/NanoFlannImpl.h"

using namespace open3d::core::nns;
using namespace radius_search_opkernel;
using namespace tensorflow;

namespace {

template <class T>
class OutputAllocator {
public:
    explicit OutputAllocator(tensorflow::OpKernelContext* context)
        : context(context) {}

    void AllocIndices(int32_t** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<int32>();
        static_assert(sizeof(int32) == sizeof(int32_t),
                      "int32 and int32_t not compatible");
        *ptr = (int32_t*)flat_tensor.data();
    }

    void AllocDistances(T** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(2, shape, &tensor));
        auto flat_tensor = tensor->flat<T>();
        *ptr = flat_tensor.data();
    }

private:
    tensorflow::OpKernelContext* context;
};

template <class T>
class OutputAllocatorTmp {
public:
    OutputAllocatorTmp() {}

    void AllocIndices(int32_t** ptr, size_t num) {
        index.resize(num);
        *ptr = index.data();
    }

    void AllocDistances(T** ptr, size_t num) {
        distance.resize(num);
        *ptr = distance.data();
    }

    std::vector<int32_t> index;
    std::vector<T> distance;
};

}  // namespace

template <class T>
class RadiusSearchOpKernelCPU : public RadiusSearchOpKernel {
public:
    explicit RadiusSearchOpKernelCPU(OpKernelConstruction* construction)
        : RadiusSearchOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& queries,
                const tensorflow::Tensor& radius,
                const tensorflow::Tensor& points_row_splits,
                const tensorflow::Tensor& queries_row_splits,
                tensorflow::Tensor& query_neighbors_row_splits) {
        const int batch_size = points_row_splits.shape().dim_size(0) - 1;
        if (batch_size == 1) {
            OutputAllocator<T> output_allocator(context);

            std::unique_ptr<NanoFlannIndexHolderBase> holder =
                    impl::BuildKdTree<T>(points.shape().dim_size(0),
                                         points.flat<T>().data(),
                                         points.shape().dim_size(1), metric);
            impl::RadiusSearchCPU(
                    holder.get(),
                    (int64_t*)query_neighbors_row_splits.flat<int64>().data(),
                    points.shape().dim_size(0), points.flat<T>().data(),
                    queries.shape().dim_size(0), queries.flat<T>().data(), 3,
                    radius.flat<T>().data(), metric, ignore_query_point,
                    return_distances, normalize_distances, /* sort */ false,
                    output_allocator);
        } else {
            // run radius search for each batch item
            std::vector<OutputAllocatorTmp<T>> output_allocators(batch_size);
            int64_t last_neighbors_count = 0;
            for (int i = 0; i < batch_size; ++i) {
                const T* const points_i =
                        points.flat<T>().data() +
                        3 * points_row_splits.flat<int64>()(i);
                const T* const queries_i =
                        queries.flat<T>().data() +
                        3 * queries_row_splits.flat<int64>()(i);
                const T* const radius_i = radius.flat<T>().data() +
                                          queries_row_splits.flat<int64>()(i);
                size_t num_points_i = points_row_splits.flat<int64>()(i + 1) -
                                      points_row_splits.flat<int64>()(i);
                size_t num_queries_i = queries_row_splits.flat<int64>()(i + 1) -
                                       queries_row_splits.flat<int64>()(i);

                int64_t* neighbors_row_splits_i =
                        (int64_t*)(query_neighbors_row_splits.flat<int64>()
                                           .data() +
                                   queries_row_splits.flat<int64>()(i));

                std::unique_ptr<NanoFlannIndexHolderBase> holder =
                        impl::BuildKdTree<T>(num_points_i, points_i,
                                             points.shape().dim_size(1),
                                             metric);
                impl::RadiusSearchCPU(holder.get(), neighbors_row_splits_i,
                                      num_points_i, points_i, num_queries_i,
                                      queries_i, 3, radius_i, metric,
                                      ignore_query_point, return_distances,
                                      normalize_distances, /* sort */ false,
                                      output_allocators[i]);
                if (i > 0) {
                    for (size_t j = 0; j <= num_queries_i; ++j)
                        neighbors_row_splits_i[j] += last_neighbors_count;
                }
                last_neighbors_count = neighbors_row_splits_i[num_queries_i];
            }

            // combine results
            int64_t neighbors_index_size = 0;
            int64_t neighbors_distance_size = 0;
            for (const auto& a : output_allocators) {
                neighbors_index_size += a.index.size();
                neighbors_distance_size += a.distance.size();
            }

            Tensor* neighbors_index_tensor;
            TensorShape neighbors_index_shape({neighbors_index_size});
            OP_REQUIRES_OK(context,
                           context->allocate_output(0, neighbors_index_shape,
                                                    &neighbors_index_tensor));
            int32_t* neighbors_index_data_ptr =
                    neighbors_index_tensor->flat<int32_t>().data();

            Tensor* neighbors_distance_tensor;
            TensorShape neighbors_distance_shape({neighbors_distance_size});
            OP_REQUIRES_OK(context, context->allocate_output(
                                            2, neighbors_distance_shape,
                                            &neighbors_distance_tensor));
            T* neighbors_distance_data_ptr =
                    neighbors_distance_tensor->flat<T>().data();

            for (int i = 0; i < batch_size; ++i) {
                const auto& a = output_allocators[i];
                if (a.index.size()) {
                    for (const auto index : a.index) {
                        neighbors_index_data_ptr[0] =
                                index + points_row_splits.flat<int64>()(i);
                        ++neighbors_index_data_ptr;
                    }
                }
                if (a.distance.size()) {
                    memcpy(neighbors_distance_data_ptr, a.distance.data(),
                           a.distance.size() * sizeof(T));
                    neighbors_distance_data_ptr += a.distance.size();
                }
            }
        }
    }
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("Open3DRadiusSearch")          \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            RadiusSearchOpKernelCPU<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
