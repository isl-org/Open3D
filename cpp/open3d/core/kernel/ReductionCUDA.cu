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

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/FunctionTraits.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Reduction.h"
#include "open3d/utility/Logging.h"

// CUDA reduction is based on PyTorch's CUDA reduction implementation.
// See: aten/src/ATen/native/cuda/Reduce.cuh

#if __CUDA_ARCH__ >= 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif

constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

#define OPEN3D_MAX_THREADS_PER_BLOCK(val)          \
    (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                           : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define OPEN3D_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)       \
    ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
              ? (blocks_per_sm)                                          \
              : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /     \
                 (threads_per_block))))

#define OPEN3D_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm)       \
    __launch_bounds__((OPEN3D_MAX_THREADS_PER_BLOCK((max_threads_per_block))), \
                      (OPEN3D_MIN_BLOCKS_PER_SM((max_threads_per_block),       \
                                                (min_blocks_per_sm))))

template <typename T>
OPEN3D_DEVICE __forceinline__ T WARP_SHFL_DOWN(T value,
                                               unsigned int delta,
                                               int width = warpSize,
                                               unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

namespace open3d {
namespace core {
namespace kernel {

static inline int64_t DivUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Returns reduced fraction numerator & denominator
OPEN3D_HOST_DEVICE static void ReduceFraction(int64_t& numerator,
                                              int64_t& denominator) {
    // Get GCD of num and denom using Euclid's algorithm.
    // Can replace this with std::gcd if we ever support c++17.
    int64_t a = denominator;
    int64_t b = numerator;
    while (b != 0) {
        a %= b;
        int64_t tmp = a;
        a = b;
        b = tmp;
    }
    // a is now the GCD
    numerator /= a;
    denominator /= a;
}

class ReduceConfig {
public:
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA = 2;
    static constexpr int MAX_NUM_THREADS = 512;

    int num_inputs_per_output_;
    int num_outputs_;
    int step_input_ = 1;
    int step_output_ = 1;
    int ctas_per_output_ = 1;

private:
    int element_size_bytes_;
    int input_mult_[3] = {0, 0, 0};
    int output_mult_[2] = {0, 0};
    int block_width_;
    int block_height_;
    int num_threads_;

public:
    ReduceConfig(int element_size_bytes, const Indexer& indexer)
        : element_size_bytes_(element_size_bytes) {
        num_outputs_ = indexer.NumOutputElements();
        num_inputs_per_output_ = indexer.NumWorkloads() / num_outputs_;

        // Adjust block size to map block width to fastest changing dimension of
        // input tensor. This grants the best possible memory accessing pattern,
        // given that for non-contiguous tensor with space in between, we cannot
        // have perfect memory coalescing.
        bool reduction_on_fastest_striding_dimension =
                (indexer.NumReductionDims() == indexer.NumDims()) ||
                (indexer.GetInput(0).byte_strides_[0] <
                 indexer.GetInput(0).byte_strides_[indexer.NumReductionDims()]);

        // Notice that dim0 & dim1 does NOT guarantee any launch configuration
        // here! dim0 & dim1 are more like the upper bound of the block
        // dimension. The actual launch config and reduction scheme is
        // determined by setting values to `input_mult_` and
        // `output_mult_`. We try to max out dim1 so that we have enough
        // threads per CTA to deliver performance for larger problem size.
        int64_t dim0;
        int64_t dim1;

        if (reduction_on_fastest_striding_dimension) {
            // Map block.x to the fastest reducing dimension. It implies:
            //   1. BlockXReduce is required.
            //   2. block.y now max out to num_outputs.
            dim0 = indexer.GetMasterShape()[0];
            dim1 = num_outputs_;
        } else {
            // Map block.x to the fastest non reducing dimension. It implies:
            //   1. BlockXReduce is turned off.
            //   2. block.y now max out to num_inputs_per_output_.
            dim0 = indexer.GetMasterShape()[indexer.NumReductionDims()];
            dim1 = num_inputs_per_output_;
        }

        // Adjust block_width and block_height
        SetBlockDimension(dim0, dim1);

        int block_width = block_width_;
        int block_height = block_height_;

        if (indexer.NumDims() == 0 || reduction_on_fastest_striding_dimension) {
            // Split the input across lanes if the input is contiguous in the
            // reduced dimension. This will require reduction between threads
            // using warp shuffle instructions and shared memory (if
            // block_width > warpSize).
            input_mult_[0] = SplitInput(block_width);
        } else {
            // Otherwise split the output across lanes in a warp.
            output_mult_[0] = SplitOutput(block_width);
        }

        if (ValuesPerThread() >= block_height * 16 ||
            ValuesPerThread() >= 256) {
            // Divide the input across warps in a thread-block, if that leaves
            // at least 16 elements to be summed by each thread. This will
            // require inter-warp reduction using shared memory.
            input_mult_[1] = SplitInput(block_height);
        } else {
            // Otherwise, each warp handles a separate output.
            output_mult_[1] = SplitOutput(block_height);
        }

        if (input_mult_[1] != 0 && ValuesPerThread() >= 256 &&
            num_outputs_ <= 4096) {
            // Divide the input across thread-blocks if the amount of work
            // per-thread is large enough and the size of the output is small
            // enough. This will require a reduction using global memory.
            ctas_per_output_ = DivUp(ValuesPerThread(), 16);
            if (ctas_per_output_ > 65535) {
                ctas_per_output_ = 65535;
            }
            input_mult_[2] = SplitInput(ctas_per_output_);
        }
    }

    /// Returns floor(log2(n))
    static inline int LastPow2(int n) {
        // Dtype.h asserts sizeof(int) == 4.
        n |= (n >> 1);
        n |= (n >> 2);
        n |= (n >> 4);
        n |= (n >> 8);
        n |= (n >> 16);
        return std::max(1, n - (n >> 1));
    }

    void SetBlockDimension(int64_t dim0, int64_t dim1) {
        int dim0_pow2 = dim0 < MAX_NUM_THREADS
                                ? static_cast<int>(LastPow2(dim0))
                                : MAX_NUM_THREADS;
        int dim1_pow2 = dim1 < MAX_NUM_THREADS
                                ? static_cast<int>(LastPow2(dim1))
                                : MAX_NUM_THREADS;
        block_width_ = std::min(dim0_pow2, GetCUDACurrentWarpSize());
        block_height_ =
                std::min(dim1_pow2, int(MAX_NUM_THREADS / block_width_));
        block_width_ =
                std::min(dim0_pow2, int(MAX_NUM_THREADS / block_height_));
        num_threads_ = block_width_ * block_height_;
    }

    int SplitInput(int parallelism) {
        int step = step_input_;
        step_input_ *= parallelism;
        return step;
    }

    int SplitOutput(int parallelism) {
        int step = step_output_;
        step_output_ *= parallelism;
        return step;
    }

    dim3 BlockDim() const { return dim3(block_width_, block_height_); }

    dim3 GridDim() const {
        return dim3(DivUp(num_outputs_, step_output_), ctas_per_output_);
    }

    OPEN3D_HOST_DEVICE bool ShouldBlockXReduce() const {
        return input_mult_[BLOCK_X] != 0;
    }

    OPEN3D_HOST_DEVICE bool ShouldBlockYReduce() const {
        return input_mult_[BLOCK_Y] != 0;
    }

    OPEN3D_HOST_DEVICE bool ShouldGlobalReduce() const {
        return input_mult_[CTA] != 0;
    }

    OPEN3D_DEVICE bool ShouldStore(int output_idx) const {
        return output_idx < num_outputs_ &&
               (!ShouldBlockXReduce() || threadIdx.x == 0) &&
               (!ShouldBlockYReduce() || threadIdx.y == 0);
    }

    OPEN3D_HOST_DEVICE int InputIdx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta2 = blockIdx.y;
        return (lane * input_mult_[BLOCK_X] + warp * input_mult_[BLOCK_Y] +
                cta2 * input_mult_[CTA]);
    }

    OPEN3D_HOST_DEVICE int OutputIdx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta1 = blockIdx.x;
        return (lane * output_mult_[BLOCK_X] + warp * output_mult_[BLOCK_Y] +
                cta1 * step_output_);
    }

    OPEN3D_DEVICE int SharedMemoryOffset(int offset) const {
        return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
    }

    OPEN3D_DEVICE int StagingMemoryOffset(int cta2) const {
        int offset = cta2 + blockIdx.x * gridDim.y;
        if (!ShouldBlockXReduce()) {
            offset = threadIdx.x + offset * blockDim.x;
        }
        return offset;
    }

    int SharedMemorySize() const {
        if (!ShouldBlockYReduce() &&
            (!ShouldBlockXReduce() ||
             block_width_ <= GetCUDACurrentWarpSize())) {
            return 0;
        }
        return element_size_bytes_ * num_threads_;
    }

    int64_t GlobalMemorySize() const {
        if (!ShouldGlobalReduce()) {
            return 0;
        }
        auto size =
                (int64_t)element_size_bytes_ * num_outputs_ * ctas_per_output_;
        if (!ShouldBlockXReduce()) {
            size *= BlockDim().x;
        }
        return size;
    }

    int SemaphoreSize() const {
        if (!ShouldGlobalReduce()) {
            return 0;
        }
        return sizeof(int) * GridDim().x;
    }

    int ValuesPerThread() const {
        return DivUp(num_inputs_per_output_, step_input_);
    }

    std::string ToString() const {
        std::string input_mult_str = fmt::format(
                "[{},{},{}]", input_mult_[0], input_mult_[1], input_mult_[2]);
        std::string output_mult_str =
                fmt::format("[{},{}]", output_mult_[0], output_mult_[1]);
        std::string block_str = fmt::format("[{},{},{}]", BlockDim().x,
                                            BlockDim().y, BlockDim().z);
        std::string grid_str = fmt::format("[{},{},{}]", GridDim().x,
                                           GridDim().y, GridDim().z);
        std::string str = fmt::format(
                "REDUCEConfig(element_size_bytes_={}, "
                "num_inputs_per_output_={}, num_outputs_={}, "
                "step_input_={}, step_output_={}, ctas_per_output_={}, "
                "input_mult_={}, output_mult_={}, values_per_thread={}, "
                "block={}, grid={}, global_memory_size={})",
                element_size_bytes_, num_inputs_per_output_, num_outputs_,
                step_input_, step_output_, ctas_per_output_, input_mult_str,
                output_mult_str, ValuesPerThread(), block_str, grid_str,
                GlobalMemorySize());
        return str;
    }
};

template <int nt, typename R>
OPEN3D_LAUNCH_BOUNDS_2(nt, 4)
__global__ void ReduceKernel(R reduction) {
    reduction.Run();
}

template <typename index_t>
static OffsetCalculator<2, index_t> MakeOutputCalculator(
        const Indexer& indexer) {
    int num_reduction_dims = indexer.NumReductionDims();
    int num_output_dims = indexer.NumDims() - num_reduction_dims;
    std::array<const int64_t*, 2> strides = {
            indexer.GetOutput().byte_strides_ + num_reduction_dims,
            indexer.GetInput(0).byte_strides_ + num_reduction_dims,
    };
    const int64_t* shape = indexer.GetMasterShape() + num_reduction_dims;
    return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data());
}

template <typename index_t>
static OffsetCalculator<1, index_t> MakeInputCalculator(
        const Indexer& indexer) {
    int num_reduction_dims = indexer.NumReductionDims();
    std::array<const int64_t*, 1> strides = {
            indexer.GetInput(0).byte_strides_,
    };
    return OffsetCalculator<1, index_t>(
            num_reduction_dims, indexer.GetMasterShape(), strides.data());
}

template <int vt, typename index_t, typename func_t>
OPEN3D_DEVICE void StridedIterate(func_t f,
                                  index_t begin,
                                  index_t end,
                                  index_t stride) {
    if (begin + (vt - 1) * stride < end) {
#pragma unroll
        for (index_t i = 0; i < vt; i++) {
            f(i, begin + i * stride);
        }
    } else {
#pragma unroll
        for (index_t i = 0; i < vt; i++) {
            index_t idx = begin + i * stride;
            if (idx < end) {
                f(i, idx);
            }
        }
    }
}

/// Combime() and Reduce() are the same for regular reduction ops.
template <typename out_scalar_t, typename func_t>
class RegularReduceOps {
    using arg_t = typename BinaryFunctionTraits<func_t>::arg0_t;
    using scalar_t = typename BinaryFunctionTraits<func_t>::arg1_t;

public:
    RegularReduceOps(const func_t& op) : reduce_func_(op) {}

    static inline OPEN3D_DEVICE out_scalar_t Project(arg_t arg) {
        return (out_scalar_t)arg;
    }

    static inline OPEN3D_DEVICE arg_t WarpShflDown(arg_t arg, int offset) {
        return WARP_SHFL_DOWN(arg, offset);
    }

    OPEN3D_DEVICE inline arg_t Combine(arg_t acc, scalar_t val) const {
        return reduce_func_(acc, val);
    }

    /// Idx is ignored for RegularReduceOps.
    OPEN3D_DEVICE inline arg_t Reduce(arg_t acc,
                                      scalar_t val,
                                      int64_t idx) const {
        return reduce_func_(acc, val);
    }

private:
    func_t reduce_func_ = nullptr;
};

template <typename scalar_t, typename func_t>
RegularReduceOps<scalar_t, func_t> WrapRegularReduceOps(const func_t& op) {
    return RegularReduceOps<scalar_t, func_t>{op};
}

template <typename func_t>
class ArgReduceOps {
    using scalar_t = typename BinaryFunctionTraits<func_t>::arg1_t;
    using index_t = int64_t;
    using arg_t = thrust::pair<scalar_t, index_t>;

public:
    ArgReduceOps(const func_t comp_func) : comp_func_(comp_func) {}

    static OPEN3D_DEVICE index_t Project(arg_t arg) { return arg.second; }

    static OPEN3D_DEVICE arg_t WarpShflDown(arg_t arg, int offset) {
        return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                     WARP_SHFL_DOWN(arg.second, offset));
    }

    /// Combine(pair<val_t, idx_t>, pair<val_t, idx_t>) -> pair<val_t, idx_t>.
    /// Called at subsequent rounds of reduction, when values are already
    /// associated with indices.
    OPEN3D_DEVICE inline arg_t Combine(arg_t a, arg_t b) const {
        return comp_func_(a.first, b.first) ? a : b;
    }

    /// Reduce(pair<val_t, idx_t>, val_t, idx_t) -> pair<val_t, idx_t>.
    /// Called at the first round of reduction, when values are not yet
    /// associated with indices.
    OPEN3D_DEVICE inline arg_t Reduce(arg_t arg,
                                      scalar_t val,
                                      int64_t idx) const {
        return comp_func_(arg.first, val) ? arg : arg_t(val, idx);
    }

private:
    func_t comp_func_ = nullptr;
};

template <typename func_t>
ArgReduceOps<func_t> WrapArgReduceOps(const func_t& comp_func) {
    return ArgReduceOps<func_t>{comp_func};
}

template <typename scalar_t,
          typename ops_t,
          typename index_t,
          typename out_scalar_t = scalar_t,
          int vt0 = 4>
class ReduceOp {
    using traits = FunctionTraits<decltype(&ops_t::Reduce)>;
    using arg_t =
            typename std::decay<typename traits::template arg<0>::type>::type;
    using InputCalculator = OffsetCalculator<1, index_t>;
    using OutputCalculator = OffsetCalculator<2, index_t>;

public:
    ReduceOp(ops_t ops,
             ReduceConfig config,
             InputCalculator input_calc,
             OutputCalculator output_calc,
             const void* src,
             char* dst,
             void* acc_buf,
             void* cta_buf,
             int* semaphores,
             arg_t identity,
             bool accumulate,
             bool final_output)
        : ops_(ops),
          config_(config),
          input_calc_(input_calc),
          output_calc_(output_calc),
          src_(src),
          dst_(dst),
          acc_buf_(acc_buf),
          cta_buf_(cta_buf),
          semaphores_(semaphores),
          identity_(identity),
          accumulate_(accumulate),
          final_output_(final_output) {}

    OPEN3D_DEVICE void Run() const {
        extern __shared__ char shared_memory[];
        index_t output_idx = config_.OutputIdx();
        index_t input_idx = config_.InputIdx();
        auto base_offsets = output_calc_.get(output_idx);

        arg_t value = identity_;
        if (output_idx < config_.num_outputs_ &&
            input_idx < config_.num_inputs_per_output_) {
            auto input_slice = (const char*)src_ + base_offsets[1];
            value = ThreadReduce((const scalar_t*)input_slice);
        }

        if (config_.ShouldBlockYReduce()) {
            value = BlockYReduce(value, shared_memory);
        }
        if (config_.ShouldBlockXReduce()) {
            value = BlockXReduce(value, shared_memory);
        }

        auto out = (out_scalar_t*)((char*)dst_ + base_offsets[0]);
        arg_t* acc = nullptr;
        if (acc_buf_ != nullptr) {
            int64_t numerator = (int64_t)sizeof(arg_t);
            int64_t denominator = (int64_t)sizeof(out_scalar_t);
            ReduceFraction(numerator, denominator);
            acc = (arg_t*)((char*)acc_buf_ +
                           (base_offsets[0] * numerator / denominator));
        }

        if (config_.ShouldGlobalReduce()) {
            value = GlobalReduce(value, acc, shared_memory);
        } else if (config_.ShouldStore(output_idx)) {
            if (acc == nullptr) {
                if (accumulate_) {
                    value = AccumulateInOutput<can_accumulate_in_output>(out,
                                                                         value);
                }
                if (final_output_) {
                    SetResultsToOutput(value, base_offsets[0]);
                } else {
                    *out = GetAccumulatedOutput<can_accumulate_in_output>(
                            out, value);
                }
            } else {
                if (accumulate_) {
                    value = ops_.Combine(*acc, value);
                }
                if (final_output_) {
                    SetResultsToOutput(value, base_offsets[0]);
                } else {
                    *acc = value;
                }
            }
        }
    }

    OPEN3D_DEVICE arg_t ThreadReduce(const scalar_t* data) const {
        index_t idx = config_.InputIdx();
        // Multiple accumulators to remove dependency between unrolled loops.
        arg_t value_list[vt0];
#pragma unroll
        for (int i = 0; i < vt0; i++) {
            value_list[i] = identity_;
        }
        index_t end = config_.num_inputs_per_output_;
        index_t stride = config_.step_input_;
        index_t element_stride = input_calc_.strides_[0][0] / sizeof(scalar_t);

        // Reducing layers of function calls so compiler could do proper loop
        // unroll that exposes instruction level parallelism.
        while (idx < config_.num_inputs_per_output_) {
            // load input
            utility::MiniVec<scalar_t, vt0> values;
            if (input_calc_.dims_ == 1) {
                StridedIterate<vt0>(
                        [&](index_t i, index_t idx) {
                            values[i] = data[idx * element_stride];
                        },
                        idx, end, stride);
            } else {
                StridedIterate<vt0>(
                        [&](index_t i, index_t idx) {
                            values[i] = data[input_calc_.get(idx)[0] /
                                             sizeof(scalar_t)];
                        },
                        idx, end, stride);
            }
            // compute
            StridedIterate<vt0, index_t>(
                    [&](index_t i, index_t idx) {
                        value_list[i] =
                                ops_.Reduce(value_list[i], values[i], idx);
                    },
                    idx, config_.num_inputs_per_output_, config_.step_input_);
            // step offset
            idx += config_.step_input_ * vt0;
        }
#pragma unroll
        for (int i = 1; i < vt0; i++) {
            value_list[0] = ops_.Combine(value_list[0], value_list[i]);
        }
        return value_list[0];
    }

    OPEN3D_DEVICE arg_t BlockXReduce(arg_t value, char* shared_memory) const {
        int dim_x = blockDim.x;
        arg_t* shared = (arg_t*)shared_memory;
        if (dim_x > warpSize) {
            int address_base = threadIdx.x + threadIdx.y * blockDim.x;
            shared[address_base] = value;
            for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                    arg_t other = shared[address_base + offset];
                    value = ops_.Combine(value, other);
                    shared[address_base] = value;
                }
            }
            dim_x = warpSize;
        }

        __syncthreads();

        for (int offset = 1; offset < dim_x; offset <<= 1) {
            arg_t other = ops_.WarpShflDown(value, offset);
            value = ops_.Combine(value, other);
        }
        return value;
    }

    OPEN3D_DEVICE arg_t BlockYReduce(arg_t value, char* shared_memory) const {
        arg_t* shared = (arg_t*)shared_memory;
        shared[config_.SharedMemoryOffset(0)] = value;
        for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                arg_t other = shared[config_.SharedMemoryOffset(offset)];
                value = ops_.Combine(value, other);
                shared[config_.SharedMemoryOffset(0)] = value;
            }
        }
        return value;
    }

    OPEN3D_DEVICE bool MarkBlockFinished() const {
        __shared__ bool is_last_block_done_shared;

        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int prev_blocks_finished = atomicAdd(&semaphores_[blockIdx.x], 1);
            is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
        }

        __syncthreads();

        return is_last_block_done_shared;
    }

    template <bool can_acc>
    OPEN3D_DEVICE arg_t AccumulateInOutput(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<can_acc>::type* = nullptr) const {
        return ops_.Combine(*out, value);
    }

    // This function should never be called --
    // it's the version of `AccumulateInOutput`
    // when accumulation in the output is not possible.
    template <bool can_acc>
    OPEN3D_DEVICE arg_t AccumulateInOutput(
            out_scalar_t*,
            arg_t,
            typename std::enable_if<!can_acc>::type* = nullptr) const {
        OPEN3D_ASSERT(false);
        return arg_t{};
    }

    template <bool can_acc>
    OPEN3D_DEVICE out_scalar_t GetAccumulatedOutput(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<can_acc>::type* = nullptr) const {
        OPEN3D_ASSERT(!final_output_);
        return (out_scalar_t)value;
    }

    // This function should never be called --
    // it's the version of `GetAccumulatedOutput`
    // when accumulation in the output is not possible.
    template <bool can_acc>
    OPEN3D_DEVICE out_scalar_t GetAccumulatedOutput(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<!can_acc>::type* = nullptr) const {
        OPEN3D_ASSERT(false);
        return *out;
    }

    template <class T>
    OPEN3D_DEVICE void SetResults(const T x, const index_t base_offset) const {
        auto res = (out_scalar_t*)((char*)dst_ + base_offset);
        *res = x;
    }

    OPEN3D_DEVICE void SetResultsToOutput(arg_t value,
                                          index_t base_offset) const {
        OPEN3D_ASSERT(final_output_);
        SetResults(ops_.Project(value), base_offset);
    }

    OPEN3D_DEVICE arg_t GlobalReduce(arg_t value,
                                     arg_t* acc,
                                     char* shared_memory) const {
        arg_t* reduce_buffer = (arg_t*)cta_buf_;
        index_t output_idx = config_.OutputIdx();
        auto base_offsets = output_calc_.get(output_idx);
        auto out = (out_scalar_t*)((char*)dst_ + base_offsets[0]);

        bool should_store = config_.ShouldStore(config_.OutputIdx());
        if (should_store) {
            index_t offset = config_.StagingMemoryOffset(blockIdx.y);
            reduce_buffer[offset] = value;
        }

        __threadfence();  // make sure writes are globally visible
        __syncthreads();  // if multiple warps in this block wrote to staging,
                          // make sure they're all done
        bool is_last_block_done = MarkBlockFinished();

        if (is_last_block_done) {
            value = identity_;
            if (config_.ShouldBlockXReduce()) {
                index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
                index_t step = blockDim.x * blockDim.y;
                for (; input_offset < config_.ctas_per_output_;
                     input_offset += step) {
                    index_t idx = config_.StagingMemoryOffset(input_offset);
                    arg_t next = reduce_buffer[idx];
                    value = ops_.Combine(value, next);
                }
            } else {
                index_t input_offset = threadIdx.y;
                index_t step = blockDim.y;
                for (; input_offset < config_.ctas_per_output_;
                     input_offset += step) {
                    index_t idx = config_.StagingMemoryOffset(input_offset);
                    arg_t next = reduce_buffer[idx];
                    value = ops_.Combine(value, next);
                }
            }
            value = BlockYReduce(value, shared_memory);
            if (config_.ShouldBlockXReduce()) {
                value = BlockXReduce(value, shared_memory);
            }
            if (should_store) {
                if (acc == nullptr) {
                    if (accumulate_) {
                        value = AccumulateInOutput<can_accumulate_in_output>(
                                out, value);
                    }
                    if (final_output_) {
                        SetResultsToOutput(value, base_offsets[0]);
                    } else {
                        *out = GetAccumulatedOutput<can_accumulate_in_output>(
                                out, value);
                    }
                } else {
                    if (accumulate_) {
                        value = ops_.Combine(*acc, value);
                    }
                    if (final_output_) {
                        SetResultsToOutput(value, base_offsets[0]);
                    } else {
                        *acc = value;
                    }
                }
            }
        }

        return value;
    }

private:
    static constexpr bool can_accumulate_in_output =
            std::is_convertible<arg_t, out_scalar_t>::value &&
            std::is_convertible<out_scalar_t, arg_t>::value;
    static constexpr float acc_buffer_multiplier =
            (float)sizeof(arg_t) / sizeof(out_scalar_t);
    ops_t ops_;
    ReduceConfig config_;
    InputCalculator input_calc_;
    OutputCalculator output_calc_;
    const void* src_;
    const char* dst_;
    // acc_buf_ used for accumulation among sub Tensor Iterator when
    // accumulation on output is not permissible
    void* acc_buf_;
    // cta_buf_ used for accumulation between blocks during global reduction
    void* cta_buf_;
    int* semaphores_;
    arg_t identity_;
    bool accumulate_;
    bool final_output_;
};

class AccumulationBuffer {
public:
    AccumulationBuffer() {}

    AccumulationBuffer(int64_t acc_t_size,
                       int64_t out_t_size,
                       char* out_ptr,
                       int64_t size) {
        out_ptr_ = (char*)out_ptr;
        if (out_t_size >= acc_t_size) {
            // reusing output buffer for accumulation.
            acc_ptr_ = (char*)out_ptr;
            numerator_ = 1;
            denominator_ = 1;
        } else {
            Device device(Device::DeviceType::CUDA, cuda::GetDevice());
            buffer_ = std::make_unique<Blob>(size, device);
            acc_ptr_ = (char*)buffer_->GetDataPtr();
            numerator_ = acc_t_size;
            denominator_ = out_t_size;
            ReduceFraction(numerator_, denominator_);
        }
    }

    char* GetAccSlice(char* out_ptr) {
        if (numerator_ == -1 || acc_ptr_ == nullptr) {
            return nullptr;
        }
        return acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
    }

private:
    std::unique_ptr<Blob> buffer_;
    char* acc_ptr_ = nullptr;
    char* out_ptr_ = nullptr;
    float size_factor_ = -1;
    int64_t numerator_ = -1;
    int64_t denominator_ = -1;
};

class CUDAReductionEngine {
public:
    CUDAReductionEngine(const CUDAReductionEngine&) = delete;
    CUDAReductionEngine& operator=(const CUDAReductionEngine&) = delete;
    CUDAReductionEngine(const Indexer& indexer) : indexer_(indexer) {}

    template <typename func_t, typename scalar_t>
    void Run(const func_t& reduce_func, scalar_t identity) {
        if (indexer_.NumWorkloads() == 0) {
            utility::LogError(
                    "0-sized input should be handled outside of the reduction "
                    "engine.");
        }
        if (indexer_.NumInputs() != 1) {
            utility::LogError("Reduction op must have exactly one input.");
        }

        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
        using arg0_t = typename BinaryFunctionTraits<func_t>::arg0_t;
        using arg1_t = typename BinaryFunctionTraits<func_t>::arg1_t;
        if (!std::is_same<scalar_t, arg0_t>::value ||
            !std::is_same<scalar_t, arg1_t>::value) {
            utility::LogError(
                    "Function input type must match with the identity's type.");
        }

        using res_t = typename BinaryFunctionTraits<func_t>::res_t;
        if (std::is_same<res_t, bool>::value) {
            // func_t is a comparison function (for arg-reduction).
            // Signature: (scalar_t, scalar_t) -> bool.
            RunReduce<scalar_t, int64_t>(
                    indexer_, WrapArgReduceOps(reduce_func),
                    thrust::pair<scalar_t, int64_t>(identity, 0));
        } else {
            // func_t is a regular reduction function.
            // Signature: (scalar_t, scalar_t) -> scalar_t.
            RunReduce<scalar_t, scalar_t>(
                    indexer_, WrapRegularReduceOps<scalar_t>(reduce_func),
                    identity);
        }
    }

private:
    /// If the index cannot be represented in 32 bits, RunReduce calls itself
    /// recursively.
    template <typename scalar_t,
              typename out_scalar_t,
              int vt0 = 4,
              typename ops_t,
              typename ident_t>
    static void RunReduce(Indexer& indexer,
                          const ops_t& ops,
                          ident_t identity,
                          AccumulationBuffer* acc_buf_ptr = nullptr) {
        using traits = FunctionTraits<decltype(&ops_t::Reduce)>;
        using arg_t = typename traits::template arg<0>::type;
        static constexpr bool can_accumulate_in_output =
                std::is_convertible<arg_t, out_scalar_t>::value;

        bool can_use_32bit_indexing = indexer.CanUse32BitIndexing();
        std::unique_ptr<AccumulationBuffer> owned_buf_ptr;

        // The acc_buf_ptr is a shared pointer. It is create at the first
        // entrance reused by all recursive function calls.
        if (acc_buf_ptr == nullptr) {
            // acc_buf_ptr holds buffer used for accumulation among multiple
            // sub_iter when accumulation in output is not possible.
            if (!can_accumulate_in_output && !can_use_32bit_indexing) {
                int64_t output_memory_size = 1;
                for (int dim = 0; dim < indexer.NumDims(); dim++) {
                    output_memory_size = std::max(
                            output_memory_size,
                            indexer.GetMasterShape()[dim] *
                                    indexer.GetOutput().byte_strides_[dim]);
                }
                owned_buf_ptr.reset(new AccumulationBuffer(
                        sizeof(arg_t), sizeof(out_scalar_t),
                        (char*)indexer.GetOutput().data_ptr_,
                        output_memory_size * sizeof(arg_t)));
            } else {
                owned_buf_ptr.reset(new AccumulationBuffer());
            }
            acc_buf_ptr = owned_buf_ptr.get();
        }

        if (!can_use_32bit_indexing) {
            for (auto& sub_indexer : indexer.SplitTo32BitIndexing()) {
                RunReduce<scalar_t, out_scalar_t, vt0>(sub_indexer, ops,
                                                       identity, acc_buf_ptr);
            }
            return;
        }

        ReduceConfig config(sizeof(arg_t), indexer);

        std::unique_ptr<Blob> buffer_blob;
        std::unique_ptr<Blob> semaphores_blob;
        void* buffer = nullptr;
        void* semaphores = nullptr;
        if (config.ShouldGlobalReduce()) {
            Device device(Device::DeviceType::CUDA, cuda::GetDevice());

            buffer_blob =
                    std::make_unique<Blob>(config.GlobalMemorySize(), device);
            semaphores_blob =
                    std::make_unique<Blob>(config.SemaphoreSize(), device);
            buffer = buffer_blob->GetDataPtr();
            semaphores = semaphores_blob->GetDataPtr();
            OPEN3D_CUDA_CHECK(
                    cudaMemset(semaphores, 0, config.SemaphoreSize()));
        }

        OPEN3D_ASSERT(can_use_32bit_indexing);
        const char* in_data = (char*)indexer.GetInput(0).data_ptr_;
        char* out_data = (char*)indexer.GetOutput().data_ptr_;
        char* acc_data = acc_buf_ptr->GetAccSlice(out_data);
        auto output_calc = MakeOutputCalculator<uint32_t>(indexer);
        auto input_calc = MakeInputCalculator<uint32_t>(indexer);

        auto reduce_op = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0>(
                ops, config, input_calc, output_calc, in_data, out_data,
                acc_data, buffer, (int*)semaphores, identity,
                indexer.ShouldAccumulate(), indexer.IsFinalOutput());

        // Launch reduce kernel
        int shared_memory = config.SharedMemorySize();
        ReduceKernel<ReduceConfig::MAX_NUM_THREADS>
                <<<config.GridDim(), config.BlockDim(), shared_memory,
                   core::cuda::GetStream()>>>(reduce_op);
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

private:
    Indexer indexer_;
};

void ReductionCUDA(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code) {
    if (s_regular_reduce_ops.find(op_code) != s_regular_reduce_ops.end()) {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        CUDAReductionEngine re(indexer);
        Dtype dtype = src.GetDtype();

        CUDAScopedDevice scoped_device(src.GetDevice());
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            switch (op_code) {
                case ReductionOpCode::Sum:
                    if (indexer.NumWorkloads() == 0) {
                        // 0-sized input can be reduced to non-0-sized outputs,
                        // where identity elements should be filled.
                        // E.g. np.sum(np.ones((0, 5)), axis=0).shape == (5,).
                        dst.Fill(0);
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> scalar_t { return a + b; },
                               static_cast<scalar_t>(0));
                    }
                    break;
                case ReductionOpCode::Prod:
                    if (indexer.NumWorkloads() == 0) {
                        dst.Fill(1);
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> scalar_t { return a * b; },
                               static_cast<scalar_t>(1));
                    }
                    break;
                case ReductionOpCode::Min:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Min.");
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> scalar_t { return min(a, b); },
                               static_cast<scalar_t>(
                                       std::numeric_limits<scalar_t>::max()));
                    }
                    break;
                case ReductionOpCode::Max:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Max.");
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> scalar_t { return max(a, b); },
                               static_cast<scalar_t>(std::numeric_limits<
                                                     scalar_t>::lowest()));
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_arg_reduce_ops.find(op_code) != s_arg_reduce_ops.end()) {
        if (dst.GetDtype() != core::Int64) {
            utility::LogError("Arg-reduction must have int64 output dtype.");
        }
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME, dims);
        CUDAReductionEngine re(indexer);
        Dtype dtype = src.GetDtype();

        CUDAScopedDevice scoped_device(src.GetDevice());
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            switch (op_code) {
                case ReductionOpCode::ArgMin:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support ArgMin.");
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> bool { return a < b; },
                               static_cast<scalar_t>(
                                       std::numeric_limits<scalar_t>::max()));
                    }
                    break;
                case ReductionOpCode::ArgMax:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support ArgMax.");
                    } else {
                        re.Run([] OPEN3D_HOST_DEVICE(scalar_t a, scalar_t b)
                                       -> bool { return a > b; },
                               static_cast<scalar_t>(std::numeric_limits<
                                                     scalar_t>::lowest()));
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_boolean_reduce_ops.find(op_code) !=
               s_boolean_reduce_ops.end()) {
        if (src.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean input tensor.");
        }
        if (dst.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean output tensor.");
        }
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        CUDAReductionEngine re(indexer);

        CUDAScopedDevice scoped_device(src.GetDevice());
        switch (op_code) {
            case ReductionOpCode::All:
                if (indexer.NumWorkloads() == 0) {
                    dst.Fill(true);
                } else {
                    re.Run([] OPEN3D_HOST_DEVICE(uint8_t a, uint8_t b)
                                   -> uint8_t { return a && b; },
                           static_cast<uint8_t>(true));
                }
                break;
            case ReductionOpCode::Any:
                if (indexer.NumWorkloads() == 0) {
                    dst.Fill(false);
                } else {
                    re.Run([] OPEN3D_HOST_DEVICE(uint8_t a, uint8_t b)
                                   -> uint8_t { return a || b; },
                           static_cast<uint8_t>(false));
                }
                break;
            default:
                utility::LogError("Unsupported op code.");
                break;
        }
    } else {
        utility::LogError("Unsupported op code.");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
