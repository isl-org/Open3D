// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#pragma once

#include "Open3D/Core/CUDAState.cuh"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

#include <algorithm>
#include <deque>
#include <iostream>
#include <mutex>
#include <tuple>
#include <vector>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/tuple.h>

#if __CUDA_ARCH__ >= 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif

constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

constexpr int MAX_DIMS = 25;

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

namespace {

// Fallback, anything with an operator()
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// Pointers to class members that are themselves functors.
// For example, in the following code:
// template <typename func_t>
// struct S {
//     func_t f;
// };
// template <typename func_t>
// S<func_t> make_s(func_t f) {
//     return S<func_t> { .f = f };
// }
//
// auto s = make_s([] (int, float) -> double { /* ... */ });
//
// function_traits<decltype(&s::f)> traits;
template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
    // arity is the number of arguments.
    enum { arity = sizeof...(Args) };

    typedef std::tuple<Args...> ArgsTuple;
    typedef ReturnType result_type;

    template <size_t i>
    struct arg {
        typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

template <typename T>
struct nullary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
};

template <typename T>
struct unary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
    using arg1_t = typename traits::template arg<0>::type;
};

template <typename T>
struct binary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
};

// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
    Value div, mod;

    OPEN3D_HOST_DEVICE DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
    IntDivider() {}  // Dummy constructor for arrays.
    IntDivider(Value d) : divisor(d) {}

    OPEN3D_HOST_DEVICE inline Value div(Value n) const { return n / divisor; }
    OPEN3D_HOST_DEVICE inline Value mod(Value n) const { return n % divisor; }
    OPEN3D_HOST_DEVICE inline DivMod<Value> divmod(Value n) const {
        return DivMod<Value>(n / divisor, n % divisor);
    }

    Value divisor;
};

// Implement fast integer division.
template <>
struct IntDivider<unsigned int> {
    static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

    IntDivider() {}  // Dummy constructor for arrays.

    IntDivider(unsigned int d) : divisor(d) {
        assert(divisor >= 1 && divisor <= INT32_MAX);

        // TODO: gcc/clang has __builtin_clz() but it's not portable.
        for (shift = 0; shift < 32; shift++)
            if ((1U << shift) >= divisor) break;

        uint64_t one = 1;
        uint64_t magic =
                ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        m1 = magic;
        assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
    }

    OPEN3D_HOST_DEVICE inline unsigned int div(unsigned int n) const {
#if defined(__CUDA_ARCH__)
        // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n'
        // and 'm1'.
        unsigned int t = __umulhi(n, m1);
        return (t + n) >> shift;
#else
        // Using uint64_t so that the addition does not overflow.
        uint64_t t = ((uint64_t)n * m1) >> 32;
        return (t + n) >> shift;
#endif
    }

    OPEN3D_HOST_DEVICE inline unsigned int mod(unsigned int n) const {
        return n - div(n) * divisor;
    }

    OPEN3D_HOST_DEVICE inline DivMod<unsigned int> divmod(
            unsigned int n) const {
        unsigned int q = div(n);
        return DivMod<unsigned int>(q, n - q * divisor);
    }

    unsigned int divisor;  // d above.
    unsigned int m1;       // Magic number: m' above.
    unsigned int shift;    // Shift amounts.
};

}  // namespace

// Reduce.cuh /////////////////////////////////////////////////////////////////
namespace open3d {
namespace kernel {

template <typename T, int size>
struct alignas(16) Array {
    T data[size];

    OPEN3D_HOST_DEVICE T operator[](int i) const { return data[i]; }
    OPEN3D_HOST_DEVICE T& operator[](int i) { return data[i]; }

    Array() = default;
    Array(const Array&) = default;
    Array& operator=(const Array&) = default;

    // Fill the array with x.
    OPEN3D_HOST_DEVICE Array(T x) {
        for (int i = 0; i < size; i++) {
            data[i] = x;
        }
    }
};

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
    // The offset for each argument (in bytes). Wrapper around fixed-size array.
    using offset_type = Array<index_t, NARGS>;

    OffsetCalculator(int dims,
                     const int64_t* sizes,
                     const int64_t* const* strides)
        : dims(dims) {
        if (dims > MAX_DIMS) {
            utility::LogError("tensor has too many (>{}) dims", MAX_DIMS);
        }

        for (int i = 0; i < MAX_DIMS; ++i) {
            if (i < dims) {
                sizes_[i] = IntDivider<index_t>(sizes[i]);
            } else {
                sizes_[i] = IntDivider<index_t>(1);
            }
            for (int arg = 0; arg < NARGS; arg++) {
                strides_[i][arg] = i < dims ? strides[arg][i] : 0;
            }
        }
    }

    OPEN3D_HOST_DEVICE offset_type get(index_t linear_idx) const {
        offset_type offsets;
#pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }

#pragma unroll
        for (int dim = 0; dim < MAX_DIMS; ++dim) {
            if (dim == dims) {
                break;
            }
            auto divmod = sizes_[dim].divmod(linear_idx);
            linear_idx = divmod.div;

#pragma unroll
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += divmod.mod * strides_[dim][arg];
            }
        }
        return offsets;
    }

    int dims;
    IntDivider<index_t> sizes_[MAX_DIMS];
    index_t strides_[MAX_DIMS][NARGS];
};

static inline int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// returns floor(log2(n))
static inline int last_pow2(int n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return std::max(1, n - (n >> 1));
}

// returns reduced fraction numerator & denominator
OPEN3D_HOST_DEVICE static void reduce_fraction(int64_t& numerator,
                                               int64_t& denominator) {
    // get GCD of num and denom using Euclid's algorithm.
    // Can replace this with std::gcd if we ever support c++17.
    int64_t a = denominator;
    int64_t b = numerator;
    while (b != 0) {
        a %= b;
        // swap(a,b)
        int64_t tmp = a;
        a = b;
        b = tmp;
    }

    // a is now the GCD
    numerator /= a;
    denominator /= a;
}

struct ReduceConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA = 2;

    static constexpr int MAX_NUM_THREADS = 512;

    ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
        : element_size_bytes(element_size_bytes),
          num_inputs(num_inputs),
          num_outputs(num_outputs) {}

    int element_size_bytes;
    int num_inputs;
    int num_outputs;
    int step_input = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3] = {0, 0, 0};
    int output_mult[2] = {0, 0};

    int block_width;
    int block_height;
    int num_threads;

    void set_block_dimension(int64_t dim0, int64_t dim1) {
        int dim0_pow2 = dim0 < MAX_NUM_THREADS
                                ? static_cast<int>(last_pow2(dim0))
                                : MAX_NUM_THREADS;
        int dim1_pow2 = dim1 < MAX_NUM_THREADS
                                ? static_cast<int>(last_pow2(dim1))
                                : MAX_NUM_THREADS;
        block_width =
                std::min(dim0_pow2, CUDAState::GetInstance()->GetWarpSize());
        block_height = std::min(dim1_pow2, int(MAX_NUM_THREADS / block_width));
        block_width = std::min(dim0_pow2, int(MAX_NUM_THREADS / block_height));
        num_threads = block_width * block_height;
    }

    int split_input(int parallelism) {
        int step = step_input;
        step_input *= parallelism;
        return step;
    }

    int split_output(int parallelism) {
        int step = step_output;
        step_output *= parallelism;
        return step;
    }

    dim3 block() const { return dim3(block_width, block_height); }

    dim3 grid() const {
        return dim3(div_up(num_outputs, step_output), ctas_per_output);
    }

    OPEN3D_HOST_DEVICE bool should_block_x_reduce() const {
        return input_mult[BLOCK_X] != 0;
    }

    OPEN3D_HOST_DEVICE bool should_block_y_reduce() const {
        return input_mult[BLOCK_Y] != 0;
    }

    OPEN3D_HOST_DEVICE bool should_global_reduce() const {
        return input_mult[CTA] != 0;
    }

    OPEN3D_DEVICE bool should_store(int output_idx) const {
        return output_idx < num_outputs &&
               (!should_block_x_reduce() || threadIdx.x == 0) &&
               (!should_block_y_reduce() || threadIdx.y == 0);
    }

    OPEN3D_HOST_DEVICE int input_idx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta2 = blockIdx.y;
        return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] +
                cta2 * input_mult[CTA]);
    }

    OPEN3D_HOST_DEVICE int output_idx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta1 = blockIdx.x;
        return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] +
                cta1 * step_output);
    }

    OPEN3D_DEVICE int shared_memory_offset(int offset) const {
        return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
    }

    OPEN3D_DEVICE int staging_memory_offset(int cta2) const {
        int offset = cta2 + blockIdx.x * gridDim.y;
        if (!should_block_x_reduce()) {
            offset = threadIdx.x + offset * blockDim.x;
        }
        return offset;
    }

    int shared_memory_size() const {
        if (!should_block_y_reduce() &&
            (!should_block_x_reduce() ||
             block_width <= CUDAState::GetInstance()->GetWarpSize())) {
            return 0;
        }
        return element_size_bytes * num_threads;
    }

    int64_t global_memory_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
        if (!should_block_x_reduce()) {
            size *= block().x;
        }
        return size;
    }

    int semaphore_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        return sizeof(int) * grid().x;
    }

    int values_per_thread() const { return div_up(num_inputs, step_input); }
};

static inline std::ostream& operator<<(std::ostream& out, dim3 dim) {
    if (dim.y == 1 && dim.z == 1) {
        out << dim.x;
    } else {
        out << "[" << dim.x << "," << dim.y << "," << dim.z << "]";
    }
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const ReduceConfig& config) {
    out << "ReduceConfig(";
    out << "element_size_bytes=" << config.element_size_bytes << ", ";
    out << "num_inputs=" << config.num_inputs << ", ";
    out << "num_outputs=" << config.num_outputs << ", ";
    out << "step_input=" << config.step_input << ", ";
    out << "step_output=" << config.step_output << ", ";
    out << "ctas_per_output=" << config.ctas_per_output << ", ";
    out << "input_mult=[";
    for (int i = 0; i < 3; i++) {
        if (i != 0) {
            out << ",";
        }
        out << config.input_mult[i];
    }
    out << "], ";
    out << "output_mult=[";
    for (int i = 0; i < 2; i++) {
        if (i != 0) {
            out << ",";
        }
        out << config.output_mult[i];
    }
    out << "], ";
    out << "values_per_thread=" << config.values_per_thread() << ", ";
    out << "block=" << config.block() << ", ";
    out << "grid=" << config.grid() << ", ";
    out << "global_memory_size=" << config.global_memory_size();
    out << ")";
    return out;
}

template <int nt, typename R>
OPEN3D_LAUNCH_BOUNDS_2(nt, 4)
__global__ void reduce_kernel(R reduction) {
    reduction.run();
}

template <typename index_t>
static OffsetCalculator<2, index_t> make_output_calculator(
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
static OffsetCalculator<1, index_t> make_input_calculator(
        const Indexer& indexer) {
    int num_reduction_dims = indexer.NumReductionDims();
    std::array<const int64_t*, 1> strides = {
            indexer.GetInput(0).byte_strides_,
    };
    return OffsetCalculator<1, index_t>(
            num_reduction_dims, indexer.GetMasterShape(), strides.data());
}

template <int vt, typename index_t, typename func_t>
OPEN3D_DEVICE void strided_iterate(func_t f,
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

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
    using arg_t = typename binary_function_traits<func_t>::arg1_t;
    using scalar_t = typename binary_function_traits<func_t>::arg2_t;

    func_t combine;
    static inline OPEN3D_DEVICE out_scalar_t project(arg_t arg) {
        return (out_scalar_t)arg;
    }
    static inline OPEN3D_DEVICE arg_t warp_shfl_down(arg_t arg, int offset) {
        return WARP_SHFL_DOWN(arg, offset);
    }

    func_wrapper_t(const func_t& op) : combine(op) {}

    // wrap a normal reduction that ignores the index
    OPEN3D_DEVICE arg_t reduce(arg_t acc, scalar_t val, int64_t idx) const {
        return combine(acc, val);
    }
};

template <typename scalar_t, typename func_t>
func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t& op) {
    return func_wrapper_t<scalar_t, func_t>{op};
}

template <typename scalar_t,
          typename ops_t,
          typename index_t,
          typename out_scalar_t = scalar_t,
          int vt0 = 4>
struct ReduceOp {
    using traits = function_traits<decltype(&ops_t::reduce)>;
    using arg_t =
            typename std::decay<typename traits::template arg<0>::type>::type;

    using InputCalculator = OffsetCalculator<1, index_t>;
    using OutputCalculator = OffsetCalculator<2, index_t>;

    static constexpr bool can_accumulate_in_output =
            std::is_convertible<arg_t, out_scalar_t>::value &&
            std::is_convertible<out_scalar_t, arg_t>::value;

    static constexpr float acc_buffer_multiplier =
            (float)sizeof(arg_t) / sizeof(out_scalar_t);

    ops_t ops;
    arg_t ident;
    ReduceConfig config;
    InputCalculator input_calc;
    OutputCalculator output_calc;
    const void* src;
    const char* dst;
    // acc_buf used for accumulation among sub Tensor Iterator when accumulation
    // on output is not permissible
    void* acc_buf;
    // cta_buf used for accumulation between blocks during global reduction
    void* cta_buf;
    int* semaphores;
    bool accumulate;
    bool final_output;
    int noutputs;

    ReduceOp(ops_t ops,
             ReduceConfig config,
             InputCalculator input_calc,
             OutputCalculator output_calc,
             const void* src,
             char* dst,
             void* acc_buf,
             void* cta_buf,
             int* semaphores,
             arg_t ident,
             int noutputs)
        : ops(ops),
          config(config),
          input_calc(input_calc),
          output_calc(output_calc),
          src(src),
          dst(dst),
          acc_buf(acc_buf),
          cta_buf(cta_buf),
          semaphores(semaphores),
          ident(ident),
          noutputs(noutputs) {}

    OPEN3D_DEVICE void run() const {
        extern __shared__ char shared_memory[];
        index_t output_idx = config.output_idx();
        index_t input_idx = config.input_idx();
        auto base_offsets = output_calc.get(output_idx);

        arg_t value = ident;
        if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
            auto input_slice = (const char*)src + base_offsets[1];
            value = thread_reduce((const scalar_t*)input_slice);
        }

        if (config.should_block_y_reduce()) {
            value = block_y_reduce(value, shared_memory);
        }
        if (config.should_block_x_reduce()) {
            value = block_x_reduce(value, shared_memory);
        }

        auto out = (out_scalar_t*)((char*)dst + base_offsets[0]);
        arg_t* acc = nullptr;
        if (acc_buf != nullptr) {
            int64_t numerator = (int64_t)sizeof(arg_t);
            int64_t denominator = (int64_t)sizeof(out_scalar_t);
            reduce_fraction(numerator, denominator);
            acc = (arg_t*)((char*)acc_buf +
                           (base_offsets[0] * numerator / denominator));
        }

        if (config.should_global_reduce()) {
            value = global_reduce(value, acc, shared_memory);
        } else if (config.should_store(output_idx)) {
            if (acc == nullptr) {
                if (accumulate) {
                    value = accumulate_in_output<can_accumulate_in_output>(
                            out, value);
                }
                if (final_output) {
                    set_results_to_output(value, base_offsets[0]);
                } else {
                    *out = get_accumulated_output<can_accumulate_in_output>(
                            out, value);
                }
            } else {
                if (accumulate) {
                    value = ops.combine(*acc, value);
                }
                if (final_output) {
                    set_results_to_output(value, base_offsets[0]);
                } else {
                    *acc = value;
                }
            }
        }
    }

    OPEN3D_DEVICE arg_t thread_reduce(const scalar_t* data) const {
        index_t idx = config.input_idx();
        // Multiple accumulators to remove dependency between unrolled loops.
        arg_t value_list[vt0];
#pragma unroll
        for (int i = 0; i < vt0; i++) {
            value_list[i] = ident;
        }
        index_t end = config.num_inputs;
        index_t stride = config.step_input;
        index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);

        // Reducing layers of function calls so compiler could do proper loop
        // unroll that exposes instruction level parallelism.
        while (idx < config.num_inputs) {
            // load input
            Array<scalar_t, vt0> values;
            if (input_calc.dims == 1) {
                strided_iterate<vt0>(
                        [&](index_t i, index_t idx) {
                            values[i] = data[idx * element_stride];
                        },
                        idx, end, stride);
            } else {
                strided_iterate<vt0>(
                        [&](index_t i, index_t idx) {
                            values[i] = data[input_calc.get(idx)[0] /
                                             sizeof(scalar_t)];
                        },
                        idx, end, stride);
            }
            // compute
            strided_iterate<vt0, index_t>(
                    [&](index_t i, index_t idx) {
                        value_list[i] =
                                ops.reduce(value_list[i], values[i], idx);
                    },
                    idx, config.num_inputs, config.step_input);
            // step offset
            idx += config.step_input * vt0;
        }
#pragma unroll
        for (int i = 1; i < vt0; i++) {
            value_list[0] = ops.combine(value_list[0], value_list[i]);
        }
        return value_list[0];
    }

    OPEN3D_DEVICE arg_t block_x_reduce(arg_t value, char* shared_memory) const {
        int dim_x = blockDim.x;
        arg_t* shared = (arg_t*)shared_memory;
        if (dim_x > warpSize) {
            int address_base = threadIdx.x + threadIdx.y * blockDim.x;
            shared[address_base] = value;
            for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                    arg_t other = shared[address_base + offset];
                    value = ops.combine(value, other);
                    shared[address_base] = value;
                }
            }
            dim_x = warpSize;
        }

        __syncthreads();

        for (int offset = 1; offset < dim_x; offset <<= 1) {
            arg_t other = ops.warp_shfl_down(value, offset);
            value = ops.combine(value, other);
        }
        return value;
    }

    OPEN3D_DEVICE arg_t block_y_reduce(arg_t value, char* shared_memory) const {
        arg_t* shared = (arg_t*)shared_memory;
        shared[config.shared_memory_offset(0)] = value;
        for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                arg_t other = shared[config.shared_memory_offset(offset)];
                value = ops.combine(value, other);
                shared[config.shared_memory_offset(0)] = value;
            }
        }
        return value;
    }

    OPEN3D_DEVICE bool mark_block_finished() const {
        __shared__ bool is_last_block_done_shared;

        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
            is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
        }

        __syncthreads();

        return is_last_block_done_shared;
    }

    template <bool can_acc>
    OPEN3D_DEVICE arg_t accumulate_in_output(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<can_acc>::type* = nullptr) const {
        return ops.combine(*out, value);
    }

    template <bool can_acc>
    OPEN3D_DEVICE out_scalar_t get_accumulated_output(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<can_acc>::type* = nullptr) const {
        assert(!final_output);
        return (out_scalar_t)value;
    }

    // This function should never be called --
    // it's the version of `accumulate_in_output`
    // when accumulation in the output is not possible.
    template <bool can_acc>
    OPEN3D_DEVICE arg_t accumulate_in_output(
            out_scalar_t*,
            arg_t,
            typename std::enable_if<!can_acc>::type* = nullptr) const {
        assert(false);  // can't use AT_ASSERT in Cuda.
        return arg_t{};
    }

    // This function should never be called --
    // it's the version of `get_accumulated_output`
    // when accumulation in the output is not possible.
    template <bool can_acc>
    OPEN3D_DEVICE out_scalar_t get_accumulated_output(
            out_scalar_t* out,
            arg_t value,
            typename std::enable_if<!can_acc>::type* = nullptr) const {
        assert(false);
        return *out;
    }

    template <class T>
    OPEN3D_DEVICE void set_results(const T x, const index_t base_offset) const {
        assert(noutputs == 1);
        auto res = (out_scalar_t*)((char*)dst + base_offset);
        *res = x;
    }

    // Currently implemented for max of two outputs
    template <class T>
    OPEN3D_DEVICE void set_results(const thrust::tuple<T, T> x,
                                   const index_t base_offset) const {
        auto res0 = (out_scalar_t*)((char*)dst + base_offset);
        *res0 = thrust::get<0>(x);
    }

    OPEN3D_DEVICE void set_results_to_output(arg_t value,
                                             index_t base_offset) const {
        assert(final_output);
        set_results(ops.project(value), base_offset);
    }

    OPEN3D_DEVICE arg_t global_reduce(arg_t value,
                                      arg_t* acc,
                                      char* shared_memory) const {
        arg_t* reduce_buffer = (arg_t*)cta_buf;
        index_t output_idx = config.output_idx();
        auto base_offsets = output_calc.get(output_idx);
        auto out = (out_scalar_t*)((char*)dst + base_offsets[0]);

        bool should_store = config.should_store(config.output_idx());
        if (should_store) {
            index_t offset = config.staging_memory_offset(blockIdx.y);
            reduce_buffer[offset] = value;
        }

        __threadfence();  // make sure writes are globally visible
        __syncthreads();  // if multiple warps in this block wrote to staging,
                          // make sure they're all done
        bool is_last_block_done = mark_block_finished();

        if (is_last_block_done) {
            value = ident;
            if (config.should_block_x_reduce()) {
                index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
                index_t step = blockDim.x * blockDim.y;
                for (; input_offset < config.ctas_per_output;
                     input_offset += step) {
                    index_t idx = config.staging_memory_offset(input_offset);
                    arg_t next = reduce_buffer[idx];
                    value = ops.combine(value, next);
                }
            } else {
                index_t input_offset = threadIdx.y;
                index_t step = blockDim.y;
                for (; input_offset < config.ctas_per_output;
                     input_offset += step) {
                    index_t idx = config.staging_memory_offset(input_offset);
                    arg_t next = reduce_buffer[idx];
                    value = ops.combine(value, next);
                }
            }
            value = block_y_reduce(value, shared_memory);
            if (config.should_block_x_reduce()) {
                value = block_x_reduce(value, shared_memory);
            }
            if (should_store) {
                if (acc == nullptr) {
                    if (accumulate) {
                        value = accumulate_in_output<can_accumulate_in_output>(
                                out, value);
                    }
                    if (final_output) {
                        set_results_to_output(value, base_offsets[0]);
                    } else {
                        *out = get_accumulated_output<can_accumulate_in_output>(
                                out, value);
                    }
                } else {
                    if (accumulate) {
                        value = ops.combine(*acc, value);
                    }
                    if (final_output) {
                        set_results_to_output(value, base_offsets[0]);
                    } else {
                        *acc = value;
                    }
                }
            }
        }

        return value;
    }
};

template <int nt, typename R>
static void launch_reduce_kernel(const ReduceConfig& config,
                                 const R& reduction) {
    dim3 block = config.block();
    dim3 grid = config.grid();

    // TODO: add stream
    int shared_memory = config.shared_memory_size();
    reduce_kernel<nt, R><<<grid, block, shared_memory>>>(reduction);
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

struct AccumulationBuffer {
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
            int device_id = CUDAState::GetInstance()->GetCurentDeviceID();
            Device device(Device::DeviceType::CUDA, device_id);
            buffer_ = (char*)MemoryManager::Malloc(size, device);
            acc_ptr_ = (char*)buffer_;
            numerator_ = acc_t_size;
            denominator_ = out_t_size;
            reduce_fraction(numerator_, denominator_);
        }
    }

    char* get_acc_slice(char* out_ptr) {
        if (numerator_ == -1 || acc_ptr_ == nullptr) {
            return nullptr;
        }
        return acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
    }

    char* acc_ptr_ = nullptr;
    char* out_ptr_ = nullptr;
    float size_factor_ = -1;
    int64_t numerator_ = -1;
    int64_t denominator_ = -1;
    char* buffer_ = nullptr;
};

template <typename scalar_t, typename out_scalar_t, int vt0 = 4, typename ops_t>
inline void gpu_reduce_kernel(Indexer& indexer,
                              const ops_t& ops,
                              scalar_t ident = 0,
                              AccumulationBuffer* acc_buf_ptr = nullptr) {
    assert(indexer.NumWorkloads() > 0 && indexer.NumInputs() == 1);

    using traits = function_traits<decltype(&ops_t::reduce)>;
    using arg_t = typename traits::template arg<0>::type;
    static constexpr bool can_accumulate_in_output =
            std::is_convertible<arg_t, out_scalar_t>::value;

    bool can_use_32bit_indexing = indexer.CanUse32BitIndexing();
    std::unique_ptr<AccumulationBuffer> owned_buf_ptr;

    // The acc_buf_ptr is a shared pointer. It is create at the first entrance
    // reused by all recursive function calls.
    if (acc_buf_ptr == NULL) {
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
            owned_buf_ptr.reset(
                    new AccumulationBuffer(sizeof(arg_t), sizeof(out_scalar_t),
                                           (char*)indexer.GetOutput().data_ptr_,
                                           output_memory_size * sizeof(arg_t)));
        } else {
            owned_buf_ptr.reset(new AccumulationBuffer());
        }
        acc_buf_ptr = owned_buf_ptr.get();
    }

    if (!can_use_32bit_indexing) {
        for (auto& sub_indexer : indexer.SplitTo32BitIndexing()) {
            gpu_reduce_kernel<scalar_t, out_scalar_t, vt0>(sub_indexer, ops,
                                                           ident, acc_buf_ptr);
        }
        return;
    }

    const char* in_data = (char*)indexer.GetInput(0).data_ptr_;
    char* out_data = (char*)indexer.GetOutput().data_ptr_;
    const int noutputs = 1;
    char* acc_data = acc_buf_ptr->get_acc_slice(out_data);

    // Start by assuming that each thread handles a single output and all
    // the inputs for that output.
    int64_t num_outputs = indexer.NumOutputElements();
    int64_t inputs_per_output = indexer.NumWorkloads() / num_outputs;

    auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

    int64_t dim0;
    int64_t dim1;

    // Adjust block size to map block width to fastest changing dimension of
    // input
    // tensor. This grants the best possible memory accessing pattern, given
    // that for non-contiguous tensor with space in between, we cannot have
    // perfect memory coalescing.
    bool reduction_on_fastest_striding_dimension =
            (indexer.NumReductionDims() == indexer.NumDims()) ||
            (indexer.GetInput(0).byte_strides_[0] <
             indexer.GetInput(0).byte_strides_[indexer.NumReductionDims()]);

    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
        // Map block.x to the fastest reducing dimension. It implies:
        //   1. block_x_reduce is required.
        //   2. block.y now max out to num_outputs.
        dim0 = indexer.GetMasterShape()[0];
        dim1 = num_outputs;
    } else {
        // Map block.x to the fastest non reducing dimension. It implies:
        //   1. block_x_reduce is turned off.
        //   2. block.y now max out to inputs_per_output.
        dim0 = indexer.GetMasterShape()[indexer.NumReductionDims()];
        dim1 = inputs_per_output;
    }

    // Adjust block_width and block_height
    config.set_block_dimension(dim0, dim1);

    int block_width = config.block_width;
    int block_height = config.block_height;

    if (indexer.NumDims() == 0 || reduction_on_fastest_striding_dimension) {
        // Split the input across lanes if the input is contiguous in the
        // reduced dimension. This will require reduction between threads using
        // warp shuffle instructions and shared memory (if block_width >
        // warpSize).
        config.input_mult[0] = config.split_input(block_width);
    } else {
        // Otherwise split the output across lanes in a warp.
        config.output_mult[0] = config.split_output(block_width);
    }

    if (config.values_per_thread() >= block_height * 16 ||
        config.values_per_thread() >= 256) {
        // Divide the input across warps in a thread-block, if that leaves at
        // least 16 elements to be summed by each thread. This will require
        // inter-warp reduction using shared memory.
        config.input_mult[1] = config.split_input(block_height);
    } else {
        // Otherwise, each warp handles a separate output.
        config.output_mult[1] = config.split_output(block_height);
    }

    if (config.input_mult[1] != 0 && config.values_per_thread() >= 256 &&
        num_outputs <= 4096) {
        // Divide the input across thread-blocks if the amount of work
        // per-thread is large enough and the size of the output is small
        // enough. This will require a reduction using global memory.
        config.ctas_per_output = div_up(config.values_per_thread(), 16);
        if (config.ctas_per_output > 65535) {
            config.ctas_per_output = 65535;
        }
        config.input_mult[2] = config.split_input(config.ctas_per_output);
    }

    void* buffer = nullptr;
    void* semaphores = nullptr;
    if (config.should_global_reduce()) {
        int device_id = CUDAState::GetInstance()->GetCurentDeviceID();
        Device device(Device::DeviceType::CUDA, device_id);

        buffer = MemoryManager::Malloc(config.global_memory_size(), device);
        semaphores = MemoryManager::Malloc(config.semaphore_size(), device);
        OPEN3D_CUDA_CHECK(cudaMemset(semaphores, 0, config.semaphore_size()));
    }

    assert(can_use_32bit_indexing);
    auto output_calc = make_output_calculator<uint32_t>(indexer);
    auto input_calc = make_input_calculator<uint32_t>(indexer);

    auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0>(
            ops, config, input_calc, output_calc, in_data, out_data, acc_data,
            buffer, (int*)semaphores, ident, noutputs);
    reduce.accumulate = indexer.ShouldAccumulate();
    reduce.final_output = indexer.IsFinalOutput();

    launch_reduce_kernel<ReduceConfig::MAX_NUM_THREADS>(config, reduce);
}

template <typename scalar_t,
          typename acc_t = scalar_t,
          typename out_t = scalar_t>
inline void sum_kernel_impl(Indexer& indexer) {
    gpu_reduce_kernel<scalar_t, out_t>(
            indexer,
            func_wrapper<out_t>([] OPEN3D_HOST_DEVICE(acc_t a, acc_t b)
                                        -> acc_t { return a + b; }),
            0);
}

template <typename scalar_t,
          typename acc_t = scalar_t,
          typename out_t = scalar_t>
inline void prod_kernel_impl(Indexer& indexer) {
    gpu_reduce_kernel<scalar_t, out_t>(
            indexer,
            func_wrapper<out_t>([] OPEN3D_HOST_DEVICE(acc_t a, acc_t b)
                                        -> acc_t { return a * b; }),
            1);
}

template <typename scalar_t,
          typename acc_t = scalar_t,
          typename out_t = scalar_t>
inline void min_kernel_impl(Indexer& indexer) {
    gpu_reduce_kernel<scalar_t, out_t>(
            indexer,
            func_wrapper<out_t>([] OPEN3D_HOST_DEVICE(acc_t a, acc_t b)
                                        -> acc_t { return a < b ? a : b; }),
            std::numeric_limits<scalar_t>::max());
}

template <typename scalar_t,
          typename acc_t = scalar_t,
          typename out_t = scalar_t>
inline void max_kernel_impl(Indexer& indexer) {
    gpu_reduce_kernel<scalar_t, out_t>(
            indexer,
            func_wrapper<out_t>([] OPEN3D_HOST_DEVICE(acc_t a, acc_t b)
                                        -> acc_t { return a > b ? a : b; }),
            std::numeric_limits<scalar_t>::min());
}

}  // namespace kernel
}  // namespace open3d
