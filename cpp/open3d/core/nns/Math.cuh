#pragma once

// #include <faiss/gpu/utils/ConversionOperators.cuh>
// #include <faiss/gpu/utils/Float16.cuh>

//
// Templated wrappers to express math for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace open3d {
namespace core {

template <typename T>
struct Math {
    typedef T ScalarType;

    static inline __device__ T add(T a, T b) { return a + b; }

    static inline __device__ T sub(T a, T b) { return a - b; }

    static inline __device__ T mul(T a, T b) { return a * b; }

    static inline __device__ T neg(T v) { return -v; }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    // static inline __device__ float reduceAdd(T v) {
    //     return ConvertTo<float>::to(v);
    // }

    static inline __device__ bool lt(T a, T b) { return a < b; }

    static inline __device__ bool gt(T a, T b) { return a > b; }

    static inline __device__ bool eq(T a, T b) { return a == b; }

    static inline __device__ T zero() { return (T)0; }
};

}  // namespace core
}  // namespace open3d