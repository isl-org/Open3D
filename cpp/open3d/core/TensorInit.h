// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <initializer_list>
#include <utility>

#include "open3d/core/SizeVector.h"

namespace open3d {
namespace core {
namespace tensor_init {

// Conventions used in this file:
// T: scalar value type
// D: dimension of type size_t
// L: (nested) initializer_list, or a scalar value (0-d nested)

template <typename T, size_t D>
struct NestedInitializerImpl {
    using type = std::initializer_list<
            typename NestedInitializerImpl<T, D - 1>::type>;
};

template <typename T>
struct NestedInitializerImpl<T, 0> {
    using type = T;
};

template <typename T, size_t D>
using NestedInitializerList = typename NestedInitializerImpl<T, D>::type;

template <typename L>
struct InitializerDim {
    static constexpr size_t value = 0;
};

template <typename L>
struct InitializerDim<std::initializer_list<L>> {
    static constexpr size_t value = 1 + InitializerDim<L>::value;
};

template <size_t D>
struct InitializerShapeImpl {
    template <typename L>
    static constexpr size_t value(const L& list) {
        if (list.size() == 0) {
            return 0;
        }
        size_t dim = InitializerShapeImpl<D - 1>::value(*list.begin());
        for (const auto& value : list) {
            if (dim != InitializerShapeImpl<D - 1>::value(value)) {
                utility::LogError(
                        "Input contains ragged nested sequences"
                        "(nested lists with unequal sizes or shapes).");
            }
        }
        return dim;
    }
};

template <>
struct InitializerShapeImpl<0> {
    template <typename L>
    static constexpr size_t value(const L& list) {
        return list.size();
    }
};

template <typename L, size_t... D>
SizeVector InitializerShape(const L& list, std::index_sequence<D...>) {
    return SizeVector{
            static_cast<int64_t>(InitializerShapeImpl<D>::value(list))...};
}

template <typename L>
SizeVector InferShape(const L& list) {
    SizeVector shape = InitializerShape<L>(
            list, std::make_index_sequence<InitializerDim<L>::value>());
    // Handle 0-dimensional inputs.
    size_t last_dim = 0;
    while (shape.size() > (last_dim + 1) && shape[last_dim] != 0) {
        last_dim++;
    }
    shape.resize(last_dim + 1);
    return shape;
}

template <typename T, typename L>
void NestedCopy(T&& iter, const L& list) {
    *iter++ = list;
}

template <typename T, typename L>
void NestedCopy(T&& iter, const std::initializer_list<L>& list) {
    for (const auto& value : list) {
        NestedCopy(std::forward<T>(iter), value);
    }
}

template <typename T, size_t D>
std::vector<T> ToFlatVector(
        const SizeVector& shape,
        const tensor_init::NestedInitializerList<T, D>& nested_list) {
    std::vector<T> values(shape.NumElements());
    tensor_init::NestedCopy(values.begin(), nested_list);
    return values;
}

}  // namespace tensor_init
}  // namespace core
}  // namespace open3d
