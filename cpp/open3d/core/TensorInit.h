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

#include <initializer_list>
#include <utility>

namespace open3d {
namespace core {
namespace tensor_init {

template <typename T, size_t I>
struct NestedInitializerList {
    using type = std::initializer_list<
            typename NestedInitializerList<T, I - 1>::type>;
};

template <typename T>
struct NestedInitializerList<T, 0> {
    using type = T;
};

template <typename T, size_t I>
using NestedInitializerListT = typename NestedInitializerList<T, I>::type;

template <typename T, typename S>
void NestedCopy(T&& iter, const S& s) {
    *iter++ = s;
}

template <typename T, typename S>
void NestedCopy(T&& iter, std::initializer_list<S> s) {
    for (auto it = s.begin(); it != s.end(); ++it) {
        NestedCopy(std::forward<T>(iter), *it);
    }
}

template <typename U>
struct InitializerDepthImpl {
    static constexpr size_t value = 0;
};

template <typename T>
struct InitializerDepthImpl<std::initializer_list<T>> {
    static constexpr size_t value = 1 + InitializerDepthImpl<T>::value;
};

template <typename U>
struct InitializerDimension {
    static constexpr size_t value = InitializerDepthImpl<U>::value;
};

template <size_t I>
struct InitializerShapeImpl {
    template <typename T>
    static constexpr size_t value(T t) {
        if (t.size() == 0) {
            return 0;
        }
        size_t dim = InitializerShapeImpl<I - 1>::value(*t.begin());
        for (auto it = t.begin(); it != t.end(); ++it) {
            if (dim != InitializerShapeImpl<I - 1>::value(*it)) {
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
    template <typename T>
    static constexpr size_t value(T t) {
        return t.size();
    }
};

template <typename R, typename U, size_t... I>
constexpr R InitializerShape(U t, std::index_sequence<I...>) {
    using size_type = typename R::value_type;
    return {size_type(InitializerShapeImpl<I>::value(t))...};
}

template <typename R, typename T>
constexpr R Shape(T t) {
    return InitializerShape<R, decltype(t)>(
            t, std::make_index_sequence<
                       InitializerDimension<decltype(t)>::value>());
}

}  // namespace tensor_init
}  // namespace core
}  // namespace open3d
