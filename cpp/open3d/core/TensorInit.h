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

namespace open3d {
namespace core {

template <class T, std::size_t I>
struct NestedInitializerList {
    using type = std::initializer_list<
            typename NestedInitializerList<T, I - 1>::type>;
};

template <class T>
struct NestedInitializerList<T, 0> {
    using type = T;
};

template <class T, std::size_t I>
using NestedInitializerListT = typename NestedInitializerList<T, I>::type;

template <class T, class S>
void NestedCopy(T&& iter, const S& s) {
    *iter++ = s;
}

template <class T, class S>
void NestedCopy(T&& iter, std::initializer_list<S> s) {
    for (auto it = s.begin(); it != s.end(); ++it) {
        NestedCopy(std::forward<T>(iter), *it);
    }
}

template <class U>
struct InitializerDepthImpl {
    static constexpr std::size_t value = 0;
};

template <class T>
struct InitializerDepthImpl<std::initializer_list<T>> {
    static constexpr std::size_t value = 1 + InitializerDepthImpl<T>::value;
};

template <class U>
struct InitializerDimension {
    static constexpr std::size_t value = InitializerDepthImpl<U>::value;
};

template <std::size_t I>
struct InitializerShapeImpl {
    template <class T>
    static constexpr std::size_t value(T t) {
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
    template <class T>
    static constexpr std::size_t value(T t) {
        return t.size();
    }
};

template <class R, class U, std::size_t... I>
constexpr R InitializerShape(U t, std::index_sequence<I...>) {
    using size_type = typename R::value_type;
    return {size_type(InitializerShapeImpl<I>::value(t))...};
}

template <class R, class T>
constexpr R Shape(T t) {
    return InitializerShape<R, decltype(t)>(
            t, std::make_index_sequence<
                       InitializerDimension<decltype(t)>::value>());
}

}  // namespace core
}  // namespace open3d
