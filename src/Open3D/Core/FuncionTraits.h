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

/// This class is from: PyTorch's aten/src/ATen/detail/FunctionTraits.h
/// Fallback, anything with an operator()
template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {};

/// Pointers to class members that are themselves functors.
/// For example, in the following code:
/// template <typename func_t>
/// struct S {
///     func_t f;
/// };
/// template <typename func_t>
/// S<func_t> make_s(func_t f) {
///     return S<func_t> { .f = f };
/// }
///
/// auto s = make_s([] (int, float) -> double { /* ... */ });
///
/// FunctionTraits<decltype(&s::f)> traits;
template <typename ClassType, typename T>
struct FunctionTraits<T ClassType::*> : public FunctionTraits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const>
    : public FunctionTraits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct FunctionTraits<T&> : public FunctionTraits<T> {};
template <typename T>
struct FunctionTraits<T*> : public FunctionTraits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType(Args...)> {
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
struct NullaryFunctionTraits {
    using traits = FunctionTraits<T>;
    using res_t = typename traits::result_type;
};

template <typename T>
struct UnaryFunctionTraits {
    using traits = FunctionTraits<T>;
    using res_t = typename traits::result_type;
    using arg0_t = typename traits::template arg<0>::type;
};

template <typename T>
struct BinaryFunctionTraits {
    using traits = FunctionTraits<T>;
    using res_t = typename traits::result_type;
    using arg0_t = typename traits::template arg<0>::type;
    using arg1_t = typename traits::template arg<1>::type;
};

}  // namespace open3d
