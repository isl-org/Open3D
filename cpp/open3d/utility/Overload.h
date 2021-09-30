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

#pragma once

namespace open3d {
namespace utility {

/// Generic functor for overloading (lambda) functions.
/// See Overload(...) function on how to use it.
///
/// \note In C++17, this could be simplified to:
///
/// \code
/// template <typename... Ts>
/// struct Overloaded : Ts... {
///     using Ts::operator()...;
/// };
/// \endcode
template <typename... Ts>
struct Overloaded;

template <typename T1, typename... Ts>
struct Overloaded<T1, Ts...> : T1, Overloaded<Ts...> {
    Overloaded(T1 t1, Ts... ts) : T1(t1), Overloaded<Ts...>(ts...) {}

    using T1::operator();
    using Overloaded<Ts...>::operator();
};

template <typename T1>
struct Overloaded<T1> : T1 {
    Overloaded(T1 t1) : T1(t1) {}

    using T1::operator();
};

/// Overloads an arbitrary set of (lambda) functions.
///
/// Example:
///
/// \code
/// auto Func = utility::Overload(
///         [&](int i) { utility::LogInfo("Got int {}", i); },
///         [&](float f) { utility::LogInfo("Got float {}", f); });
///
/// Func(1);     // Prints: Got int 1
/// Func(2.4f);  // Prints: Got float 2.4
/// \endcode
template <typename... Ts>
Overloaded<Ts...> Overload(Ts... ts) {
    return Overloaded<Ts...>(ts...);
}

}  // namespace utility
}  // namespace open3d
