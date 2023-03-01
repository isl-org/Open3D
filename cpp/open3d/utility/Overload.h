// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
