// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace utility {

/// Generic functor for overloading (lambda) functions.
/// See Overload(...) function on how to use it.
///
/// Uses C++17 parameter pack expansion in using declarations.
template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

// C++17 deduction guide
template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

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
    return Overloaded{ts...};  // C++17 deduction guide makes <Ts...> redundant
}

}  // namespace utility
}  // namespace open3d
