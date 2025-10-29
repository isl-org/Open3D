// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <optional>

namespace open3d {
namespace utility {

// Use C++17 std::optional instead of custom implementation
template <class T>
using optional = std::optional<T>;

// Use C++17 std::nullopt instead of custom implementation
using std::nullopt;
using nullopt_t = std::nullopt_t;

// Use C++17 std::in_place instead of custom implementation
using std::in_place;
using in_place_t = std::in_place_t;

// Use C++17 std::make_optional instead of custom implementation
using std::make_optional;

// Use C++17 std::bad_optional_access instead of custom implementation
using bad_optional_access = std::bad_optional_access;

}  // namespace utility
}  // namespace open3d

// Provide std::hash specialization for open3d::utility::optional (aliased from std::optional)
// This is not strictly necessary since std::hash<std::optional<T>> already exists,
// but we provide it for compatibility with existing code that may use it.
namespace std {
template <typename T>
struct hash<open3d::utility::optional<T>> {
    typedef typename hash<T>::result_type result_type;
    typedef open3d::utility::optional<T> argument_type;

    constexpr result_type operator()(argument_type const& arg) const {
        return arg ? std::hash<T>{}(*arg) : result_type{};
    }
};
}  // namespace std
