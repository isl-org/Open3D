// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <mutex>
#include <random>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {
namespace random {

/// Set Open3D global random seed.
void Seed(const int seed);

/// Get global singleton random engine.
/// You must also lock the global mutex before calling the engine.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// {
///     // Put the lock and the call to the engine in the same scope.
///     std::lock_guard<std::mutex> lock(*utility::random::GetMutex());
///     std::shuffle(vals.begin(), vals.end(), *utility::random::GetEngine());
/// }
/// ```
std::mt19937* GetEngine();

/// Get global singleton mutex to protect the engine call. Also see
/// random::GetEngine().
std::mutex* GetMutex();

/// Generate a random uint32.
/// This function is globally seeded by utility::random::Seed().
/// This function is automatically protected by the global random mutex.
uint32_t RandUint32();

/// Generate uniformly distributed random integers in [low, high).
/// This class is globally seeded by utility::random::Seed().
/// This class is a wrapper around std::uniform_int_distribution.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// // Globally seed Open3D. This will affect all random functions.
/// utility::random::Seed(0);
///
/// // Generate a random int in [0, 100).
/// utility::random::UniformIntGenerator<int> gen(0, 100);
/// for (size_t i = 0; i < 10; i++) {
///     std::cout << gen() << std::endl;
/// }
/// ```
template <typename T>
class UniformIntGenerator {
public:
    /// Generate uniformly distributed random integer from
    /// [low, low + 1, ... high - 1].
    ///
    /// \param low The lower bound (inclusive).
    /// \param high The upper bound (exclusive). \p high must be > \p low.
    UniformIntGenerator(const T low, const T high) : distribution_(low, high) {
        if (low < 0) {
            utility::LogError("low must be > 0, but got {}.", low);
        }
        if (low >= high) {
            utility::LogError("low must be < high, but got low={} and high={}.",
                              low, high);
        }
    }

    /// Call this to generate a uniformly distributed integer.
    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

protected:
    std::uniform_int_distribution<T> distribution_;
};

/// Generate uniformly distributed floating point values in [low, high).
/// This class is globally seeded by utility::random::Seed().
/// This class is a wrapper around std::uniform_real_distribution.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// // Globally seed Open3D. This will affect all random functions.
/// utility::random::Seed(0);
///
/// // Generate a random double in [0, 1).
/// utility::random::UniformRealGenerator<double> gen(0, 1);
/// for (size_t i = 0; i < 10; i++) {
///    std::cout << gen() << std::endl;
/// }
/// ```
template <typename T>
class UniformRealGenerator {
public:
    /// Generate uniformly distributed floating point values in [low, high).
    ///
    /// \param low The lower bound (inclusive).
    /// \param high The upper bound (exclusive).
    UniformRealGenerator(const T low = 0.0, const T high = 1.0)
        : distribution_(low, high) {
        if (low >= high) {
            utility::LogError("low must be < high, but got low={} and high={}.",
                              low, high);
        }
    }

    /// Call this to generate a uniformly distributed floating point value.
    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

protected:
    std::uniform_real_distribution<T> distribution_;
};

/// Generate normally distributed floating point values with mean and std.
/// This class is globally seeded by utility::random::Seed().
/// This class is a wrapper around std::normal_distribution.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// // Globally seed Open3D. This will affect all random functions.
/// utility::random::Seed(0);
///
/// // Generate a random double with mean 0 and std 1.
/// utility::random::NormalGenerator<double> gen(0, 1);
/// for (size_t i = 0; i < 10; i++) {
///     std::cout << gen() << std::endl;
/// }
/// ```
template <typename T>
class NormalGenerator {
public:
    /// Generate normally distributed floating point value with mean and std.
    ///
    /// \param mean The mean of the distribution.
    /// \param stddev The standard deviation of the distribution.
    NormalGenerator(const T mean = 0.0, const T stddev = 1.0)
        : distribution_(mean, stddev) {
        if (stddev <= 0) {
            utility::LogError("stddev must be > 0, but got {}.", stddev);
        }
    }

    /// Call this to generate a normally distributed floating point value.
    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

protected:
    std::normal_distribution<T> distribution_;
};

/// Generate discretely distributed integer values according to a range of
/// weight values.
/// This class is globally seeded by utility::random::Seed().
/// This class is a wrapper around std::discrete_distribution.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// // Globally seed Open3D. This will affect all random functions.
/// utility::random::Seed(0);
///
/// // Weighted random choice of size_t
/// std::vector<double> weights{1, 2, 3, 4, 5};
/// utility::random::DiscreteGenerator<size_t> gen(weights.cbegin(),
/// weights.cend()); for (size_t i = 0; i < 10; i++) {
///     std::cout << gen() << std::endl;
/// }
/// ```
template <typename T>
class DiscreteGenerator {
public:
    /// Generate discretely distributed integer values according to a range of
    /// weight values.
    /// \param first The iterator or pointer pointing to the first element in
    /// the range of weights.
    /// \param last The iterator or pointer pointing to one past the last
    /// element in the range of weights.
    template <typename InputIt>
    DiscreteGenerator(InputIt first, InputIt last)
        : distribution_(first, last) {
        if (first > last) {
            utility::LogError("first must be <= last.");
        }
    }

    /// Call this to generate a discretely distributed integer value.
    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

protected:
    std::discrete_distribution<T> distribution_;
};

}  // namespace random
}  // namespace utility
}  // namespace open3d
