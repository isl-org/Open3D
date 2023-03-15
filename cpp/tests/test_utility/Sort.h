// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <vector>

#include "open3d/utility/Logging.h"
#include "tests/test_utility/Compare.h"

namespace open3d {
namespace tests {

/// \brief Functor class for comparing Eigen matrices of the same size.
template <class EigenMatrix>
struct EigenLess {
    bool operator()(const EigenMatrix& lhs, const EigenMatrix& rhs) {
        if (lhs.size() != rhs.size()) {
            utility::LogError("Eigen matrices have different sizes {} != {}",
                              lhs.size(), rhs.size());
        }
        for (int i = 0; i < lhs.size(); i++) {
            if (lhs(i) == rhs(i)) {
                continue;
            } else {
                return lhs(i) < rhs(i);
            }
        }
        return false;
    }
};

/// \brief Apply indices to a vector of matrices
///
/// \param vals A vector of T.
/// \param indices The sorting indices.
/// \return A vector of T, s.t. out_vals[i] = vals[indices[i]].
template <class T>
std::vector<T> ApplyIndices(const std::vector<T>& vals,
                            const std::vector<size_t>& indices) {
    std::vector<T> vals_sorted;
    for (const size_t& i : indices) {
        vals_sorted.push_back(vals[i]);
    }
    return vals_sorted;
};

/// \brief Returns ascending sorted Eigen matrices and the sorting indices.
///
/// \param vals A vector of Eigen matrices.
/// \return A pair of sorted_vals and indices, s.t. sorted_val[i] =
/// vals[indices[i]].
template <class EigenMatrix>
std::pair<std::vector<EigenMatrix>, std::vector<size_t>> SortWithIndices(
        const std::vector<EigenMatrix>& vals) {
    // Sort with indices: https://stackoverflow.com/a/12399290/1255535
    std::vector<size_t> indices(vals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&vals](size_t lhs, size_t rhs) -> bool {
                         return EigenLess<EigenMatrix>()(vals[lhs], vals[rhs]);
                     });
    return std::make_pair(ApplyIndices(vals, indices), indices);
};

/// \brief Returns ascending sorted Eigen matrices.
///
/// \param vals Sorted Eigen matrices.
template <class T, int M, int N, int A>
std::vector<Eigen::Matrix<T, M, N, A>> Sort(
        const std::vector<Eigen::Matrix<T, M, N, A>>& vals) {
    return SortWithIndices(vals).first;
};

/// \brief Returns indices that can transform array A to B, s.t. B[i] ~=
/// A[indices[i]].
///
/// This assumes the sorted \p a_vals and the sorted \p b_vals are close or
/// equal.
///
/// \param a Values of array A.
/// \param b Values of array B.
/// \return indices such that B[i] ~= A[indices[i]]
template <class T, int M, int N, int A>
std::vector<size_t> GetIndicesAToB(
        const std::vector<Eigen::Matrix<T, M, N, A>>& a,
        const std::vector<Eigen::Matrix<T, M, N, A>>& b,
        double threshold = 1e-6) {
    if (a.size() != b.size()) {
        utility::LogError("a.size() != b.size(): {} != {}", a.size(), b.size());
    }
    size_t size = a.size();

    std::vector<Eigen::Matrix<T, M, N, A>> a_sorted;
    std::vector<size_t> indices_a_to_sorted;
    std::tie(a_sorted, indices_a_to_sorted) = SortWithIndices(a);
    std::vector<Eigen::Matrix<T, M, N, A>> b_sorted;
    std::vector<size_t> indices_b_to_sorted;
    std::tie(b_sorted, indices_b_to_sorted) = SortWithIndices(b);
    ExpectEQ(a_sorted, b_sorted, threshold);

    std::vector<size_t> indices_sorted_to_b(size);
    for (size_t i = 0; i < size; ++i) {
        indices_sorted_to_b[indices_b_to_sorted[i]] = i;
    }
    std::vector<size_t> indices_a_to_b(size);
    for (size_t i = 0; i < size; i++) {
        indices_a_to_b[i] = indices_a_to_sorted[indices_sorted_to_b[i]];
    }
    return indices_a_to_b;
};

}  // namespace tests
}  // namespace open3d
