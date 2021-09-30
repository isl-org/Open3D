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

#include "benchmarks/benchmark_utilities/Rand.h"

#include <random>
#include <type_traits>
#include <vector>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace benchmarks {

core::Tensor Rand(const core::SizeVector& shape,
                  size_t seed,
                  const std::pair<core::Scalar, core::Scalar>& range,
                  core::Dtype dtype,
                  const core::Device& device) {
    // Initialize on CPU, then copy to device
    core::Tensor random =
            core::Tensor::Empty(shape, dtype, core::Device("CPU:0"));
    core::TensorIterator random_it(random);

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        scalar_t low = range.first.To<scalar_t>();
        scalar_t high = range.second.To<scalar_t>();

        using uniform_distribution = std::conditional_t<
                std::is_same<scalar_t, bool>::value,
                std::uniform_int_distribution<uint16_t>,
                std::conditional_t<
                        std::is_same<scalar_t, uint8_t>::value,
                        std::uniform_int_distribution<uint16_t>,
                        std::conditional_t<
                                std::is_same<scalar_t, int8_t>::value,
                                std::uniform_int_distribution<int16_t>,
                                std::conditional_t<
                                        std::is_integral<scalar_t>::value,
                                        std::uniform_int_distribution<scalar_t>,
                                        std::conditional_t<
                                                std::is_floating_point<
                                                        scalar_t>::value,
                                                std::uniform_real_distribution<
                                                        scalar_t>,
                                                void>>>>>;

        int num_threads = utility::EstimateMaxThreads();
        std::vector<std::default_random_engine> rng;
        for (int64_t i = 0; i < num_threads; ++i) {
            rng.emplace_back(seed + i);
        }
        uniform_distribution dist(low, high);

        core::ParallelFor(core::Device("CPU:0"), num_threads, [&](int64_t i) {
            int64_t start = random.NumElements() * i / num_threads;
            int64_t end = std::min<int64_t>(
                    random.NumElements() * (i + 1) / num_threads,
                    random.NumElements());
            for (int64_t idx = start; idx < end; ++idx) {
                *static_cast<scalar_t*>(random_it.GetPtr(idx)) = dist(rng[i]);
            }
        });
    });

    return random.To(device);
}

}  // namespace benchmarks
}  // namespace open3d
