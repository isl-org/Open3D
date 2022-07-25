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

#include "open3d/utility/Random.h"

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {
namespace random {

/// Global thread-safe singleton instance for random generation.
/// Generates compiler/OS/device independent random numbers.
class GlobalContext {
public:
    GlobalContext(GlobalContext const&) = delete;
    void operator=(GlobalContext const&) = delete;

    /// Returns the singleton instance.
    static GlobalContext& GetInstance() {
        static GlobalContext instance;
        return instance;
    }

    /// Seed the random number generator (globally).
    void Seed(const int seed) {
        seed_ = seed;
        engine_ = std::mt19937(seed_);
    }

    /// This is used by other downstream random generators.
    /// You must also lock the GetMutex() before calling the engine.
    std::mt19937* GetEngine() { return &engine_; }

    /// Get global singleton mutex to protect the engine call.
    std::mutex* GetMutex() { return &mutex_; }

private:
    GlobalContext() {
        // Randomly seed the seed by default.
        std::random_device rd;
        Seed(rd());
    }
    int seed_;
    std::mt19937 engine_;
    std::mutex mutex_;
};

void Seed(const int seed) { GlobalContext::GetInstance().Seed(seed); }

std::mt19937* GetEngine() { return GlobalContext::GetInstance().GetEngine(); }

std::mutex* GetMutex() { return GlobalContext::GetInstance().GetMutex(); }

uint32_t RandUint32() {
    std::lock_guard<std::mutex> lock(*GetMutex());
    return (*GetEngine())();
}

UniformIntGenerator::UniformIntGenerator(const int low, const int high)
    : distribution_(low, high) {
    if (low < 0) {
        utility::LogError("low must be > 0, but got {}.", low);
    }
    if (low >= high) {
        utility::LogError("low must be < high, but got low={} and high={}.",
                          low, high);
    }
}

int UniformIntGenerator::operator()() {
    std::lock_guard<std::mutex> lock(*GetMutex());
    return distribution_(*GetEngine());
}

UniformDoubleGenerator::UniformDoubleGenerator(const double low,
                                               const double high) {
    if (low >= high) {
        utility::LogError("low must be < high, but got low={} and high={}.",
                          low, high);
    }
    distribution_ = std::uniform_real_distribution<double>(low, high);
}

double UniformDoubleGenerator::operator()() {
    std::lock_guard<std::mutex> lock(*GetMutex());
    return distribution_(*GetEngine());
}

}  // namespace random
}  // namespace utility
}  // namespace open3d
