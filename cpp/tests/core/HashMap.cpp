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

#include "open3d/core/hashmap/HashMap.h"

#include <random>
#include <unordered_map>

#include "open3d/core/Device.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/hashmap/HashSet.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Optional.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

template <typename K, typename V>
class HashData {
public:
    HashData(int count, int slots) {
        keys_.resize(count);
        vals_.resize(count);

        std::vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine(0));

        // Ensure enough duplicates for harder tests
        for (int i = 0; i < count; ++i) {
            int v = indices[i] % slots;
            keys_[i] = v * k_factor_;
            vals_[i] = v;
        }
    }

public:
    const int k_factor_ = 100;
    std::vector<K> keys_;
    std::vector<V> vals_;
};

class HashMapPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(HashMap,
                         HashMapPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(HashMapPermuteDevices, SimpleInit) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    for (auto backend : backends) {
        int n = 5;
        std::vector<int> keys_val = {100, 300, 500, 700, 900};
        std::vector<int> values_val = {1, 3, 5, 7, 9};

        core::Tensor keys(keys_val, {n}, core::Int32, device);
        core::Tensor values(values_val, {n}, core::Int32, device);

        int init_capacity = n * 2;
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);

        EXPECT_TRUE(masks.All().Item<bool>());
        EXPECT_EQ(hashmap.Size(), 5);
    }
}

TEST_P(HashMapPermuteDevices, Find) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;
    // Insert once, find twice
    HashData<int, int> data(n, slots);

    core::Tensor keys(data.keys_, {n}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        hashmap.Find(keys, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), n);

        // Check found results
        core::Tensor valid_indices =
                buf_indices.IndexGet({masks}).To(core::Int64);
        std::vector<core::Tensor> ai({valid_indices});

        core::Tensor buffer_keys = hashmap.GetKeyTensor();
        core::Tensor buffer_values = hashmap.GetValueTensor();

        core::Tensor valid_keys = buffer_keys.IndexGet(ai);
        core::Tensor valid_values = buffer_values.IndexGet(ai);
        EXPECT_TRUE(valid_keys.AllClose(valid_values * data.k_factor_));
    }
}

TEST_P(HashMapPermuteDevices, Insert) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int, int> data(n, slots);
    core::Tensor keys(data.keys_, {n}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        int64_t s = hashmap.Size();
        EXPECT_EQ(s, slots);
        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);

        core::Tensor active_indices = active_buf_indices.To(core::Int64);
        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

        std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
        std::vector<int> active_values_vec = active_values.ToFlatVector<int>();

        // Check matches
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_keys_vec[i],
                      data.k_factor_ * active_values_vec[i]);
        }
        // Check existence
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec[i], i);
        }
    }
}

TEST_P(HashMapPermuteDevices, Erase) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int, int> data_insert(n, slots);

    core::Tensor keys_insert(data_insert.keys_, {n}, core::Int32, device);
    core::Tensor values_insert(data_insert.vals_, {n}, core::Int32, device);

    HashData<int, int> data_erase(n, slots / 2);
    core::Tensor keys_erase(data_erase.keys_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices_insert, masks_insert;
        hashmap.Insert(keys_insert, values_insert, buf_indices_insert,
                       masks_insert);
        EXPECT_EQ(masks_insert.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        core::Tensor masks_erase;
        hashmap.Erase(keys_erase, masks_erase);
        EXPECT_EQ(masks_erase.To(core::Int64).Sum({0}).Item<int64_t>(),
                  slots / 2);

        int64_t s = hashmap.Size();
        EXPECT_EQ(s, slots - slots / 2);
        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);

        core::Tensor active_indices = active_buf_indices.To(core::Int64);
        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

        std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
        std::vector<int> active_values_vec = active_values.ToFlatVector<int>();

        // Check matches
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_keys_vec[i],
                      data_insert.k_factor_ * active_values_vec[i]);
        }
        // Check existence
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec[i], i + slots / 2);
        }
    }
}

TEST_P(HashMapPermuteDevices, Reserve) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int, int> data(n, slots);

    core::Tensor keys(data.keys_, {n}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        hashmap.Reserve(hashmap.GetBucketCount() * 2);
        EXPECT_EQ(hashmap.Size(), slots);

        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);

        core::Tensor active_indices = active_buf_indices.To(core::Int64);
        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

        std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
        std::vector<int> active_values_vec = active_values.ToFlatVector<int>();
        // Check matches
        for (int i = 0; i < slots; ++i) {
            EXPECT_EQ(active_keys_vec[i],
                      data.k_factor_ * active_values_vec[i]);
        }
        // Check existence
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < slots; ++i) {
            EXPECT_EQ(active_values_vec[i], i);
        }
    }
}

TEST_P(HashMapPermuteDevices, Clear) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int, int> data(n, slots);
    core::Tensor keys(data.keys_, {n}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {1}, core::Int32, {1},
                              device, backend);

        // Insert first
        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        int64_t s = hashmap.Size();
        EXPECT_EQ(s, slots);

        // Then clear
        hashmap.Clear();
        s = hashmap.Size();
        EXPECT_EQ(s, 0);

        // Then insert again
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);
        s = hashmap.Size();
        EXPECT_EQ(s, slots);

        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);

        core::Tensor active_indices = active_buf_indices.To(core::Int64);
        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

        std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
        std::vector<int> active_values_vec = active_values.ToFlatVector<int>();

        // Check matches
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_keys_vec[i],
                      data.k_factor_ * active_values_vec[i]);
        }
        // Check existence
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec[i], i);
        }
    }
}

class int3 {
public:
    int3() : x_(0), y_(0), z_(0){};
    int3(int k) : x_(k), y_(k * 2), z_(k * 4){};
    bool operator==(const int3 &other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }
    int x_;
    int y_;
    int z_;
};

TEST_P(HashMapPermuteDevices, InsertComplexKeys) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int3, int> data(n, slots);

    std::vector<int> keys_int3;
    keys_int3.assign(reinterpret_cast<int *>(data.keys_.data()),
                     reinterpret_cast<int *>(data.keys_.data()) + 3 * n);
    core::Tensor keys(keys_int3, {n, 3}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {3}, core::Int32, {1},
                              device, backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        int64_t s = hashmap.Size();
        EXPECT_EQ(s, slots);

        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);
        EXPECT_EQ(s, active_buf_indices.GetShape()[0]);

        core::Tensor active_indices = active_buf_indices.To(core::Int64);

        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

        EXPECT_TRUE(
                active_keys
                        .GetItem({core::TensorKey::Slice(core::None, core::None,
                                                         core::None),
                                  core::TensorKey::Index(0)})
                        .AllClose(active_values.View({s}) * data.k_factor_));
        // Check existence
        std::vector<int> active_values_vec = active_values.ToFlatVector<int>();
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec[i], i);
        }
    }
}

TEST_P(HashMapPermuteDevices, MultivalueInsertion) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int3, int> data(n, slots);

    std::vector<int> keys_int3;
    keys_int3.assign(reinterpret_cast<int *>(data.keys_.data()),
                     reinterpret_cast<int *>(data.keys_.data()) + 3 * n);
    core::Tensor keys(keys_int3, {n, 3}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);
    core::Tensor values_f64 = values.To(core::Float64);

    for (auto backend : backends) {
        core::HashMap hashmap(init_capacity, core::Int32, {3},
                              {core::Int32, core::Float64}, {{1}, {1}}, device,
                              backend);

        core::Tensor buf_indices, masks;
        hashmap.Insert(keys, {values, values_f64}, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        int64_t s = hashmap.Size();
        EXPECT_EQ(s, slots);

        core::Tensor active_buf_indices;
        hashmap.GetActiveIndices(active_buf_indices);
        EXPECT_EQ(s, active_buf_indices.GetShape()[0]);
        core::Tensor active_indices = active_buf_indices.To(core::Int64);

        std::vector<core::Tensor> ai = {active_indices};
        core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
        core::Tensor active_values_0 = hashmap.GetValueTensor(0).IndexGet(ai);
        core::Tensor active_values_1 = hashmap.GetValueTensor(1).IndexGet(ai);

        core::Tensor key_dim0 = active_keys.GetItem(
                {core::TensorKey::Slice(core::None, core::None, core::None),
                 core::TensorKey::Index(0)});

        EXPECT_TRUE(
                key_dim0.AllClose(active_values_0.View({s}) * data.k_factor_));
        EXPECT_TRUE(
                key_dim0.To(core::Float64)
                        .AllClose(active_values_1.View({s}) * data.k_factor_));

        // Check existence
        std::vector<int> active_values_vec =
                active_values_0.ToFlatVector<int>();
        std::sort(active_values_vec.begin(), active_values_vec.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec[i], i);
        }

        std::vector<double> active_values_vec_f64 =
                active_values_1.ToFlatVector<double>();
        std::sort(active_values_vec_f64.begin(), active_values_vec_f64.end());
        for (int i = 0; i < s; ++i) {
            EXPECT_EQ(active_values_vec_f64[i], i);
        }
    }
}

TEST_P(HashMapPermuteDevices, HashSet) {
    core::Device device = GetParam();
    std::vector<core::HashBackendType> backends;
    if (device.IsCUDA()) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;

    // Insert once, find twice
    HashData<int3, int> data(n, slots);

    std::vector<int> keys_int3;
    keys_int3.assign(reinterpret_cast<int *>(data.keys_.data()),
                     reinterpret_cast<int *>(data.keys_.data()) + 3 * n);
    core::Tensor keys(keys_int3, {n, 3}, core::Int32, device);

    for (auto backend : backends) {
        core::HashSet hashset(init_capacity, core::Int32, {3}, device, backend);

        core::Tensor buf_indices, masks;
        hashset.Insert(keys, buf_indices, masks);
        EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

        int64_t s = hashset.Size();
        EXPECT_EQ(s, slots);
    }
}

TEST_P(HashMapPermuteDevices, HashMapIO) {
    const core::Device &device = GetParam();
    const std::string file_name_noext = "hashmap";
    const std::string file_name_ext = "hashmap.npz";

    const int n = 10000;
    const int slots = 1023;
    int init_capacity = n * 2;
    HashData<int3, int> data(n, slots);

    std::vector<int> keys_int3;
    keys_int3.assign(reinterpret_cast<int *>(data.keys_.data()),
                     reinterpret_cast<int *>(data.keys_.data()) + 3 * n);
    core::Tensor keys(keys_int3, {n, 3}, core::Int32, device);
    core::Tensor values(data.vals_, {n}, core::Int32, device);

    core::HashMap hashmap(init_capacity, core::Int32, {3}, core::Int32, {1},
                          device);
    core::Tensor buf_indices, masks;
    hashmap.Insert(keys, values, buf_indices, masks);
    EXPECT_EQ(masks.To(core::Int64).Sum({0}).Item<int64_t>(), slots);

    hashmap.Save(file_name_noext);
    core::HashMap hashmap_loaded = core::HashMap::Load(file_name_ext);
    EXPECT_EQ(hashmap_loaded.Size(), hashmap.Size());

    core::Tensor active_indices;
    hashmap_loaded.GetActiveIndices(active_indices);

    // Check found results
    std::vector<core::Tensor> ai({active_indices.To(core::Int64)});
    core::Tensor valid_keys = hashmap_loaded.GetKeyTensor().IndexGet(ai);
    core::Tensor valid_values = hashmap_loaded.GetValueTensor().IndexGet(ai);
    EXPECT_TRUE(
            valid_keys.T()[0].AllClose(valid_values.T()[0] * data.k_factor_));

    EXPECT_TRUE(utility::filesystem::FileExists(file_name_ext));
    utility::filesystem::RemoveFile(file_name_ext);
}

}  // namespace tests
}  // namespace open3d
