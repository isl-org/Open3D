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

#include "open3d/core/hashmap/Hashmap.h"

#include <random>
#include <unordered_map>

#include "open3d/core/Device.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

template <typename K, typename V>
class HashData {
public:
    HashData(int count, int slots) {
        keys.resize(count);
        vals.resize(count);

        std::vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine(0));

        // Ensure enough duplicates for harder tests
        for (int i = 0; i < count; ++i) {
            int v = indices[i] % slots;
            keys[i] = v * kFactor;
            vals[i] = v;
        }
    }

public:
    const int kFactor = 100;
    std::vector<K> keys;
    std::vector<V> vals;
};

class HashmapPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Hashmap,
                         HashmapPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(HashmapPermuteDevices, Init) {
    core::Device device = GetParam();

    int n = 5;
    std::vector<int> keys_val = {100, 300, 500, 700, 900};
    std::vector<int> values_val = {1, 3, 5, 7, 9};

    core::Tensor keys(keys_val, {5}, core::Dtype::Int32, device);
    core::Tensor values(values_val, {5}, core::Dtype::Int32, device);

    int init_capacity = n * 2;
    core::Hashmap hashmap(init_capacity, core::Dtype::Int32, core::Dtype::Int32,
                          device);

    core::Tensor addrs({n}, core::Dtype::Int32, device);
    core::Tensor masks({n}, core::Dtype::Bool, device);
    hashmap.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                   static_cast<core::addr_t *>(addrs.GetDataPtr()),
                   static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_TRUE(masks.All());
    EXPECT_EQ(hashmap.Size(), 5);
}

TEST_P(HashmapPermuteDevices, Find) {
    core::Device device = GetParam();
    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;
    core::Hashmap hashmap(init_capacity, core::Dtype::Int32, core::Dtype::Int32,
                          device);

    // Insert once, find twice
    HashData<int, int> data(n, slots);

    core::Tensor keys(data.keys, {n}, core::Dtype::Int32, device);
    core::Tensor values(data.vals, {n}, core::Dtype::Int32, device);

    core::Tensor addrs({n}, core::Dtype::Int32, device);
    core::Tensor masks({n}, core::Dtype::Bool, device);

    hashmap.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                   static_cast<core::addr_t *>(addrs.GetDataPtr()),
                   static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.To(core::Dtype::Int64).Sum({0}).Item<int64_t>(), slots);

    hashmap.Find(keys.GetDataPtr(),
                 static_cast<core::addr_t *>(addrs.GetDataPtr()),
                 static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.To(core::Dtype::Int64).Sum({0}).Item<int64_t>(), n);

    // Check found results
    core::Tensor valid_indices = addrs.IndexGet({masks}).To(core::Dtype::Int64);
    std::vector<core::Tensor> ai({valid_indices});

    core::Tensor buffer_keys = core::Hashmap::ReinterpretBufferTensor(
            hashmap.GetKeyTensor(), {hashmap.GetCapacity()},
            core::Dtype::Int32);
    core::Tensor buffer_values = core::Hashmap::ReinterpretBufferTensor(
            hashmap.GetValueTensor(), {hashmap.GetCapacity()},
            core::Dtype::Int32);

    core::Tensor valid_keys = buffer_keys.IndexGet(ai);
    core::Tensor valid_values = buffer_values.IndexGet(ai);
    std::vector<int> valid_keys_vec = valid_keys.ToFlatVector<int>();
    std::vector<int> valid_values_vec = valid_values.ToFlatVector<int>();
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(valid_keys_vec[i], data.kFactor * valid_values_vec[i]);
    }
}

TEST_P(HashmapPermuteDevices, Insert) {
    core::Device device = GetParam();
    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;
    core::Hashmap hashmap(init_capacity, core::Dtype::Int32, core::Dtype::Int32,
                          device);

    // Insert once, find twice
    HashData<int, int> data(n, slots);

    core::Tensor keys(data.keys, {n}, core::Dtype::Int32, device);
    core::Tensor values(data.vals, {n}, core::Dtype::Int32, device);

    core::Tensor addrs({n}, core::Dtype::Int32, device);
    core::Tensor masks({n}, core::Dtype::Bool, device);

    hashmap.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                   static_cast<core::addr_t *>(addrs.GetDataPtr()),
                   static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.To(core::Dtype::Int64).Sum({0}).Item<int64_t>(), slots);

    int64_t s = hashmap.Size();
    EXPECT_EQ(s, slots);
    core::Tensor active_addrs({s}, core::Dtype::Int32, device);
    hashmap.GetActiveIndices(
            static_cast<core::addr_t *>(active_addrs.GetDataPtr()));

    core::Tensor active_indices = active_addrs.To(core::Dtype::Int64);
    std::vector<core::Tensor> ai = {active_indices};
    core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
    core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

    std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
    std::vector<int> active_values_vec = active_values.ToFlatVector<int>();

    // Check matches
    for (int i = 0; i < s; ++i) {
        EXPECT_EQ(active_keys_vec[i], data.kFactor * active_values_vec[i]);
    }
    // Check existence
    std::sort(active_values_vec.begin(), active_values_vec.end());
    for (int i = 0; i < s; ++i) {
        EXPECT_EQ(active_values_vec[i], i);
    }
}

TEST_P(HashmapPermuteDevices, Erase) {
    core::Device device = GetParam();

    int n = 5;
    std::vector<int> keys_val = {100, 300, 500, 700, 900};
    std::vector<int> values_val = {1, 3, 5, 7, 9};

    core::Tensor keys(keys_val, {5}, core::Dtype::Int32, device);
    core::Tensor values(values_val, {5}, core::Dtype::Int32, device);

    int init_capacity = n * 2;
    core::Hashmap hashmap(init_capacity, core::Dtype::Int32, core::Dtype::Int32,
                          device);

    core::Tensor masks({n}, core::Dtype::Bool, device);
    core::Tensor addrs({n}, core::Dtype::Int32, device);
    hashmap.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                   static_cast<core::addr_t *>(addrs.GetDataPtr()),
                   static_cast<bool *>(masks.GetDataPtr()), n);

    std::vector<int> keys_erase_val = {100, 500, 800, 900, 1000};
    core::Tensor keys_erase(keys_erase_val, {5}, core::Dtype::Int32, device);
    hashmap.Erase(keys_erase.GetDataPtr(),
                  static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(hashmap.Size(), 2);
    EXPECT_EQ(masks.ToFlatVector<bool>(),
              std::vector<bool>({true, true, false, true, false}));

    n = hashmap.Size();
    core::Tensor active_addrs({n}, core::Dtype::Int32, device);
    hashmap.GetActiveIndices(
            static_cast<core::addr_t *>(active_addrs.GetDataPtr()));
    core::Tensor active_indices = active_addrs.To(core::Dtype::Int64);
    core::Tensor active_keys =
            hashmap.GetKeyTensor().IndexGet({active_indices});
    core::Tensor active_values =
            hashmap.GetValueTensor().IndexGet({active_indices});

    std::unordered_map<int, int> key_value_all = {
            {300, 3},
            {700, 7},
    };
    for (int64_t i = 0; i < n; ++i) {
        int k = active_keys[i].Item<int>();
        int v = active_values[i].Item<int>();

        auto it = key_value_all.find(k);
        EXPECT_TRUE(it != key_value_all.end());
        EXPECT_EQ(it->first, k);
        EXPECT_EQ(it->second, v);
    }
}

TEST_P(HashmapPermuteDevices, Rehash) {
    core::Device device = GetParam();
    const int n = 1000000;
    const int slots = 1023;
    int init_capacity = n * 2;
    core::Hashmap hashmap(init_capacity, core::Dtype::Int32, core::Dtype::Int32,
                          device);

    // Insert once, find twice
    HashData<int, int> data(n, slots);

    core::Tensor keys(data.keys, {n}, core::Dtype::Int32, device);
    core::Tensor values(data.vals, {n}, core::Dtype::Int32, device);

    core::Tensor addrs({n}, core::Dtype::Int32, device);
    core::Tensor masks({n}, core::Dtype::Bool, device);

    hashmap.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                   static_cast<core::addr_t *>(addrs.GetDataPtr()),
                   static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.To(core::Dtype::Int64).Sum({0}).Item<int64_t>(), slots);

    hashmap.Rehash(hashmap.GetBucketCount() * 2);
    EXPECT_EQ(hashmap.Size(), slots);

    core::Tensor active_addrs({slots}, core::Dtype::Int32, device);
    hashmap.GetActiveIndices(
            static_cast<core::addr_t *>(active_addrs.GetDataPtr()));

    core::Tensor active_indices = active_addrs.To(core::Dtype::Int64);
    std::vector<core::Tensor> ai = {active_indices};
    core::Tensor active_keys = hashmap.GetKeyTensor().IndexGet(ai);
    core::Tensor active_values = hashmap.GetValueTensor().IndexGet(ai);

    std::vector<int> active_keys_vec = active_keys.ToFlatVector<int>();
    std::vector<int> active_values_vec = active_values.ToFlatVector<int>();
    // Check matches
    for (int i = 0; i < slots; ++i) {
        EXPECT_EQ(active_keys_vec[i], data.kFactor * active_values_vec[i]);
    }
    // Check existence
    std::sort(active_values_vec.begin(), active_values_vec.end());
    for (int i = 0; i < slots; ++i) {
        EXPECT_EQ(active_values_vec[i], i);
    }
}

}  // namespace tests
}  // namespace open3d
