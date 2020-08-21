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

#include <unordered_map>

#include "open3d/core/Device.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/hashmap/HashmapBase.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

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

    int max_buckets = n * 2;
    std::shared_ptr<core::DefaultHashmap> hashmap = core::CreateDefaultHashmap(
            max_buckets, sizeof(int), sizeof(int), device);

    core::Tensor masks({n}, core::Dtype::Bool, device);
    iterator_t *iterators = static_cast<iterator_t *>(
            core::MemoryManager::Malloc(sizeof(iterator_t) * n, device));
    hashmap->Insert(keys.GetDataPtr(), values.GetDataPtr(), iterators,
                    static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.All(), true);

    core::MemoryManager::Free(iterators, device);
}

TEST_P(HashmapPermuteDevices, Find) {
    core::Device device = GetParam();

    int n = 5;
    std::vector<int> keys_val = {100, 300, 500, 700, 900};
    std::vector<int> values_val = {1, 3, 5, 7, 9};

    core::Tensor keys(keys_val, {5}, core::Dtype::Int32, device);
    core::Tensor values(values_val, {5}, core::Dtype::Int32, device);

    int max_buckets = n * 2;
    std::shared_ptr<core::DefaultHashmap> hashmap = core::CreateDefaultHashmap(
            max_buckets, sizeof(int), sizeof(int), device);

    core::Tensor masks({n}, core::Dtype::Bool, device);
    iterator_t *iterators = static_cast<iterator_t *>(
            core::MemoryManager::Malloc(sizeof(iterator_t) * n, device));
    hashmap->Insert(keys.GetDataPtr(), values.GetDataPtr(), iterators,
                    static_cast<bool *>(masks.GetDataPtr()), n);
    hashmap->Find(keys.GetDataPtr(), iterators,
                  static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks.All(), true);

    std::vector<int> keys_query_val = {100, 500, 800, 900, 1000};
    core::Tensor keys_query(keys_query_val, {5}, core::Dtype::Int32, device);
    hashmap->Find(keys_query.GetDataPtr(), iterators,
                  static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks[0].Item<bool>(), true);
    EXPECT_EQ(masks[1].Item<bool>(), true);
    EXPECT_EQ(masks[2].Item<bool>(), false);
    EXPECT_EQ(masks[3].Item<bool>(), true);
    EXPECT_EQ(masks[4].Item<bool>(), false);

    core::MemoryManager::Free(iterators, device);
}

TEST_P(HashmapPermuteDevices, Insert) {
    core::Device device = GetParam();

    int n = 5;
    std::vector<int> keys_val = {100, 300, 500, 700, 900};
    std::vector<int> values_val = {1, 3, 5, 7, 9};

    core::Tensor keys(keys_val, {5}, core::Dtype::Int32, device);
    core::Tensor values(values_val, {5}, core::Dtype::Int32, device);

    int max_buckets = n * 2;
    std::shared_ptr<core::DefaultHashmap> hashmap = core::CreateDefaultHashmap(
            max_buckets, sizeof(int), sizeof(int), device);

    core::Tensor masks({n}, core::Dtype::Bool, device);
    iterator_t *iterators = static_cast<iterator_t *>(
            core::MemoryManager::Malloc(sizeof(iterator_t) * n, device));
    hashmap->Insert(keys.GetDataPtr(), values.GetDataPtr(), iterators,
                    static_cast<bool *>(masks.GetDataPtr()), n);

    std::vector<int> keys_insert_val = {100, 500, 800, 900, 1000};
    std::vector<int> values_insert_val = {1, 5, 8, 9, 10};
    core::Tensor keys_insert(keys_insert_val, {5}, core::Dtype::Int32, device);
    core::Tensor values_insert(values_insert_val, {5}, core::Dtype::Int32,
                               device);
    hashmap->Insert(keys_insert.GetDataPtr(), values_insert.GetDataPtr(),
                    iterators, static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks[0].Item<bool>(), false);
    EXPECT_EQ(masks[1].Item<bool>(), false);
    EXPECT_EQ(masks[2].Item<bool>(), true);
    EXPECT_EQ(masks[3].Item<bool>(), false);
    EXPECT_EQ(masks[4].Item<bool>(), true);

    core::MemoryManager::Free(iterators, device);
}

TEST_P(HashmapPermuteDevices, Erase) {
    core::Device device = GetParam();

    int n = 5;
    std::vector<int> keys_val = {100, 300, 500, 700, 900};
    std::vector<int> values_val = {1, 3, 5, 7, 9};

    core::Tensor keys(keys_val, {5}, core::Dtype::Int32, device);
    core::Tensor values(values_val, {5}, core::Dtype::Int32, device);

    int max_buckets = n * 2;
    std::shared_ptr<core::DefaultHashmap> hashmap = core::CreateDefaultHashmap(
            max_buckets, sizeof(int), sizeof(int), device);

    core::Tensor masks({n}, core::Dtype::Bool, device);
    iterator_t *iterators = static_cast<iterator_t *>(
            core::MemoryManager::Malloc(sizeof(iterator_t) * n, device));
    hashmap->Insert(keys.GetDataPtr(), values.GetDataPtr(), iterators,
                    static_cast<bool *>(masks.GetDataPtr()), n);

    std::vector<int> keys_erase_val = {100, 500, 800, 900, 1000};
    core::Tensor keys_erase(keys_erase_val, {5}, core::Dtype::Int32, device);
    hashmap->Erase(keys_erase.GetDataPtr(),
                   static_cast<bool *>(masks.GetDataPtr()), n);
    EXPECT_EQ(masks[0].Item<bool>(), true);
    EXPECT_EQ(masks[1].Item<bool>(), true);
    EXPECT_EQ(masks[2].Item<bool>(), false);
    EXPECT_EQ(masks[3].Item<bool>(), true);
    EXPECT_EQ(masks[4].Item<bool>(), false);

    core::MemoryManager::Free(iterators, device);
}

}  // namespace tests
}  // namespace open3d
