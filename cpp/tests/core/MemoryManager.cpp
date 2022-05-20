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

#include "open3d/core/MemoryManager.h"

#include <map>

#include "open3d/core/Device.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class MemoryManagerPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(MemoryManager,
                         MemoryManagerPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class MemoryManagerPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        MemoryManager,
        MemoryManagerPermuteDevicePairs,
        testing::ValuesIn(MemoryManagerPermuteDevicePairs::TestCases()));

class DummyMemoryManager : public core::DeviceMemoryManager {
public:
    DummyMemoryManager(const core::Device& device,
                       size_t limit = std::numeric_limits<size_t>::max())
        : device_(device), limit_(limit) {}

    virtual ~DummyMemoryManager() {
        if (count_malloc_ != count_free_) {
            utility::LogError("Found memory leaks: {} {} --> {}", count_malloc_,
                              count_free_, count_malloc_ - count_free_);
        }
    }

    void* Malloc(size_t byte_size, const core::Device& device) override {
        if (GetAllocatedSize() + byte_size > limit_) {
            utility::LogError(
                    "This should be caught: Limit {} reached via {} + {} = {}",
                    limit_, GetAllocatedSize(), byte_size,
                    GetAllocatedSize() + byte_size);
            return nullptr;
        }

        void* ptr = (void*)running_address_;
        allocations_.emplace(ptr, byte_size);
        running_address_ += byte_size;
        ++count_malloc_;
        return ptr;
    }

    void Free(void* ptr, const core::Device& device) override {
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            utility::LogError("Untracked pointer {}", fmt::ptr(ptr));
        }
        allocations_.erase(it);
        ++count_free_;
    }

    void Memcpy(void* dst_ptr,
                const core::Device& dst_device,
                const void* src_ptr,
                const core::Device& src_device,
                size_t num_bytes) override {
        utility::LogError("Unimplemented.");
    }

    size_t GetAllocatedSize() const {
        return std::accumulate(allocations_.begin(), allocations_.end(), 0,
                               [](size_t count, auto ptr_byte_size) -> size_t {
                                   return count + ptr_byte_size.second;
                               });
    }

    int64_t GetMallocCount() const { return count_malloc_; }

    int64_t GetFreeCount() const { return count_free_; }

protected:
    int64_t count_malloc_ = 0;
    int64_t count_free_ = 0;
    size_t running_address_ = 80;
    std::map<void*, size_t> allocations_;

    core::Device device_;
    size_t limit_;
};

std::shared_ptr<core::CachedMemoryManager> MakeCachedDeviceMemoryManager(
        const core::Device& device) {
    if (device.GetType() == core::Device::DeviceType::CPU) {
        return std::make_shared<core::CachedMemoryManager>(
                std::make_shared<core::CPUMemoryManager>());
    }
#ifdef BUILD_CUDA_MODULE
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        return std::make_shared<core::CachedMemoryManager>(
                std::make_shared<core::CUDAMemoryManager>());
    }
#endif

    utility::LogError("Unimplemented device: {}", device.ToString());
}

core::Device MakeDummyDevice() { return core::Device("CUDA:9999"); }

TEST_P(MemoryManagerPermuteDevices, MallocFree) {
    core::Device device = GetParam();

    void* ptr = core::MemoryManager::Malloc(10, device);
    core::MemoryManager::Free(ptr, device);
}

TEST_P(MemoryManagerPermuteDevicePairs, Memcpy) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    char dst_vals[6] = "xxxxx";
    char src_vals[6] = "hello";
    size_t num_bytes = strlen(src_vals) + 1;

    void* dst_ptr = core::MemoryManager::Malloc(num_bytes, dst_device);
    void* src_ptr = core::MemoryManager::Malloc(num_bytes, src_device);
    core::MemoryManager::MemcpyFromHost(src_ptr, src_device, (void*)src_vals,
                                        num_bytes);

    core::MemoryManager::Memcpy(dst_ptr, dst_device, src_ptr, src_device,
                                num_bytes);
    core::MemoryManager::MemcpyToHost((void*)dst_vals, dst_ptr, dst_device,
                                      num_bytes);
    ASSERT_STREQ(dst_vals, src_vals);

    core::MemoryManager::Free(dst_ptr, dst_device);
    core::MemoryManager::Free(src_ptr, src_device);
}

void ExpectStatistic(const std::shared_ptr<DummyMemoryManager>& dummy_mm,
                     int64_t malloc_count,
                     int64_t free_count,
                     size_t allocated_size) {
    EXPECT_EQ(dummy_mm->GetMallocCount(), malloc_count);
    EXPECT_EQ(dummy_mm->GetFreeCount(), free_count);
    EXPECT_EQ(dummy_mm->GetAllocatedSize(), allocated_size);
}

TEST(MemoryManagerPermuteDevices, NestedCachedMemoryManager) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    EXPECT_THROW(std::make_shared<core::CachedMemoryManager>(cached_mm),
                 std::runtime_error);
}

TEST(MemoryManagerPermuteDevices, CachedNone) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = (void*)10;
    EXPECT_THROW(cached_mm->Free(ptr, device), std::runtime_error);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);
}

TEST(MemoryManagerPermuteDevices, CachedSingle) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(16, device);
    ExpectStatistic(dummy_mm, 1, 0, 16);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 16);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 1, 1, 0);
}

TEST(MemoryManagerPermuteDevices, CachedMultiple) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(16, device);
    ExpectStatistic(dummy_mm, 1, 0, 16);

    void* ptr2 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    void* ptr3 = cached_mm->Malloc(16, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    void* ptr4 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    cached_mm->Free(ptr4, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    cached_mm->Free(ptr3, device);
    ExpectStatistic(dummy_mm, 2, 0, 48);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 2, 2, 0);
}

TEST(MemoryManagerPermuteDevices, CachedRepeated) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    for (int i = 0; i < 5; ++i) {
        void* ptr = cached_mm->Malloc(16, device);
        ExpectStatistic(dummy_mm, 1, 0, 16);

        cached_mm->Free(ptr, device);
        ExpectStatistic(dummy_mm, 1, 0, 16);
    }

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 1, 1, 0);
}

TEST(MemoryManagerPermuteDevices, CachedSplitMergePrev) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(104, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part1 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part2 = cached_mm->Malloc(72, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part1, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part2, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 1, 1, 0);
}

TEST(MemoryManagerPermuteDevices, CachedSplitMergeNext) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(104, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part1 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part2 = cached_mm->Malloc(72, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part2, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part1, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 1, 1, 0);
}

TEST(MemoryManagerPermuteDevices, CachedSplitMergePrevAndNext) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(104, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part1 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part2 = cached_mm->Malloc(40, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    void* ptr_part3 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part1, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part3, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    cached_mm->Free(ptr_part2, device);
    ExpectStatistic(dummy_mm, 1, 0, 104);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 1, 1, 0);
}

TEST(MemoryManagerPermuteDevices, CachedSmallNewMalloc) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    void* ptr2 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 2, 0, 4128);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 2, 0, 4128);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 2, 2, 0);
}

TEST(MemoryManagerPermuteDevices, CachedLargeAutoReleaseSingle) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    void* ptr2 = cached_mm->Malloc(2048, device);
    ExpectStatistic(dummy_mm, 2, 0, 6144);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 2, 0, 6144);

    void* ptr3 = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 3, 1, 8192);

    cached_mm->Free(ptr3, device);
    ExpectStatistic(dummy_mm, 3, 1, 8192);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 3, 1, 8192);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 3, 3, 0);
}

TEST(MemoryManagerPermuteDevices, CachedLargeAutoReleaseMultiple) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    void* ptr2 = cached_mm->Malloc(2048, device);
    ExpectStatistic(dummy_mm, 2, 0, 6144);

    void* ptr3 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    void* ptr4 = cached_mm->Malloc(6144, device);
    ExpectStatistic(dummy_mm, 4, 2, 6176);

    cached_mm->Free(ptr3, device);
    ExpectStatistic(dummy_mm, 4, 2, 6176);

    cached_mm->Free(ptr4, device);
    ExpectStatistic(dummy_mm, 4, 2, 6176);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 4, 4, 0);
}

TEST(MemoryManagerPermuteDevices, CachedLargeAutoReleaseAll) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    void* ptr2 = cached_mm->Malloc(2048, device);
    ExpectStatistic(dummy_mm, 2, 0, 6144);

    void* ptr3 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr3, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    void* ptr4 = cached_mm->Malloc(8192, device);
    ExpectStatistic(dummy_mm, 4, 3, 8192);

    cached_mm->Free(ptr4, device);
    ExpectStatistic(dummy_mm, 4, 3, 8192);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 4, 4, 0);
}

TEST(MemoryManagerPermuteDevices, CachedTooLarge) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    EXPECT_THROW(cached_mm->Malloc(16384, device), std::runtime_error);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);
}

TEST(MemoryManagerPermuteDevices, CachedTooLargeAutoReleaseAll) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    void* ptr2 = cached_mm->Malloc(2048, device);
    ExpectStatistic(dummy_mm, 2, 0, 6144);

    void* ptr3 = cached_mm->Malloc(32, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr3, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr2, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 3, 0, 6176);

    EXPECT_THROW(cached_mm->Malloc(16384, device), std::runtime_error);
    ExpectStatistic(dummy_mm, 3, 3, 0);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 3, 3, 0);
}

// This must be the last test for core::CachedMemoryManager.
TEST(MemoryManagerPermuteDevices, CachedFreeOnProgramEnd) {
    core::Device device = MakeDummyDevice();
    auto dummy_mm = std::make_shared<DummyMemoryManager>(device, 8192);
    auto cached_mm = std::make_shared<core::CachedMemoryManager>(dummy_mm);

    core::CachedMemoryManager::ReleaseCache(device);
    ExpectStatistic(dummy_mm, 0, 0, 0);

    void* ptr = cached_mm->Malloc(4096, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    cached_mm->Free(ptr, device);
    ExpectStatistic(dummy_mm, 1, 0, 4096);

    // No cache release to test free on program end.
}

}  // namespace tests
}  // namespace open3d
