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

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAUtils.h"
#endif

namespace open3d {
namespace core {

// This implementation is insipred by PyTorch's CUDA memory manager.
// Reference: https://git.io/JUqUA

template <typename Block>
struct SizeOrder {
    bool operator()(const std::shared_ptr<Block>& lhs,
                    const std::shared_ptr<Block>& rhs) const {
        if (lhs->byte_size_ != rhs->byte_size_) {
            return lhs->byte_size_ < rhs->byte_size_;
        }
        return lhs->ptr_ < rhs->ptr_;
    }
};

template <typename Block>
struct PointerOrder {
    bool operator()(const std::shared_ptr<Block>& lhs,
                    const std::shared_ptr<Block>& rhs) const {
        if (lhs->ptr_ != rhs->ptr_) {
            return lhs->ptr_ < rhs->ptr_;
        }
        return lhs->byte_size_ < rhs->byte_size_;
    }
};

struct RealBlock;

struct VirtualBlock {
    VirtualBlock(void* ptr,
                 size_t byte_size,
                 const std::weak_ptr<RealBlock>& r_block)
        : ptr_(ptr), byte_size_(byte_size), r_block_(r_block) {}

    void* ptr_ = nullptr;
    size_t byte_size_ = 0;

    std::weak_ptr<RealBlock> r_block_;
};

struct RealBlock {
    RealBlock(void* ptr, size_t byte_size) : ptr_(ptr), byte_size_(byte_size) {}

    void* ptr_ = nullptr;
    size_t byte_size_ = 0;

    std::shared_ptr<MemoryManagerDevice> device_mm_;
    std::set<std::shared_ptr<VirtualBlock>, PointerOrder<VirtualBlock>>
            v_blocks_;
};

class MemoryCache {
public:
    MemoryCache() = default;
    MemoryCache(const MemoryCache&) = delete;
    MemoryCache& operator=(const MemoryCache&) = delete;

    /// Computes an internal byte size that ensures a proper alignment of the
    /// real and virtual blocks.
    static size_t AlignByteSize(size_t byte_size, size_t alignment = 8) {
        return ((byte_size + alignment - 1) / alignment) * alignment;
    }

    /// Allocates memory from the set of free virtual blocks.
    /// Returns nullptr if no suitable block was found.
    void* Malloc(size_t byte_size) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        auto free_block = ExtractFreeBlock(byte_size);

        if (free_block != nullptr) {
            size_t remaining_size = free_block->byte_size_ - byte_size;

            if (remaining_size == 0) {
                // No update of real block required for perfect fit.
                allocated_virtual_blocks_.emplace(free_block->ptr_, free_block);

                return free_block->ptr_;
            } else {
                // Split virtual block.
                auto new_block = std::make_shared<VirtualBlock>(
                        free_block->ptr_, byte_size, free_block->r_block_);
                auto remaining_block = std::make_shared<VirtualBlock>(
                        static_cast<char*>(free_block->ptr_) + byte_size,
                        remaining_size, free_block->r_block_);

                // Update real block.
                auto real_block = free_block->r_block_.lock();
                real_block->v_blocks_.erase(free_block);
                real_block->v_blocks_.insert(new_block);
                real_block->v_blocks_.insert(remaining_block);

                allocated_virtual_blocks_.emplace(new_block->ptr_, new_block);
                free_virtual_blocks_.insert(remaining_block);

                return new_block->ptr_;
            }
        }

        return nullptr;
    }

    /// Frees memory by moving the corresponding block to the set of free
    /// virtual blocks. Consecutive free virtual blocks belonging to the same
    /// real block are merged together.
    void Free(void* ptr) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        auto ptr_it = allocated_virtual_blocks_.find(ptr);

        if (ptr_it == allocated_virtual_blocks_.end()) {
            // Should never reach here
            utility::LogError("Block of {} should have been recorded.",
                              fmt::ptr(ptr));
        }

        auto v_block = ptr_it->second;
        allocated_virtual_blocks_.erase(ptr_it);

        auto r_block = v_block->r_block_.lock();
        auto& v_block_set = r_block->v_blocks_;

        const auto v_block_it = v_block_set.find(v_block);
        if (v_block_it == v_block_set.end()) {
            utility::LogError(
                    "Virtual block ({} @ {} bytes) not recorded in real block "
                    "{} @ {} bytes.",
                    fmt::ptr(v_block->ptr_), v_block->byte_size_,
                    fmt::ptr(r_block->ptr_), r_block->byte_size_);
        }

        auto merged_v_block = v_block;

        // Merge with previous block.
        if (v_block_it != v_block_set.begin()) {
            // Use copy to keep original iterator unchanged.
            auto v_block_it_copy = v_block_it;
            auto v_block_it_prev = --v_block_it_copy;

            auto v_block_prev = *v_block_it_prev;

            if (free_virtual_blocks_.find(v_block_prev) !=
                free_virtual_blocks_.end()) {
                // Update merged block.
                merged_v_block = std::make_shared<VirtualBlock>(
                        v_block_prev->ptr_,
                        v_block_prev->byte_size_ + merged_v_block->byte_size_,
                        r_block);

                // Remove from sets.
                v_block_set.erase(v_block_prev);
                free_virtual_blocks_.erase(v_block_prev);
            }
        }

        // Merge with next block.

        // Use copy to keep original iterator unchanged.
        auto v_block_it_copy = v_block_it;
        auto v_block_it_next = ++v_block_it_copy;

        if (v_block_it_next != v_block_set.end()) {
            auto v_block_next = *v_block_it_next;

            if (free_virtual_blocks_.find(v_block_next) !=
                free_virtual_blocks_.end()) {
                // Update merged block.
                merged_v_block = std::make_shared<VirtualBlock>(
                        merged_v_block->ptr_,
                        merged_v_block->byte_size_ + v_block_next->byte_size_,
                        r_block);

                // Remove from sets.
                v_block_set.erase(v_block_next);
                free_virtual_blocks_.erase(v_block_next);
            }
        }

        v_block_set.erase(v_block);
        v_block_set.insert(merged_v_block);
        free_virtual_blocks_.insert(merged_v_block);
    }

    /// Acquires ownership of the new real allocated blocks.
    void Acquire(void* ptr,
                 size_t byte_size,
                 const std::shared_ptr<MemoryManagerDevice>& device_mm) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        auto r_block = std::make_shared<RealBlock>(ptr, byte_size);
        auto v_block = std::make_shared<VirtualBlock>(ptr, byte_size, r_block);
        r_block->device_mm_ = device_mm;
        r_block->v_blocks_.insert(v_block);

        real_blocks_.insert(r_block);
        allocated_virtual_blocks_.emplace(v_block->ptr_, v_block);
    }

    /// Releases ownership of unused real allocated blocks whose sizes sum up to
    /// the requested byte size.
    /// Strategy:
    ///  - Best single fit: argmin_x { x.byte_size_ >= byte_size }.
    ///  - If not found, use next-best fit and repeat.
    std::vector<std::pair<void*, std::shared_ptr<MemoryManagerDevice>>> Release(
            size_t byte_size) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        // Filter releasable blocks.
        std::set<std::shared_ptr<RealBlock>, SizeOrder<RealBlock>>
                releasable_real_blocks;
        std::copy_if(
                real_blocks_.begin(), real_blocks_.end(),
                std::inserter(releasable_real_blocks,
                              releasable_real_blocks.begin()),
                [this](const auto& r_block) { return IsReleasable(r_block); });

        // Determine greedy "minimal" subset
        std::vector<std::pair<void*, std::shared_ptr<MemoryManagerDevice>>>
                released_pointers;
        size_t released_size = 0;
        while (!releasable_real_blocks.empty() && released_size < byte_size) {
            size_t remaining_size = byte_size - released_size;
            auto query_size =
                    std::make_shared<RealBlock>(nullptr, remaining_size);
            auto it = releasable_real_blocks.lower_bound(query_size);
            if (it == releasable_real_blocks.end()) {
                --it;
            }
            auto r_block = *it;

            real_blocks_.erase(r_block);
            for (const auto& v_block : r_block->v_blocks_) {
                free_virtual_blocks_.erase(v_block);
            }

            releasable_real_blocks.erase(r_block);
            released_pointers.emplace_back(r_block->ptr_, r_block->device_mm_);
            released_size += r_block->byte_size_;
        }

        return released_pointers;
    }

    /// Releases ownership of all unused real allocated blocks.
    std::vector<std::pair<void*, std::shared_ptr<MemoryManagerDevice>>>
    ReleaseAll() {
        return Release(std::numeric_limits<size_t>::max());
    }

    /// Returns the number of allocated real blocks.
    size_t Size() const { return real_blocks_.size(); }

    /// True if the set of allocated real blocks is empty, false otherwise
    bool Empty() const { return Size() == 0; }

private:
    /// Finds and extracts a suitable free block from the cache.
    /// Strategy:
    ///  - Best fit: argmin_x { x.byte_size_ >= byte_size }.
    ///  - Bounded fragmentation: Avoids using huge blocks for tiny sizes.
    std::shared_ptr<VirtualBlock> ExtractFreeBlock(size_t byte_size) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        size_t max_byte_size = static_cast<size_t>(
                kMaxFragmentation * static_cast<double>(byte_size));

        // Consider blocks with size in range
        // [byte_size, max_byte_size].
        auto query_size = std::make_shared<VirtualBlock>(
                nullptr, byte_size, std::weak_ptr<RealBlock>());
        auto it = free_virtual_blocks_.lower_bound(query_size);
        while (it != free_virtual_blocks_.end() &&
               (*it)->byte_size_ <= max_byte_size) {
            auto r_block = (*it)->r_block_.lock();
            if (r_block->byte_size_ <= max_byte_size) {
                auto block = *it;
                free_virtual_blocks_.erase(it);
                return block;
            }
            ++it;
        }

        return nullptr;
    }

    /// Checks if a real block can be released.
    bool IsReleasable(const std::shared_ptr<RealBlock>& r_block) {
        if (r_block->v_blocks_.size() != 1) {
            return false;
        }

        auto v_block = *(r_block->v_blocks_.begin());
        if (r_block->ptr_ != v_block->ptr_ ||
            r_block->byte_size_ != v_block->byte_size_) {
            utility::LogError(
                    "Real block {} @ {} bytes has single "
                    "virtual block {} @ {} bytes",
                    fmt::ptr(r_block->ptr_), r_block->byte_size_,
                    fmt::ptr(v_block->ptr_), v_block->byte_size_);
        }

        return free_virtual_blocks_.find(v_block) != free_virtual_blocks_.end();
    }

    /// Heuristic constant to bound fragmentation.
    const double kMaxFragmentation = 4.0;

    std::set<std::shared_ptr<RealBlock>, SizeOrder<RealBlock>> real_blocks_;

    std::unordered_map<void*, std::shared_ptr<VirtualBlock>>
            allocated_virtual_blocks_;
    std::set<std::shared_ptr<VirtualBlock>, SizeOrder<VirtualBlock>>
            free_virtual_blocks_;

    std::recursive_mutex mutex_;
};

class Cacher {
public:
    static Cacher& GetInstance() {
        // Ensure the static Logger instance is instantiated before the
        // Cacher instance.
        // Since destruction of static instances happens in reverse order,
        // this guarantees that the Logger can be used at any point in time.
        utility::Logger::GetInstance();

#ifdef BUILD_CUDA_MODULE
        // Ensure CUDAState is initialized before Cacher.
        CUDAState::GetInstance();
#endif

        static Cacher instance;
        return instance;
    }

    ~Cacher() {
        for (const auto& cache_pair : device_caches_) {
            // Simulate C++17 structured bindings for better readability.
            const auto& device = cache_pair.first;
            const auto& cache = cache_pair.second;

            Clear(device);

            if (!cache.Empty()) {
                utility::LogError("{} leaking memory blocks on {}",
                                  cache.Size(), device.ToString());
            }
        }
    }

    Cacher(const Cacher&) = delete;
    Cacher& operator=(Cacher&) = delete;

    void* Malloc(size_t byte_size,
                 const Device& device,
                 const std::shared_ptr<MemoryManagerDevice>& device_mm) {
        Init(device);

        size_t internal_byte_size = MemoryCache::AlignByteSize(byte_size);

        // Malloc from cache.
        void* ptr = device_caches_.at(device).Malloc(internal_byte_size);
        if (ptr != nullptr) {
            return ptr;
        }

        // Malloc from real memory manager.
        try {
            ptr = device_mm->Malloc(internal_byte_size, device);
        } catch (const std::runtime_error&) {
        }

        // Free cached memory and try again.
        if (ptr == nullptr) {
            auto old_ptrs =
                    device_caches_.at(device).Release(internal_byte_size);
            for (const auto& old_pair : old_ptrs) {
                // Simulate C++17 structured bindings for better readability.
                const auto& old_ptr = old_pair.first;
                const auto& old_device_mm = old_pair.second;

                old_device_mm->Free(old_ptr, device);
            }

            // Do not catch the error if the allocation still fails.
            ptr = device_mm->Malloc(internal_byte_size, device);
        }

        device_caches_.at(device).Acquire(ptr, internal_byte_size, device_mm);

        return ptr;
    }

    void Free(void* ptr, const Device& device) {
        Init(device);

        device_caches_.at(device).Free(ptr);
    }

    void Clear(const Device& device) {
        Init(device);

        auto old_ptrs = device_caches_.at(device).ReleaseAll();
        for (const auto& old_pair : old_ptrs) {
            // Simulate C++17 structured bindings for better readability.
            const auto& old_ptr = old_pair.first;
            const auto& old_device_mm = old_pair.second;

            old_device_mm->Free(old_ptr, device);
        }
    }

    void Clear() {
        // Collect all devices in a thread-safe manner. This avoids potential
        // issues with newly initialized/inserted elements while iterating over
        // the container.
        std::vector<Device> devices;
        {
            std::lock_guard<std::recursive_mutex> lock(init_mutex_);
            for (const auto& cache_pair : device_caches_) {
                devices.push_back(cache_pair.first);
            }
        }

        for (const auto& device : devices) {
            Clear(device);
        }
    }

private:
    Cacher() = default;

    /// Resolves race conditions and avoids locking in the operations.
    /// Must be called at the beginning of all operations.
    void Init(const Device& device) {
        std::lock_guard<std::recursive_mutex> lock(init_mutex_);

        // Performs no action if already initialized.
        device_caches_.emplace(std::piecewise_construct,
                               std::forward_as_tuple(device),
                               std::forward_as_tuple());
    }

    std::unordered_map<Device, MemoryCache> device_caches_;
    std::recursive_mutex init_mutex_;
};

MemoryManagerCached::MemoryManagerCached(
        const std::shared_ptr<MemoryManagerDevice>& device_mm)
    : device_mm_(device_mm) {
    if (std::dynamic_pointer_cast<MemoryManagerCached>(device_mm_) != nullptr) {
        utility::LogError(
                "An instance of type MemoryManagerCached as the underlying "
                "non-cached manager is forbidden.");
    }
}

void* MemoryManagerCached::Malloc(size_t byte_size, const Device& device) {
    if (byte_size == 0) {
        return nullptr;
    }

    return Cacher::GetInstance().Malloc(byte_size, device, device_mm_);
}

void MemoryManagerCached::Free(void* ptr, const Device& device) {
    if (ptr == nullptr) {
        return;
    }

    Cacher::GetInstance().Free(ptr, device);
}

void MemoryManagerCached::Memcpy(void* dst_ptr,
                                 const Device& dst_device,
                                 const void* src_ptr,
                                 const Device& src_device,
                                 size_t num_bytes) {
    device_mm_->Memcpy(dst_ptr, dst_device, src_ptr, src_device, num_bytes);
}

void MemoryManagerCached::ReleaseCache(const Device& device) {
    Cacher::GetInstance().Clear(device);
}

void MemoryManagerCached::ReleaseCache() { Cacher::GetInstance().Clear(); }

}  // namespace core
}  // namespace open3d
