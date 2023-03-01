// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace open3d {
namespace core {
namespace nns {

/// A class for managing memory segments within a memory allocation.
class MemoryAllocation {
public:
    /// Creates a MemoryAllocation object that manages memory segments within
    /// a memory allocation.
    ///
    /// \param ptr          Pointer to the beginning of the memory.
    /// \param size         Size of the memory.
    /// \param alignment    The alignment for returned segments.
    ///
    MemoryAllocation(void* ptr, size_t size, size_t alignment = 1)
        : _ptr(ptr),
          _size(size),
          _alignment(alignment),
          _max_size_ptr((char*)ptr) {
        // align start and end of memory segment
        void* aligned_ptr = std::align(_alignment, 1, ptr, size);
        size_t size_after_align =
                (((char*)ptr + size) - (char*)aligned_ptr) / _alignment;
        size_after_align *= _alignment;
        _free_segments.push_back(
                std::pair<void*, size_t>(aligned_ptr, size_after_align));
    }

    /// Returns a memory segment with size for type T.
    /// Returns the pointer and the size in number of elements T.
    /// May return the pair (nullptr,0) if the allocation is not possible.
    template <class T>
    std::pair<T*, size_t> Alloc(size_t size) {
        std::pair<void*, size_t> tmp = Alloc(size * sizeof(T));
        return std::pair<T*, size_t>((T*)tmp.first, tmp.first ? size : 0);
    }

    /// Returns a memory segment with size in bytes.
    /// May return the pair (nullptr,0) if the allocation is not possible.
    std::pair<void*, size_t> Alloc(size_t size) {
        // round up to alignment
        if (size % _alignment) size += _alignment - size % _alignment;

        for (size_t i = 0; i < _free_segments.size(); ++i) {
            void* ptr = std::align(_alignment, size, _free_segments[i].first,
                                   _free_segments[i].second);
            if (ptr) {
                char* end_ptr = (char*)ptr + size;
                if (end_ptr > _max_size_ptr) _max_size_ptr = end_ptr;

                _free_segments[i].first = end_ptr;
                _free_segments[i].second -= size;
                return std::pair<void*, size_t>(ptr, size);
            }
        }
        return std::pair<void*, size_t>(nullptr, 0);
    }

    /// Returns the largest free segment.
    std::pair<void*, size_t> AllocLargestSegment() {
        size_t size = 0;
        for (const auto& s : _free_segments)
            if (s.second > size) size = s.second;

        return Alloc(size);
    }

    /// Frees a previously returned segment.
    template <class T>
    void Free(const std::pair<T*, size_t>& segment) {
        size_t size = sizeof(T) * segment.second;
        if (size % _alignment) size += _alignment - size % _alignment;

        Free(std::pair<void*, size_t>(segment.first, size));
    }

    /// Frees a previously returned segment.
    void Free(const std::pair<void*, size_t>& segment) {
        if (DEBUG) {
            if ((char*)segment.first < (char*)_ptr ||
                (char*)segment.first + segment.second > (char*)_ptr + _size)
                throw std::runtime_error("free(): segment is out of bounds");
        }
        {
            size_t i;
            for (i = 0; i < _free_segments.size(); ++i) {
                if ((char*)segment.first < (char*)_free_segments[i].first)
                    break;
            }
            _free_segments.insert(_free_segments.begin() + i, segment);
        }

        // merge adjacent segments
        auto seg = _free_segments[0];
        char* end_ptr = (char*)seg.first + seg.second;
        size_t count = 0;
        for (size_t i = 1; i < _free_segments.size(); ++i) {
            const auto& seg_i = _free_segments[i];

            if (end_ptr == (char*)seg_i.first) {
                // merge with adjacent following segment
                seg.second += seg_i.second;
                end_ptr = (char*)seg.first + seg.second;
            } else {
                _free_segments[count] = seg;
                seg = _free_segments[i];
                end_ptr = (char*)seg.first + seg.second;
                ++count;
            }
        }
        _free_segments[count] = seg;
        ++count;
        _free_segments.resize(count);

        if (DEBUG) {
            // check if there are overlapping segments
            for (size_t i = 1; i < _free_segments.size(); ++i) {
                char* prev_end_ptr = (char*)_free_segments[i - 1].first +
                                     _free_segments[i - 1].second;
                if (prev_end_ptr > (char*)_free_segments[i].first) {
                    throw std::runtime_error(
                            "free(): Overlapping free segments found after "
                            "call to free");
                }
            }
        }
    }

    /// Returns the peak memory usage in bytes.
    size_t MaxUsed() const { return _max_size_ptr - (char*)_ptr; }

    /// Returns the alignment in bytes.
    size_t Alignment() const { return _alignment; }

    /// Returns the list of free segments.
    const std::vector<std::pair<void*, size_t>>& FreeSegments() const {
        return _free_segments;
    }

    /// Prints the segment. Meant for debugging.
    template <class T>
    static void PrintSegment(const std::pair<T*, size_t>& s) {
        std::cerr << "ptr " << (void*)s.first << "\t size " << s.second
                  << "\t end " << (void*)((char*)s.first + s.second) << "\n";
    }

    /// Prints all free segments. Meant for debugging.
    void PrintFreeSegments() const {
        for (const auto& s : _free_segments) PrintSegment(s);
    }

private:
    enum internal_config { DEBUG = 0 };  /// for internal debugging use

    /// Pointer to the beginning of the memory
    const void* _ptr;

    /// Size of the memory
    const size_t _size;

    /// The alignment for returned segments
    const size_t _alignment;

    /// Tracks the largest end ptr of all allocated memory segments.
    char* _max_size_ptr;

    /// List of free segments with begin ptr and size. May contain segments
    /// with zero size.
    std::vector<std::pair<void*, size_t>> _free_segments;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
