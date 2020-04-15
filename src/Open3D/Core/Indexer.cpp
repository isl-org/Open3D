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

#include "Open3D/Core/Indexer.h"

namespace open3d {

IndexerIterator Indexer::SplitTo32BitIndexing() const {
    return IndexerIterator(*this);
}

IndexerIterator::IndexerIterator(const Indexer& indexer) : indexer_(indexer) {}

IndexerIterator::Iterator::Iterator(const Indexer& indexer) {
    vec_.emplace_back(new Indexer(indexer));
    vec_.emplace_back(nullptr);
    ++(*this);
}

Indexer& IndexerIterator::Iterator::operator*() const { return *vec_.back(); }

IndexerIterator::Iterator& IndexerIterator::Iterator::operator++() {
    vec_.pop_back();
    while (!vec_.empty() && !vec_.back()->CanUse32BitIndexing()) {
        auto& indexer = *vec_.back();
        vec_.emplace_back(indexer.SplitLargestDim());
    }
    return *this;
}

bool IndexerIterator::Iterator::operator==(const Iterator& other) const {
    return this == &other || (vec_.empty() && other.vec_.empty());
}
bool IndexerIterator::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

IndexerIterator::Iterator IndexerIterator::begin() const {
    return IndexerIterator::Iterator(indexer_);
}

IndexerIterator::Iterator IndexerIterator::end() const {
    return IndexerIterator::Iterator();
}

}  // namespace open3d
