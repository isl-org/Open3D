/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include "cutlass_unit_test.h"
#include <algorithm>
#include "tools/test/unit/core/layout_verification.h"


namespace test {

Layout::Layout() {

}

Layout::Layout(Layout::SpanVector const &_layout) {
    reset(_layout);
}

struct SpanCompareDim {
    bool operator()(Layout::Span const &a, Layout::Span const &b) const {
        return a.dim < b.dim;
    }
};

/// Updates the layout
void Layout::reset(Layout::SpanVector const &_layout) {
    layout_ = _layout;

    extent_.clear();
    extent_.resize(layout_.size(), 1);

    int _rank = std::max_element(layout_.begin(), layout_.end(), SpanCompareDim())->dim + 1;

    dim_extent_.clear();
    dim_extent_.resize(_rank, extent_);

    // initialize extent vector
    for (size_t i = layout_.size(); i > 0; --i) {
        extent_.at(i - 1) = layout_.at(i - 1).size * (i < layout_.size() ? extent_.at(i) : 1);
    }

    // initialize the dim_extent vector
    for (size_t rank_idx = 0; rank_idx < dim_extent_.size(); ++rank_idx) {
        ExtentVector &_extent = dim_extent_.at(rank_idx);
        for (size_t i = layout_.size(); i > 0; --i) {
            int _size = (rank_idx == layout_.at(i - 1).dim ? layout_.at(i - 1).size : 1);
            _extent.at(i - 1) = _size * (i < layout_.size() ? _extent.at(i) : 1);
        }
    }
}

/// Computes the rank of the layout
int Layout::rank() const {
    return int(dim_extent_.size());
}

/// Prints a layout
std::ostream & Layout::write(std::ostream &out) const {
    std::cout << "Layout: [";
    for (size_t i = 0; i < layout_.size(); ++i) {
        std::cout << "(" << layout_.at(i).dim << ": " << layout_.at(i).size << ") ";
    }
    std::cout << "] - rank: " << rank() << "\n";

    std::cout << "Extent: [";
    for (size_t i = 0; i < layout_.size(); ++i) {
        std::cout << (i ? ", " : "") << extent_.at(i);
    }
    std::cout << "]\n";
    for (size_t r = 0; r < dim_extent_.size(); ++r) {
        std::cout << " Dim " << r << ": [";
        for (int i = 0; i < dim_extent_.at(r).size(); ++i) {
            std::cout << (i ? ", " : "") << dim_extent_.at(r).at(i);
        }
        std::cout << "]\n";
    }
    return out;
}

/// Maps an index to a given coordinate
Layout::Coordinate Layout::operator()(int index) const {

    Coordinate coord(rank(), 0);

    for (size_t i = 0; i < layout_.size() - 1; ++i) {

        int quotient = (index / extent_.at(i + 1));
        index = (index % extent_.at(i + 1));

        coord.at(layout_.at(i).dim) += quotient * dim_extent_.at(layout_.at(i).dim).at(i + 1);
    }

    coord.at(layout_.back().dim) += index;

    return coord;
}

/// Maps a coordinate to an index
int Layout::operator()(Layout::Coordinate const &_coord) const {

    Coordinate coord(_coord);
    int index = 0;

    for (size_t i = layout_.size(); i > 0; --i) {
        size_t idx = i - 1;

        int dim = layout_.at(idx).dim;
        int size = layout_.at(idx).size;

        int items = coord.at(dim);

        int quotient = items / size;
        int remainder = items % size;

        index += remainder * (i < layout_.size() ? extent_.at(idx + 1) : 1);
        coord.at(dim) = quotient;
    }

    return index;
}

}

// test::Layout::Coordinate is actually a std::vector<>, so for ADL lookup to
// work, the operator<< must be in std::. GCC does look it up in global
// namespace, but that's a bug.
namespace std {
std::ostream & operator<<(std::ostream &out, test::Layout::Coordinate const &coord) {
    for (int i = 0; i < coord.size(); ++i) {
        out << (i ? ", " : "") << coord.at(i);
    }
    return out;
}
} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout, igemm) {

    test::Layout::SpanVector layout_def;
    typedef test::Layout::Span Span;

    layout_def.push_back(Span(0, 8));
    layout_def.push_back(Span(1, 4));
    layout_def.push_back(Span(0, 4));

    test::Layout layout(layout_def);

    for (int i = 0; i < 33; ++i) {
        test::Layout::Coordinate coord = layout(i);
        int index = layout(coord);
        EXPECT_EQ(i, index)
            << "[" << i << "] - (" << layout(i) << ") => " << layout(layout(i)) << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout, sgemm_accum) {

    test::Layout::SpanVector layout_def;
    typedef test::Layout::Span Span;

    layout_def.push_back(Span(0, 2));
    layout_def.push_back(Span(1, 8));
    layout_def.push_back(Span(0, 2));

    test::Layout layout(layout_def);

    for (int i = 0; i < 32; ++i) {
        test::Layout::Coordinate coord = layout(i);
        int index = layout(coord);
        EXPECT_EQ(i, index)
            << "[" << i << "] - (" << layout(i) << ") => " << layout(layout(i)) << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
