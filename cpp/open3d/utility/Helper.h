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

#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace open3d {
namespace utility {

/// hash_tuple defines a general hash function for std::tuple
/// See this post for details:
///   http://stackoverflow.com/questions/7110301
/// The hash_combine code is from boost
/// Reciprocal of the golden ratio helps spread entropy and handles duplicates.
/// See Mike Seymour in magic-numbers-in-boosthash-combine:
///   http://stackoverflow.com/questions/4948780

template <typename TT>
struct hash_tuple {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& hash_seed, T const& v) {
    hash_seed ^= std::hash<T>()(v) + 0x9e3779b9 + (hash_seed << 6) +
                 (hash_seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(hash_seed, tuple);
        hash_combine(hash_seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        hash_combine(hash_seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace

template <typename... TT>
struct hash_tuple<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t hash_seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(hash_seed, tt);
        return hash_seed;
    }
};

template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t hash_seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            hash_seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                         (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};

// Hash function for enum class for C++ standard less than C++14
// https://stackoverflow.com/a/24847480/1255535
struct hash_enum_class {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

/// Function to split a string, mimics boost::split
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiters = " ",
                                     bool trim_empty_str = true);

/// Returns true of the source string contains the destination string.
/// \param src Source string.
/// \param dst Destination string.
bool ContainsString(const std::string& src, const std::string& dst);

/// Returns true if \p src starts with \p tar.
/// \param src Source string.
/// \param tar Target string.
bool StringStartsWith(const std::string& src, const std::string& tar);

/// Returns true if \p src ends with \p tar.
/// \param src Source string.
/// \param tar Target string.
bool StringEndsWith(const std::string& src, const std::string& tar);

std::string JoinStrings(const std::vector<std::string>& strs,
                        const std::string& delimiter = ", ");

/// String util: find length of current word staring from a position
/// By default, alpha numeric chars and chars in valid_chars are considered
/// as valid characters in a word
size_t WordLength(const std::string& doc,
                  size_t start_pos,
                  const std::string& valid_chars = "_");

std::string& LeftStripString(std::string& str,
                             const std::string& chars = "\t\n\v\f\r ");

std::string& RightStripString(std::string& str,
                              const std::string& chars = "\t\n\v\f\r ");

/// Strip empty characters in front and after string. Similar to Python's
/// str.strip()
std::string& StripString(std::string& str,
                         const std::string& chars = "\t\n\v\f\r ");

/// Convert string to the lower case
std::string ToLower(const std::string& s);

/// Convert string to the upper case
std::string ToUpper(const std::string& s);

/// Format string
template <typename... Args>
inline std::string FormatString(const std::string& format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
                 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(),
                       buf.get() + size - 1);  // We don't want the '\0' inside
};

void Sleep(int milliseconds);

/// Computes the quotient of x/y with rounding up
inline int DivUp(int x, int y) {
    div_t tmp = std::div(x, y);
    return tmp.quot + (tmp.rem != 0 ? 1 : 0);
}

/// Returns current time stamp.
std::string GetCurrentTimeStamp();

}  // namespace utility
}  // namespace open3d
