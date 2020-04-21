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

#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace open3d {
namespace utility {

/// The namespace hash_tuple defines a general hash function for std::tuple
/// See this post for details:
///   http://stackoverflow.com/questions/7110301
/// The hash_combine code is from boost
/// Reciprocal of the golden ratio helps spread entropy and handles duplicates.
/// See Mike Seymour in magic-numbers-in-boosthash-combine:
///   http://stackoverflow.com/questions/4948780

namespace hash_tuple {

template <typename TT>
struct hash {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
    seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& seed, Tuple const& tuple) {
        hash_combine(seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace

template <typename... TT>
struct hash<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
        return seed;
    }
};

}  // namespace hash_tuple

namespace hash_eigen {

template <typename T>
struct hash {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

}  // namespace hash_eigen

namespace hash_enum_class {

// Hash function for enum class for C++ standard less than C++14
// https://stackoverflow.com/a/24847480/1255535
struct hash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

}  // namespace hash_enum_class

/// Function to split a string, mimics boost::split
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
void SplitString(std::vector<std::string>& tokens,
                 const std::string& str,
                 const std::string& delimiters = " ",
                 bool trim_empty_str = true);

/// String util: find length of current word staring from a position
/// By default, alpha numeric chars and chars in valid_chars are considered
/// as valid charactors in a word
size_t WordLength(const std::string& doc,
                  size_t start_pos,
                  const std::string& valid_chars = "_");

std::string& LeftStripString(std::string& str,
                             const std::string& chars = "\t\n\v\f\r ");

std::string& RightStripString(std::string& str,
                              const std::string& chars = "\t\n\v\f\r ");

/// Strip empty charactors in front and after string. Similar to Python's
/// str.strip()
std::string& StripString(std::string& str,
                         const std::string& chars = "\t\n\v\f\r ");

/// Convet string to the lower case
std::string ToLower(const std::string& s);

void Sleep(int milliseconds);

/// Computes the quotient of x/y with rounding up
inline int DivUp(int x, int y) {
    div_t tmp = std::div(x, y);
    return tmp.quot + (tmp.rem != 0 ? 1 : 0);
}

/// Thread-safe function returning a pseudo-random integer.
/// The integer is drawn from a uniform distribution bounded by min and max
/// (inclusive)
int UniformRandInt(const int min, const int max);

/// Uniformly distributed binary-friendly floating point number in [0, 1).
///
/// Binary-friendly means that the random number can be represented by floating
/// point with a few bits of mantissa. The binary-friendliness is useful for
/// unit testing since it reduces the chances of numerical errors.
///
/// E.g.
/// - 0.9 is not representable by floating point accurately, the actual value
///   stored in a float32 is 0.89999997615814208984375...
/// - 0.875 = 0.5 + 0.25 + 0.125, is binary-friendly.
///
/// \param power The possible random numbers are: n * 1 / (2 ^ power),
///              where n = 0, 1, 2, ..., (2 ^ power - 1).
template <typename T>
T UniformRandFloatBinaryFriendly(unsigned int power = 5) {
    double p = std::pow(2, power);
    int n = UniformRandInt(0, p - 1);
    return static_cast<T>(1. / p * n);
}

}  // namespace utility
}  // namespace open3d
