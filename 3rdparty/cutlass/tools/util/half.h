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
/*!
    \file
    \brief Host-side implementation of half-precision float
*/

#pragma once

#include <stdint.h>
#include <cmath>
#include <limits>
#include <utility>
#include <utility>

#include <iomanip>
#include <istream>
#include <ostream>

#include <cuda_fp16.h>

namespace cutlass {

/// IEEE binary16 floating-point value
class half_t {
 public:
  half_t();
  half_t(int);     /// conversion from integer
  half_t(float);   /// conversion from fp32
  half_t(double);  /// conversion from fp64

  static half_t bitcast(unsigned short);  /// bitcast performs no conversion

  static half_t convert(float const&);          /// FP conversion - round toward nearest even
  static float convert(unsigned short const&);  /// floating point conversion to fp32

  static half_t zero() { return bitcast(0); }          /// +zero
  static half_t one() { return bitcast(0x3c00); }      /// one
  static half_t nan() { return bitcast(0x7fff); }      /// canonical not a number
  static half_t inf() { return bitcast(0x7c00); }      /// +infinity
  static half_t ninf() { return bitcast(0xfc00); }     /// -infinity
  static half_t epsilon() { return bitcast(0x1000); }  /// Machine epsilon

  bool signbit() const;             /// sign bit - true: negative, false: positive
  int exponent() const;             /// unbiased exponent
  unsigned short mantissa() const;  /// mantissa bits

  bool isfinite() const;  /// true if neither inf nor nan
  bool isinf() const;     /// true if value is + or - infinity
  bool isnan() const;     /// true if value is not a number
  bool isnormal() const;  /// true if nonzero value is normalized
  bool iszero() const;    /// true if value is + or - zero

  bool operator==(half_t const&) const;
  bool operator!=(half_t const&) const;
  bool operator==(float const&) const;
  bool operator!=(float const&) const;

  bool operator<(half_t const&) const;
  bool operator<=(half_t const&) const;
  bool operator>(half_t const&) const;
  bool operator>=(half_t const&) const;

  half_t operator+(half_t const&) const;
  half_t operator-() const;
  half_t operator-(half_t const&) const;
  half_t operator*(half_t const&) const;
  half_t operator/(half_t const&) const;

  half_t& operator+=(half_t const&);
  half_t& operator-=(half_t const&);
  half_t& operator*=(half_t const&);
  half_t& operator/=(half_t const&);

  half_t& operator++();
  half_t& operator--();
  half_t operator++(int);
  half_t operator--(int);

  operator bool() const;   /// false if zero
  operator int() const;    /// conversion to int
  operator float() const;  /// conversion to fp32
  operator half() const;   /// conversion to half

  uint16_t& raw() { return x; }
  uint16_t raw() const { return x; }

#if defined(__clang__)
  __device__ half_t operator+(half_t const&) const;
  __device__ half_t operator*(half_t const&) const;
  __device__ operator float() const;  /// conversion to fp32
#endif

  //
  // Stream interactions
  //

  /// put to stream - half_t-precision types bitcast as unsigned shorts if base is hexadecimal
  friend std::ostream& operator<<(std::ostream& out, cutlass::half_t const& h) {
    if (out.flags() & std::ios::hex) {
      return out << h.x;
    } else {
      return out << float(h);
    }
  }

  /// read from stream - half_t-precision types parsed as unsigned shorts if base is hexadecimal
  friend std::istream& operator>>(std::istream& in, cutlass::half_t& h) {
    if (in.flags() & std::ios::hex) {
      unsigned short u = 0;
      in >> u;
      h = cutlass::half_t::bitcast(u);
    } else {
      float f = 0;
      in >> f;
      h = cutlass::half_t(f);
    }
    return in;
  }

 public:
  /// data
  unsigned short x;
};

/// Packed pair of half-precision elements
class half2_t {
 public:
  half2_t();
  half2_t(half_t lo, half_t hi);
  half2_t(std::pair<float, float> const&);
  explicit half2_t(unsigned data);

  half2_t operator+(half2_t const&) const;
  half2_t operator-(half2_t const&) const;
  half2_t operator*(half2_t const&)const;
  half2_t operator/(half2_t const&) const;

  half2_t& operator+=(half2_t const&);
  half2_t& operator-=(half2_t const&);
  half2_t& operator*=(half2_t const&);
  half2_t& operator/=(half2_t const&);

  float dot(half2_t const&) const;         /// dot product with single-precision accumulation
  float dot(half2_t const&, float) const;  /// dot product with single-precision accumulation

  half_t doth(half2_t const&) const;          /// dot product with half_t-precision accumulation
  half_t doth(half2_t const&, half_t) const;  /// dot product with half_t-precision accumulation

  unsigned packed() const;

  operator std::pair<float, float>() const;
  operator unsigned() const;

 public:
  half_t lo;
  half_t hi;
};

template <typename Dest, typename Src>
Dest bitcast(Src const&);
template <>
float bitcast<float, unsigned>(unsigned const&);
template <>
float bitcast<float, int>(int const&);
template <>
unsigned bitcast<unsigned, float>(float const&);
template <>
half_t bitcast<half_t, unsigned short>(unsigned short const&);
template <>
unsigned short bitcast<unsigned short, half_t>(half_t const&);
template <>
half bitcast<half, unsigned short>(unsigned short const&);
}  // namespace cutlass

cutlass::half_t operator+(float, cutlass::half_t const&);
cutlass::half_t operator-(float, cutlass::half_t const&);
cutlass::half_t operator*(float, cutlass::half_t const&);
cutlass::half_t operator/(float, cutlass::half_t const&);

#ifdef BOOST_LEXICAL_CAST_INCLUDED
namespace boost {

/// lexical cast from string to half_t
template <>
cutlass::half_t lexical_cast<cutlass::half_t>(std::string const& arg);

/// lexical cast from half_t to string
template <>
std::string lexical_cast<std::string>(cutlass::half_t const& arg);
}  // namespace boost
#endif

#define HLF_MANT_DIG 10

namespace cutlass {

cutlass::half_t abs(cutlass::half_t const&);  /// absolute value

bool isnan(cutlass::half_t const&);  /// true if argument is NaN

bool isfinite(cutlass::half_t const&);  /// true if argument is neither NaN nor infinity

cutlass::half_t nanh(const char* = 0);  /// returns a not-a-number

bool isinf(cutlass::half_t const&);  /// returns true if argument is infinitey (+ or -)

bool isnormal(
    cutlass::half_t const&);  /// returns true if argument is normal (neither zero nor infinity)

int fpclassify(cutlass::half_t const&);  /// returns a flag classifying floating-point value

bool signbit(cutlass::half_t const&);  /// returns true if negative, false if positive

cutlass::half_t sqrt(cutlass::half_t const&);  /// square root of half_t
#if __cplusplus >= 201103L
cutlass::half_t copysign(cutlass::half_t const&, cutlass::half_t const&);
#endif
}

namespace std {
/// Numeric limits
template <>
struct numeric_limits<cutlass::half_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = HLF_MANT_DIG;

  static cutlass::half_t min() { return cutlass::half_t::bitcast(0x0001); }

  static cutlass::half_t lowest() { return cutlass::half_t::bitcast(0xfbff); }

  static cutlass::half_t max() { return cutlass::half_t::bitcast(0x7bff); }

  /// Returns smallest finite value
  static cutlass::half_t epsilon() { return cutlass::half_t::epsilon(); }

  /// Returns smallest finite value
  static cutlass::half_t round_error() { return cutlass::half_t(0.5f); }

  /// Returns smallest finite value
  static cutlass::half_t infinity() { return cutlass::half_t::inf(); }

  /// Returns smallest finite value
  static cutlass::half_t quiet_NaN() { return cutlass::half_t::nan(); }

  /// Returns smallest finite value
  static cutlass::half_t signaling_NaN() { return cutlass::half_t::nan(); }

  /// Returns smallest finite value
  static cutlass::half_t denorm_min() { return cutlass::half_t::bitcast(0x0001); }
};
}  // namespace std

//
//
//

inline cutlass::half_t cutlass::half_t::bitcast(unsigned short _x) {
  half_t h;
  h.x = _x;
  return h;
}

/// FP32 -> FP16 conversion - rounds to nearest even
inline cutlass::half_t cutlass::half_t::convert(float const& flt) {
  // software implementation rounds toward nearest even
  unsigned const& s = *reinterpret_cast<unsigned const*>(&flt);
  uint16_t sign = uint16_t((s >> 16) & 0x8000);
  int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
  int mantissa = s & 0x7fffff;
  uint16_t u = 0;

  if ((s & 0x7fffffff) == 0) {
    // sign-preserving zero
    return cutlass::half_t::bitcast(sign);
  }

  if (exp > 15) {
    if (exp == 128 && mantissa) {
      // not a number
      u = 0x7fff;
    } else {
      // overflow to infinity
      u = sign | 0x7c00;
    }
    return cutlass::half_t::bitcast(u);
  }

  int sticky_bit = 0;

  if (exp >= -14) {
    // normal fp32 to normal fp16
    exp = uint16_t(exp + uint16_t(15));
    u = uint16_t(((exp & 0x1f) << 10));
    u = uint16_t(u | (mantissa >> 13));
  } else {
    // normal single-precision to subnormal half_t-precision representation
    int rshift = (-14 - exp);
    if (rshift < 32) {
      mantissa |= (1 << 23);

      sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

      mantissa = (mantissa >> rshift);
      u = (uint16_t(mantissa >> 13) & 0x3ff);
    } else {
      mantissa = 0;
      u = 0;
    }
  }

  // round to nearest even
  int round_bit = ((mantissa >> 12) & 1);
  sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

  if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
    u = uint16_t(u + 1);
  }

  u |= sign;

  return cutlass::half_t::bitcast(u);
}

inline float cutlass::half_t::convert(unsigned short const& h) {
  int sign = ((h >> 15) & 1);
  int exp = ((h >> 10) & 0x1f);
  int mantissa = (h & 0x3ff);
  unsigned f = 0;

  if (exp > 0 && exp < 31) {
    // normal
    exp += 112;
    f = (sign << 31) | (exp << 23) | (mantissa << 13);
  } else if (exp == 0) {
    if (mantissa) {
      // subnormal
      exp += 113;
      while ((mantissa & (1 << 10)) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else {
      // sign-preserving zero
      f = (sign << 31);
    }
  } else if (exp == 31) {
    if (mantissa) {
      f = 0x7fffffff;  // not a number
    } else {
      f = (0xff << 23) | (sign << 31);  //  inf
    }
  }
  return *reinterpret_cast<float const*>(&f);
}

inline cutlass::half_t::half_t() {}

inline cutlass::half_t::half_t(int i) { x = convert(float(i)).x; }

inline cutlass::half_t::half_t(float f) { x = convert(f).x; }

inline cutlass::half_t::half_t(double d) { x = convert(float(d)).x; }

inline bool cutlass::half_t::signbit() const { return (x >> 15) & 1; }

inline int cutlass::half_t::exponent() const { return ((x >> 10) & 0x1f) - 15; }

inline unsigned short cutlass::half_t::mantissa() const { return x & 0x3ff; }

inline cutlass::half_t::operator bool() const { return (x & 0x7fff) != 0; }

inline cutlass::half_t::operator int() const { return static_cast<int>(convert(x)); }

inline cutlass::half_t::operator float() const { return convert(x); }

inline cutlass::half_t::operator half() const { return cutlass::bitcast<half, unsigned short>(x); }

inline bool cutlass::half_t::operator==(cutlass::half_t const& h) const {
  if (iszero() && h.iszero()) {
    return true;
  }
  return x == h.x;
}

inline bool cutlass::half_t::operator!=(cutlass::half_t const& h) const {
  if (iszero() && h.iszero()) {
    return false;
  }
  return x != h.x;
}

inline bool cutlass::half_t::operator==(float const& b) const { return x == half_t(b).x; }

inline bool cutlass::half_t::operator!=(float const& b) const { return x != half_t(b).x; }

inline bool cutlass::half_t::iszero() const { return (x & 0x7fff) == 0; }

inline bool cutlass::half_t::isfinite() const { return (exponent() < 16); }

inline bool cutlass::half_t::isnan() const {
  int exp = ((x >> 10) & 0x1f);
  if (exp == 0x1f) {
    return (x & 0x3ff) != 0;
  }
  return false;
}

inline bool cutlass::half_t::isinf() const {
  int exp = ((x >> 10) & 0x1f);
  if (exp == 0x1f) {
    return (x & 0x3ff) == 0;
  }
  return false;
}

inline bool cutlass::half_t::isnormal() const {
  int exp = exponent();
  return exp > -15 && exp < 16;
}

inline bool cutlass::half_t::operator<(half_t const& h) const {
  int sign = ((x >> 15) & 1);
  int h_sign = ((h.x >> 15) & 1);
  if (sign == h_sign) {
    return (x & 0x7fff) < (h.x & 0x7fff);
  } else if (sign) {
    return true;
  }
  return false;
}

inline bool cutlass::half_t::operator<=(half_t const& h) const {
  int sign = ((x >> 15) & 1);
  int h_sign = ((h.x >> 15) & 1);
  if (sign == h_sign) {
    return (x & 0x7fff) <= (h.x & 0x7fff);
  } else if (sign) {
    return true;
  }
  return false;
}

inline bool cutlass::half_t::operator>(half_t const& h) const {
  int sign = ((x >> 15) & 1);
  int h_sign = ((h.x >> 15) & 1);
  if (sign == h_sign) {
    return (x & 0x7fff) > (h.x & 0x7fff);
  } else if (h_sign) {
    return true;
  }
  return false;
}

inline bool cutlass::half_t::operator>=(half_t const& h) const {
  int sign = ((x >> 15) & 1);
  int h_sign = ((h.x >> 15) & 1);
  if (sign == h_sign) {
    return (x & 0x7fff) >= (h.x & 0x7fff);
  } else if (h_sign) {
    return true;
  }
  return false;
}

inline cutlass::half_t cutlass::half_t::operator+(cutlass::half_t const& b) const {
  return cutlass::half_t(float(*this) + float(b));
}

inline cutlass::half_t cutlass::half_t::operator-() const { return bitcast(x ^ 0x8000); }

inline cutlass::half_t cutlass::half_t::operator-(cutlass::half_t const& b) const {
  return cutlass::half_t(float(*this) - float(b));
}

inline cutlass::half_t cutlass::half_t::operator*(cutlass::half_t const& b) const {
  return cutlass::half_t(float(*this) * float(b));
}

inline cutlass::half_t cutlass::half_t::operator/(cutlass::half_t const& b) const {
  return cutlass::half_t(float(*this) / float(b));
}

inline cutlass::half_t& cutlass::half_t::operator+=(cutlass::half_t const& b) {
  *this = cutlass::half_t(float(*this) + float(b));
  return *this;
}

inline cutlass::half_t& cutlass::half_t::operator-=(cutlass::half_t const& b) {
  *this = cutlass::half_t(float(*this) - float(b));
  return *this;
}

inline cutlass::half_t& cutlass::half_t::operator*=(cutlass::half_t const& b) {
  *this = cutlass::half_t(float(*this) * float(b));
  return *this;
}

inline cutlass::half_t& cutlass::half_t::operator/=(cutlass::half_t const& b) {
  *this = cutlass::half_t(float(*this) / float(b));
  return *this;
}

inline cutlass::half_t& cutlass::half_t::operator++() {
  *this = cutlass::half_t(float(*this) + 1.0f);
  return *this;
}

inline cutlass::half_t& cutlass::half_t::operator--() {
  *this = cutlass::half_t(float(*this) - 1.0f);
  return *this;
}

inline cutlass::half_t cutlass::half_t::operator++(int) {
  half_t h = *this;
  *this = cutlass::half_t(float(*this) + 1.0f);
  return h;
}

inline cutlass::half_t cutlass::half_t::operator--(int) {
  half_t h = *this;
  *this = cutlass::half_t(float(*this) - 1.0f);
  return h;
}

inline cutlass::half_t operator+(float a, cutlass::half_t const& b) {
  return cutlass::half_t(a + float(b));
}

inline cutlass::half_t operator-(float a, cutlass::half_t const& b) {
  return cutlass::half_t(a - float(b));
}

inline cutlass::half_t operator*(float a, cutlass::half_t const& b) {
  return cutlass::half_t(a * float(b));
}

inline cutlass::half_t operator/(float a, cutlass::half_t const& b) {
  return cutlass::half_t(a / float(b));
}

//
//
//

inline cutlass::half2_t::half2_t() {}

inline cutlass::half2_t::half2_t(half_t lo, half_t hi) : lo(lo), hi(hi) {}

inline cutlass::half2_t::half2_t(std::pair<float, float> const& p) : lo(p.first), hi(p.second) {}

inline cutlass::half2_t::half2_t(unsigned data)
    : lo(half_t::bitcast(uint16_t(data & 0x0ffff))),
      hi(half_t::bitcast(uint16_t((data >> 16) & 0x0ffff))) {}

inline cutlass::half2_t cutlass::half2_t::operator+(half2_t const& b) const {
  return half2_t(lo + b.lo, hi + b.hi);
}

inline cutlass::half2_t cutlass::half2_t::operator-(half2_t const& b) const {
  return half2_t(lo - b.lo, hi - b.hi);
}

inline cutlass::half2_t cutlass::half2_t::operator*(half2_t const& b) const {
  return half2_t(lo * b.lo, hi * b.hi);
}

inline cutlass::half2_t cutlass::half2_t::operator/(half2_t const& b) const {
  return half2_t(lo / b.lo, hi / b.hi);
}

inline cutlass::half2_t& cutlass::half2_t::operator+=(half2_t const& b) {
  lo += b.lo;
  hi += b.hi;
  return *this;
}

inline cutlass::half2_t& cutlass::half2_t::operator-=(half2_t const& b) {
  lo -= b.lo;
  hi -= b.hi;
  return *this;
}

inline cutlass::half2_t& cutlass::half2_t::operator*=(half2_t const& b) {
  lo *= b.lo;
  hi *= b.hi;
  return *this;
}

inline cutlass::half2_t& cutlass::half2_t::operator/=(half2_t const& b) {
  lo /= b.lo;
  hi /= b.hi;
  return *this;
}

inline float cutlass::half2_t::dot(half2_t const& b) const {
  return float(lo) * float(b.lo) + float(hi) * float(b.hi);
}

inline float cutlass::half2_t::dot(half2_t const& b, float c) const { return c + dot(b); }

inline cutlass::half_t cutlass::half2_t::doth(half2_t const& b) const {
  return cutlass::half_t(dot(b));
}

inline cutlass::half_t cutlass::half2_t::doth(half2_t const& b, half_t c) const {
  return cutlass::half_t(dot(b, float(c)));
}

inline cutlass::half2_t::operator std::pair<float, float>() const {
  return std::pair<float, float>(float(lo), float(hi));
}

inline unsigned cutlass::half2_t::packed() const { return (lo.x | (hi.x << 16)); }

inline cutlass::half2_t::operator unsigned() const { return packed(); }

//
//
//

template <>
inline float cutlass::bitcast<float, unsigned>(unsigned const& u) {
  return *reinterpret_cast<float const*>(&u);
}

template <>
inline float cutlass::bitcast<float, int>(int const& i) {
  return *reinterpret_cast<float const*>(&i);
}

template <>
inline unsigned cutlass::bitcast<unsigned, float>(float const& f) {
  return *reinterpret_cast<unsigned const*>(&f);
}

template <>
inline cutlass::half_t cutlass::bitcast<cutlass::half_t, unsigned short>(unsigned short const& s) {
  return *reinterpret_cast<cutlass::half_t const*>(&s);
}

template <>
inline unsigned short cutlass::bitcast<unsigned short, cutlass::half_t>(cutlass::half_t const& h) {
  return *reinterpret_cast<unsigned short const*>(&h);
}

template <>
inline half cutlass::bitcast<half, unsigned short>(unsigned short const& s) {
  return *reinterpret_cast<half const*>(&s);
}

//
// Lexical casts
//

#ifdef BOOST_LEXICAL_CAST_INCLUDED
namespace boost {
template <>
cutlass::half_t lexical_cast<cutlass::half_t>(std::string const& arg) {
  return cutlass::half_t(boost::lexical_cast<float>(arg));
}

template <>
std::string lexical_cast<std::string>(cutlass::half_t const& arg) {
  return boost::lexical_cast<std::string>(float(arg));
}
}  // namespace boost
#endif

//
// Standard Library Operations
//

namespace cutlass {

inline cutlass::half_t abs(cutlass::half_t const& h) {
  return cutlass::half_t::bitcast(h.x & 0x7fff);
}

inline bool isnan(cutlass::half_t const& h) { return h.isnan(); }

inline bool isfinite(cutlass::half_t const& h) { return h.isfinite(); }

inline cutlass::half_t nanh(const char*) { return cutlass::half_t::nan(); }

inline bool isinf(cutlass::half_t const& h) { return h.isinf(); }

inline bool isnormal(cutlass::half_t const& h) { return h.isnormal(); }

inline int fpclassify(cutlass::half_t const& h) {
  int exp = h.exponent();
  unsigned short mantissa = h.mantissa();
  if (exp < -14) {
    if (mantissa == 0) {
      return FP_ZERO;
    } else {
      return FP_SUBNORMAL;
    }
  } else if (exp > 15) {
    if (mantissa == 0) {
      return FP_INFINITE;
    } else {
      return FP_NAN;
    }
  }
  return FP_NORMAL;
}

inline bool signbit(cutlass::half_t const& h) { return h.signbit(); }

inline cutlass::half_t sqrt(cutlass::half_t const& h) {
  return cutlass::half_t(std::sqrt(float(h)));
}

#if __cplusplus >= 201103L
inline cutlass::half_t copysign(cutlass::half_t const& a,
                                cutlass::half_t const& b) {
  return cutlass::half_t(std::copysign(float(a), float(b)));
}
#endif
}  // namespace std
