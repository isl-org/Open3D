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
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace open3d {
namespace ml {
namespace op_util {

/// Class for representing a possibly unknown dimension value
class DimValue {
public:
    DimValue() : value_(0), constant_(false) {}
    DimValue(int64_t v) : value_(v), constant_(true) {}
    DimValue& operator*=(const DimValue& b) {
        if (constant_ && b.constant_)
            value_ *= b.value_;
        else
            constant_ = false;
        return *this;
    }
    std::string ToString() const {
        if (constant_)
            return std::to_string(value_);
        else
            return "?";
    }
    int64_t& value() {
        if (!constant_) throw std::runtime_error("DimValue is not constant");
        return value_;
    }
    bool& constant() { return constant_; }

private:
    int64_t value_;
    bool constant_;
};

inline DimValue UnknownValue() { return DimValue(); }

/// Class for dimensions for which the value should be inferred.
class Dim {
public:
    explicit Dim() : value_(0), constant_(false), origin_(this) {}

    explicit Dim(const std::string& name)
        : value_(0), constant_(false), origin_(this), name_(name) {}

    Dim(int64_t value, const std::string& name = "")
        : value_(value), constant_(true), origin_(nullptr), name_(name) {}

    Dim(const Dim& other)
        : value_(other.value_),
          constant_(other.constant_),
          origin_(other.origin_),
          name_(other.name_) {}

    ~Dim() {}

    Dim& operator=(const Dim&) = delete;

    int64_t& value() {
        if (origin_)
            return origin_->value_;
        else
            return value_;
    }

    bool& constant() {
        if (origin_)
            return origin_->constant_;
        else
            return constant_;
    }

    /// tries to assign a value to the Dim if not marked as constant and
    /// compares the Dim value with the value to be assigned.
    bool assign(int64_t a) {
        if (!constant()) {
            value() = a;
            constant() = true;
        }
        return value() == a;
    }

    std::string ToString(bool show_value = true) {
        if (name_.size()) {
            if (show_value)
                return name_ + "(" +
                       (constant() ? std::to_string(value()) : "?") + ")";
            else
                return name_;
        }
        if (constant())
            return std::to_string(value());
        else
            return "?";
    }

private:
    int64_t value_;
    bool constant_;
    Dim* origin_;
    std::string name_;
};

//
// Dim expression operator classes
//

struct DimXPlus {
    static bool constant() { return true; };
    static int64_t apply(int64_t a, int64_t b) { return a + b; }

    template <class T1, class T2>
    static bool backprop(int64_t ans, T1 a, T2 b) {
        if (!a.constant() && a.constant() == b.constant()) {
            std::string exstr =
                    GetString(a, false) + ToString() + GetString(b, false);
            throw std::runtime_error("Illegal dim expression: " + exstr);
            return false;
        } else if (!a.constant()) {
            return a.assign(ans - b.value());
        } else {
            return b.assign(ans - a.value());
        }
    }

    static std::string ToString() { return "+"; }
};

struct DimXMinus {
    static bool constant() { return true; };
    static int64_t apply(int64_t a, int64_t b) { return a - b; }

    template <class T1, class T2>
    static bool backprop(int64_t ans, T1 a, T2 b) {
        if (!a.constant() && a.constant() == b.constant()) {
            std::string exstr =
                    GetString(a, false) + ToString() + GetString(b, false);
            throw std::runtime_error("Illegal dim expression: " + exstr);
            return false;
        } else if (!a.constant()) {
            return a.assign(ans + b.value());
        } else {
            return b.assign(a.value() - ans);
        }
    }

    static std::string ToString() { return "-"; }
};

struct DimXMultiply {
    static bool constant() { return true; };
    static int64_t apply(int64_t a, int64_t b) { return a * b; }

    template <class T1, class T2>
    static bool backprop(int64_t ans, T1 a, T2 b) {
        std::string exstr =
                GetString(a, false) + ToString() + GetString(b, false);
        throw std::runtime_error("Illegal dim expression: " + exstr);
        return false;
    }

    static std::string ToString() { return "*"; }
};

struct DimXDivide {
    static bool constant() { return true; };
    static int64_t apply(int64_t a, int64_t b) { return a / b; }

    template <class T1, class T2>
    static bool backprop(int64_t ans, T1 a, T2 b) {
        std::string exstr =
                GetString(a, false) + ToString() + GetString(b, false);
        throw std::runtime_error("Illegal dim expression: " + exstr);
        return false;
    }

    static std::string ToString() { return "/"; }
};

struct DimXOr {
    static bool constant() { return false; };
    static int64_t apply(int64_t a, int64_t b) {
        throw std::runtime_error("Cannot evaluate OR expression");
        return 0;
    }
    template <class T1, class T2>
    static bool backprop(int64_t ans, T1 a, T2 b) {
        return a.assign(ans) || b.assign(ans);
    }

    static std::string ToString() { return "||"; }
};

/// Dim expression class
template <class TLeft, class TRight, class TOp>
class DimX {
public:
    static DimX<TLeft, TRight, TOp> Create(TLeft left, TRight right) {
        return DimX(left, right);
    }

    int64_t value() {
        if (constant_) {
            return TOp::apply(left_.value(), right_.value());
        }
        return 0;
    }

    bool& constant() { return constant_; }

    /// assigns a value to the expression
    bool assign(int64_t a) {
        if (constant_) {
            return value() == a;
        } else {
            return TOp::backprop(a, left_, right_);
        }
    }

    std::string ToString(bool show_value = true) {
        return left_.ToString(show_value) + TOp::ToString() +
               right_.ToString(show_value);
    }

private:
    DimX(TLeft left, TRight right) : left_(left), right_(right) {
        constant_ = left.constant() && right.constant() && TOp::constant();
    }
    TLeft left_;
    TRight right_;
    bool constant_;
};

//
// define operators for dim expressions
//

#define DEFINE_DIMX_OPERATOR(opclass, symbol)                             \
    inline DimX<Dim, Dim, opclass> operator symbol(Dim a, Dim b) {        \
        return DimX<Dim, Dim, opclass>::Create(a, b);                     \
    }                                                                     \
                                                                          \
    template <class TL, class TR, class TOp>                              \
    inline DimX<Dim, DimX<TL, TR, TOp>, opclass> operator symbol(         \
            Dim a, DimX<TL, TR, TOp>&& b) {                               \
        return DimX<Dim, DimX<TL, TR, TOp>, opclass>::Create(a, b);       \
    }                                                                     \
                                                                          \
    template <class TL, class TR, class TOp>                              \
    inline DimX<DimX<TL, TR, TOp>, Dim, opclass> operator symbol(         \
            DimX<TL, TR, TOp>&& a, Dim b) {                               \
        return DimX<DimX<TL, TR, TOp>, Dim, opclass>::Create(a, b);       \
    }                                                                     \
                                                                          \
    template <class TL1, class TR1, class TOp1, class TL2, class TR2,     \
              class TOp2>                                                 \
    inline DimX<DimX<TL1, TR1, TOp1>, DimX<TL2, TR2, TOp2>, opclass>      \
    operator symbol(DimX<TL1, TR1, TOp1>&& a, DimX<TL2, TR2, TOp2>&& b) { \
        return DimX<DimX<TL1, TR1, TOp1>, DimX<TL2, TR2, TOp2>,           \
                    opclass>::Create(a, b);                               \
    }

DEFINE_DIMX_OPERATOR(DimXPlus, +)
DEFINE_DIMX_OPERATOR(DimXMinus, -)
DEFINE_DIMX_OPERATOR(DimXMultiply, *)
DEFINE_DIMX_OPERATOR(DimXDivide, /)
DEFINE_DIMX_OPERATOR(DimXOr, ||)
#undef DEFINE_DIMX_OPERATOR

//
// define operators for comparing DimValue to dim expressions.
// Using these operators will try to assign the dim value to the expression.
//

template <class TLeft, class TRight, class TOp>
inline bool operator==(DimValue a, DimX<TLeft, TRight, TOp>&& b) {
    if (a.constant()) {
        auto b_copy(b);
        return b_copy.assign(a.value());
    } else
        return true;
}

inline bool operator==(DimValue a, Dim b) {
    if (a.constant())
        return b.assign(a.value());
    else
        return true;
}

//
// some helper classes
//

template <class... args>
struct CountArgs {
    static const size_t value = sizeof...(args);
};

template <class TLeft, class TRight, class TOp>
std::string GetString(DimX<TLeft, TRight, TOp> a, bool show_value = true) {
    return a.ToString(show_value);
}

inline std::string GetString(Dim a, bool show_value = true) {
    return a.ToString(show_value);
}

template <class TLeft, class TRight, class TOp>
int64_t GetValue(DimX<TLeft, TRight, TOp> a) {
    return a.value();
}

template <class TLeft, class TRight, class TOp>
int64_t GetValue(DimX<TLeft, TRight, TOp> a, int64_t unknown_dim_value) {
    if (a.constant()) {
        return a.value();
    } else {
        return unknown_dim_value;
    }
    return a.value();
}

inline int64_t GetValue(Dim a) { return a.value(); }

inline int64_t GetValue(Dim a, int64_t unknown_dim_value) {
    if (a.constant()) {
        return a.value();
    } else {
        return unknown_dim_value;
    }
}

inline std::string CreateDimXString() { return std::string(); }

template <class TDimX>
std::string CreateDimXString(TDimX dimex) {
    return GetString(dimex);
}

template <class TDimX, class... TArgs>
std::string CreateDimXString(TDimX dimex, TArgs... args) {
    return GetString(dimex) + ", " + CreateDimXString(args...);
}

template <class TDimX>
void CreateDimVector(std::vector<int64_t>& out,
                     int64_t unknown_dim_value,
                     TDimX dimex) {
    out.push_back(GetValue(dimex, unknown_dim_value));
}

template <class TDimX, class... TArgs>
void CreateDimVector(std::vector<int64_t>& out,
                     int64_t unknown_dim_value,
                     TDimX dimex,
                     TArgs... args) {
    out.push_back(GetValue(dimex, unknown_dim_value));
    CreateDimVector(out, unknown_dim_value, args...);
}

template <class TDimX>
std::vector<int64_t> CreateDimVector(int64_t unknown_dim_value, TDimX dimex) {
    std::vector<int64_t> out;
    CreateDimVector(out, unknown_dim_value, dimex);
    return out;
}

template <class TDimX, class... TArgs>
std::vector<int64_t> CreateDimVector(int64_t unknown_dim_value,
                                     TDimX dimex,
                                     TArgs... args) {
    std::vector<int64_t> out;
    CreateDimVector(out, unknown_dim_value, dimex, args...);
    return out;
}

//
// classes which check if the dim value is compatible with the expression
//

template <class TLeft, class TRight, class TOp>
bool CheckDim(const DimValue& lhs, DimX<TLeft, TRight, TOp>&& rhs) {
    bool status = (lhs == std::forward<DimX<TLeft, TRight, TOp>>(rhs));
    return status;
}

inline bool CheckDim(const DimValue& lhs, Dim d) {
    bool status = lhs == d;
    return status;
}

/// Check shape options
enum class CSOpt {
    NONE = 0,
    COMBINE_FIRST_DIMS,
    IGNORE_FIRST_DIMS,
    COMBINE_LAST_DIMS,
    IGNORE_LAST_DIMS
};

template <CSOpt Opt = CSOpt::NONE, class TDimX>
bool _CheckShape(const std::vector<DimValue>& shape, TDimX&& dimex) {
    // check rank
    const int rank_diff = shape.size() - 1;
    if (Opt != CSOpt::NONE) {
        if (rank_diff < 0) {
            return false;
        }
    } else {
        if (rank_diff != 0) {
            return false;
        }
    }

    // check dim
    bool status;
    if (Opt == CSOpt::COMBINE_FIRST_DIMS) {
        DimValue s(1);
        for (int i = 0; i < rank_diff + 1; ++i) s *= shape[i];
        status = CheckDim(s, std::forward<TDimX>(dimex));
    } else if (Opt == CSOpt::IGNORE_FIRST_DIMS) {
        status = CheckDim(shape[rank_diff], std::forward<TDimX>(dimex));
    } else if (Opt == CSOpt::COMBINE_LAST_DIMS) {
        DimValue s(1);
        for (DimValue x : shape) s *= x;
        status = CheckDim(s, std::forward<TDimX>(dimex));
    } else {
        status = CheckDim(shape[0], std::forward<TDimX>(dimex));
    }
    return status;
}

template <CSOpt Opt = CSOpt::NONE, class TDimX, class... TArgs>
bool _CheckShape(const std::vector<DimValue>& shape,
                 TDimX&& dimex,
                 TArgs&&... args) {
    // check rank
    const int rank_diff = shape.size() - (CountArgs<TArgs...>::value + 1);
    if (Opt != CSOpt::NONE) {
        if (rank_diff < 0) {
            return false;
        }
    } else {
        if (rank_diff != 0) {
            return false;
        }
    }

    // check dim
    bool status;
    if (Opt == CSOpt::COMBINE_FIRST_DIMS) {
        DimValue s(1);
        for (int i = 0; i < rank_diff + 1; ++i) s *= shape[i];
        status = CheckDim(s, std::forward<TDimX>(dimex));
    } else if (Opt == CSOpt::IGNORE_FIRST_DIMS) {
        status = CheckDim(shape[rank_diff], std::forward<TDimX>(dimex));
    } else {
        status = CheckDim(shape[0], std::forward<TDimX>(dimex));
    }

    const int offset = 1 + (Opt == CSOpt::COMBINE_FIRST_DIMS ||
                                            Opt == CSOpt::IGNORE_FIRST_DIMS
                                    ? rank_diff
                                    : 0);
    std::vector<DimValue> shape2(shape.begin() + offset, shape.end());
    bool status2 = _CheckShape<Opt>(shape2, std::forward<TArgs>(args)...);

    return status && status2;
}

/// Function for checking a shape with dim expressions.
/// Usage example:
///
///   Dim depth("depth");
///   Dim height("height");
///   Dim width("width");
///   status = CheckShape({30,40}, height, width); // VALID, will assign values
///                                                // to height and width
///
///   status = CheckShape({50,41}, height+20, width+1); // VALID, values match
///   status = CheckShape({20,30,40}, depth+10, height, width); // VALID, will
///                                                      // assign 10 to depth
///
///   status = CheckShape({0},depth||0); // VALID, shape must match depth or 0
///   status = CheckShape({10}, depth||0); // VALID, shape must match depth or 0
///   status = CheckShape({123,10}, Dim(), depth); // VALID, first dim may be
///                                                // anything
///
///   status = CheckShape<CSOpt::COMBINE_LAST_DIMS>({123,10,4}, Dim(), width);
///                                                   // VALID, width==40==10*4
///
///   status = CheckShape<CSOpt::COMBINE_FIRST_DIMS>(
///                                     {10,2,2,123,456}, width, Dim(), Dim());
///                                                 // VALID, width==40==10*2*2
///
///   status = CheckShape({70}, height+width); // VALID, works because height
///       // and width have been initialized since the first call to CheckShape
///
///   status = CheckShape({1,2,3}, Dim(), Dim()); // INVALID, rank mismatch 3vs2
///   status = CheckShape({1,2,3}, depth, width, height); // INVALID, at least
///                                                    // one dim does not match
///
///   The options CSOpt::COMBINE_FIRST_DIMS and CSOpt::COMBINE_LAST_DIMS allow
///   to match the rank of the dim expressions by combining the shape
///   dimensions at the beginning or end.
///   The options CSOpt::IGNORE_FIRST_DIMS and CSOpt::IGNORE_LAST_DIMS allow to
///   ignore additional dimensions in the shape.
///
///   The shape to be checked may contain unknowns
///   Dim A("A");
///   Dim B("B");
///   status = CheckShape({30, UnknownValue()}, A, B); // VALID, A is 30 and B
///                                                   // is still unknown
///
///   status =
///   CheckShape<CSOpt::COMBINE_LAST_DIMS>({30,1,2,UnknownValue()},A,B);
///                                     // VALID, A is 30 and B is still unknown
///
///   The following shows some limitations of the dim expressions
///   Dim A("A");
///   Dim B("B");
///   status = CheckShape({30}, A+B); // THROWS EXCEPTION, illegal expression
///                                   // because neither A or B is a constant
///
///   However, the following will work
///   Dim A(20,"A");
///   Dim B("B");
///   status = CheckShape({30}, A+B); // VALID, B is now 10
///
///   This will work, too
///   Dim A("A"); // uninitialized
///   Dim B("B");
///   status = CheckShape({20}, A); // VALID, A is now 20
///   status = CheckShape({30}, A+B); // VALID, B is now 10
///
///   Multiplication and division are not allowed for unknown dims
///   Dim A("A");
///   status = CheckShape({30}, 3*A); // THROWS EXCEPTION, although expression
///                                   // seems reasonable
///   status = CheckShape({20}, 3*A); // THROWS EXCEPTION, this
///                 // is the reason why mul/div is only allowed for known dims
///
///   Important, do not create objects of dim expressions, i.e.,
///     auto dimx = Dim("tmp") + 3;
///     status = CheckShape({20}, dimx); // intended to not compile
///   Assigning a value to dimx will assign a value to Dim("tmp") which has a
///   shorter lifetime.
///
/// The return value is a tuple <bool,std::string>. If the bool is false
/// then the shape is INVALID and the string contains an error message of the
/// form "got [shape], expected [dim expressions]".
/// If true then the shape is VALID and the error string is empty.
///
/// Note the goal of this function is to simplify checking tensor shapes. There
/// may be cases where shapes cannot be checked with the provided functionality
/// and you have to write custom shape checking code.
///
/// \param shape  This is the actual shape of an object.
/// \param args   This is a list of dim expression
///
template <CSOpt Opt = CSOpt::NONE, class TDimX, class... TArgs>
std::tuple<bool, std::string> CheckShape(const std::vector<DimValue>& shape,
                                         TDimX&& dimex,
                                         TArgs&&... args) {
    const bool status = _CheckShape<Opt>(shape, std::forward<TDimX>(dimex),
                                         std::forward<TArgs>(args)...);
    if (status) {
        return std::make_tuple(status, std::string());
    } else {
        const int rank_diff = shape.size() - (CountArgs<TArgs...>::value + 1);

        // generate string for the actual shape. This is a bit involved because
        // of the many options.
        std::string shape_str;
        if (rank_diff <= 0) {
            shape_str = "[";
            for (int i = 0; i < int(shape.size()); ++i) {
                shape_str += shape[i].ToString();
                if (i + 1 < int(shape.size())) shape_str += ", ";
            }
            shape_str += "]";
        } else {
            if (Opt == CSOpt::COMBINE_FIRST_DIMS) {
                shape_str += "[";
                for (int i = 0; i < rank_diff; ++i) {
                    shape_str += shape[i].ToString();
                    if (i + 1 < int(shape.size())) shape_str += "*";
                }
            } else if (Opt == CSOpt::IGNORE_FIRST_DIMS) {
                shape_str += "(";
                for (int i = 0; i < rank_diff; ++i) {
                    shape_str += shape[i].ToString();
                    if (i + 1 < rank_diff) shape_str += ", ";
                }
                shape_str += ")[";
            } else {
                shape_str = "[";
            }
            int start = 0;
            if (Opt == CSOpt::COMBINE_FIRST_DIMS ||
                Opt == CSOpt::IGNORE_FIRST_DIMS) {
                start = rank_diff;
            }

            int end = shape.size();
            if (Opt == CSOpt::COMBINE_LAST_DIMS) {
                end -= rank_diff + 1;
            } else if (Opt == CSOpt::IGNORE_LAST_DIMS) {
                end -= rank_diff;
            }
            for (int i = start; i < end; ++i) {
                shape_str += shape[i].ToString();
                if (i + 1 < end) shape_str += ", ";
            }
            if (Opt == CSOpt::COMBINE_LAST_DIMS) {
                shape_str += ", ";
                for (int i = std::max<int>(0, shape.size() - rank_diff - 1);
                     i < int(shape.size()); ++i) {
                    shape_str += shape[i].ToString();
                    if (i + 1 < int(shape.size())) shape_str += "*";
                }
                shape_str += "]";
            } else if (Opt == CSOpt::IGNORE_LAST_DIMS) {
                shape_str += "](";
                for (int i = std::max<int>(0, shape.size() - rank_diff);
                     i < int(shape.size()); ++i) {
                    shape_str += shape[i].ToString();
                    if (i + 1 < int(shape.size())) shape_str += ", ";
                }
                shape_str += ")";
            } else {
                shape_str += "]";
            }
        }

        // generate string for the expected shape with the dim expressions
        std::string expected_shape;
        if ((CountArgs<TArgs...>::value + 1) == 1) {
            expected_shape = "[" + GetString(dimex) + "]";

        } else {
            expected_shape = "[" + GetString(dimex) + ", " +
                             CreateDimXString(args...) + "]";
        }

        std::string errstr;
        // print rank information if there is a problem with the rank
        if ((Opt != CSOpt::NONE && rank_diff < 0) ||
            (Opt == CSOpt::NONE && rank_diff != 0)) {
            errstr = "got rank " + std::to_string(shape.size()) + " " +
                     shape_str + ", expected rank " +
                     std::to_string(CountArgs<TArgs...>::value + 1) + " " +
                     expected_shape;
        } else {  // rank is OK print just the shapes
            errstr = "got " + shape_str + ", expected " + expected_shape;
        }
        return std::make_tuple(status, errstr);
    }
}

}  // namespace op_util
}  // namespace ml
}  // namespace open3d
