// -*- C++ -*-
//===-- utils.h -----------------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// File contains common utilities that tests rely on

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>

#include "pstl_test_config.h"
#include "iterator_utils.h"

namespace TestUtils
{

typedef double float64_t;
typedef float float32_t;

template <class T, std::size_t N>
constexpr size_t
const_size(const T (&array)[N]) noexcept
{
    return N;
}

template <typename T>
class Sequence;

// Handy macros for error reporting
#define EXPECT_TRUE(condition, message) ::TestUtils::expect(true, condition, __FILE__, __LINE__, message)
#define EXPECT_FALSE(condition, message) ::TestUtils::expect(false, condition, __FILE__, __LINE__, message)

// Check that expected and actual are equal and have the same type.
#define EXPECT_EQ(expected, actual, message) ::TestUtils::expect_equal(expected, actual, __FILE__, __LINE__, message)

// Check that sequences started with expected and actual and have had size n are equal and have the same type.
#define EXPECT_EQ_N(expected, actual, n, message)                                                                      \
    ::TestUtils::expect_equal(expected, actual, n, __FILE__, __LINE__, message)

// Issue error message from outstr, adding a newline.
// Real purpose of this routine is to have a place to hang a breakpoint.
inline void
issue_error_message(std::stringstream& outstr)
{
    outstr << std::endl;
    std::cerr << outstr.str();
    std::exit(EXIT_FAILURE);
}

inline void
expect(bool expected, bool condition, const char* file, int32_t line, const char* message)
{
    if (condition != expected)
    {
        std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message;
        issue_error_message(outstr);
    }
}

// Do not change signature to const T&.
// Function must be able to detect const differences between expected and actual.
template <typename T>
void
expect_equal(T& expected, T& actual, const char* file, int32_t line, const char* message)
{
    if (!(expected == actual))
    {
        std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected " << expected << " got "
               << actual;
        issue_error_message(outstr);
    }
}

template <typename T>
void
expect_equal(Sequence<T>& expected, Sequence<T>& actual, const char* file, int32_t line, const char* message)
{
    size_t n = expected.size();
    size_t m = actual.size();
    if (n != m)
    {
        std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected sequence of size " << n
               << " got sequence of size " << m;
        issue_error_message(outstr);
        return;
    }
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k)
    {
        if (!(expected[k] == actual[k]))
        {
            std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k << " expected "
                   << expected[k] << " got " << actual[k];
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

template <typename Iterator1, typename Iterator2, typename Size>
void
expect_equal(Iterator1 expected_first, Iterator2 actual_first, Size n, const char* file, int32_t line,
             const char* message)
{
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k, ++expected_first, ++actual_first)
    {
        if (!(*expected_first == *actual_first))
        {
            std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k;
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

template <typename Iterator, typename F>
void
fill_data(Iterator first, Iterator last, F f)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;
    for (std::size_t i = 0; first != last; ++first, ++i)
    {
        *first = T(f(i));
    }
}

// Sequence<T> is a container of a sequence of T with lots of kinds of iterators.
// Prefixes on begin/end mean:
//      c = "const"
//      f = "forward"
// No prefix indicates non-const random-access iterator.
template <typename T>
class Sequence
{
    std::vector<T> m_storage;

  public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef ForwardIterator<iterator, std::forward_iterator_tag> forward_iterator;
    typedef ForwardIterator<const_iterator, std::forward_iterator_tag> const_forward_iterator;

    typedef BidirectionalIterator<iterator, std::bidirectional_iterator_tag> bidirectional_iterator;
    typedef BidirectionalIterator<const_iterator, std::bidirectional_iterator_tag> const_bidirectional_iterator;

    typedef T value_type;
    explicit Sequence(size_t size) : m_storage(size) {}

    // Construct sequence [f(0), f(1), ... f(size-1)]
    // f can rely on its invocations being sequential from 0 to size-1.
    template <typename Func>
    Sequence(size_t size, Func f)
    {
        m_storage.reserve(size);
        // Use push_back because T might not have a default constructor
        for (size_t k = 0; k < size; ++k)
            m_storage.push_back(T(f(k)));
    }
    Sequence(const std::initializer_list<T>& data) : m_storage(data) {}

    const_iterator
    begin() const
    {
        return m_storage.begin();
    }
    const_iterator
    end() const
    {
        return m_storage.end();
    }
    iterator
    begin()
    {
        return m_storage.begin();
    }
    iterator
    end()
    {
        return m_storage.end();
    }
    const_iterator
    cbegin() const
    {
        return m_storage.cbegin();
    }
    const_iterator
    cend() const
    {
        return m_storage.cend();
    }
    forward_iterator
    fbegin()
    {
        return forward_iterator(m_storage.begin());
    }
    forward_iterator
    fend()
    {
        return forward_iterator(m_storage.end());
    }
    const_forward_iterator
    cfbegin() const
    {
        return const_forward_iterator(m_storage.cbegin());
    }
    const_forward_iterator
    cfend() const
    {
        return const_forward_iterator(m_storage.cend());
    }
    const_forward_iterator
    fbegin() const
    {
        return const_forward_iterator(m_storage.cbegin());
    }
    const_forward_iterator
    fend() const
    {
        return const_forward_iterator(m_storage.cend());
    }

    const_bidirectional_iterator
    cbibegin() const
    {
        return const_bidirectional_iterator(m_storage.cbegin());
    }
    const_bidirectional_iterator
    cbiend() const
    {
        return const_bidirectional_iterator(m_storage.cend());
    }

    bidirectional_iterator
    bibegin()
    {
        return bidirectional_iterator(m_storage.begin());
    }
    bidirectional_iterator
    biend()
    {
        return bidirectional_iterator(m_storage.end());
    }

    std::size_t
    size() const
    {
        return m_storage.size();
    }
    const T*
    data() const
    {
        return m_storage.data();
    }
    typename std::vector<T>::reference operator[](size_t j) { return m_storage[j]; }
    const T& operator[](size_t j) const { return m_storage[j]; }

    // Fill with given value
    void
    fill(const T& value)
    {
        for (size_t i = 0; i < m_storage.size(); i++)
            m_storage[i] = value;
    }

    void
    print() const;

    template <typename Func>
    void
    fill(Func f)
    {
        fill_data(m_storage.begin(), m_storage.end(), f);
    }
};

template <typename T>
void
Sequence<T>::print() const
{
    std::cout << "size = " << size() << ": { ";
    std::copy(begin(), end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << " } " << std::endl;
}

// Predicates for algorithms
template <typename DataType>
struct is_equal_to
{
    is_equal_to(const DataType& expected) : m_expected(expected) {}
    bool
    operator()(const DataType& actual) const
    {
        return actual == m_expected;
    }

  private:
    DataType m_expected;
};

// Low-quality hash function, returns value between 0 and (1<<bits)-1
// Warning: low-order bits are quite predictable.
inline size_t
HashBits(size_t i, size_t bits)
{
    size_t mask = bits >= 8 * sizeof(size_t) ? ~size_t(0) : (size_t(1) << bits) - 1;
    return (424157 * i ^ 0x24aFa) & mask;
}

// Stateful unary op
template <typename T, typename U>
class Complement
{
    int32_t val;

  public:
    Complement(T v) : val(v) {}
    U
    operator()(const T& x) const
    {
        return U(val - x);
    }
};

// Tag used to prevent accidental use of converting constructor, even if use is explicit.
struct OddTag
{
};

class Sum;

// Type with limited set of operations.  Not default-constructible.
// Only available operator is "==".
// Typically used as value type in tests.
class Number
{
    int32_t value;
    friend class Add;
    friend class Sum;
    friend class IsMultiple;
    friend class Congruent;
    friend Sum
    operator+(const Sum& x, const Sum& y);

  public:
    Number(int32_t val, OddTag) : value(val) {}
    friend bool
    operator==(const Number& x, const Number& y)
    {
        return x.value == y.value;
    }
    friend std::ostream&
    operator<<(std::ostream& o, const Number& d)
    {
        return o << d.value;
    }
};

// Stateful predicate for Number.  Not default-constructible.
class IsMultiple
{
    long modulus;

  public:
    // True if x is multiple of modulus
    bool
    operator()(Number x) const
    {
        return x.value % modulus == 0;
    }
    IsMultiple(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful equivalence-class predicate for Number.  Not default-constructible.
class Congruent
{
    long modulus;

  public:
    // True if x and y have same remainder for the given modulus.
    // Note: this is not quite the same as "equivalent modulo modulus" when x and y have different
    // sign, but nonetheless AreCongruent is still an equivalence relationship, which is all
    // we need for testing.
    bool
    operator()(Number x, Number y) const
    {
        return x.value % modulus == y.value % modulus;
    }
    Congruent(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful reduction operation for Number
class Add
{
    long bias;

  public:
    explicit Add(OddTag) : bias(1) {}
    Number
    operator()(Number x, const Number& y)
    {
        return Number(x.value + y.value + (bias - 1), OddTag());
    }
};

// Class similar to Number, but has default constructor and +.
class Sum : public Number
{
  public:
    Sum() : Number(0, OddTag()) {}
    Sum(long x, OddTag) : Number(x, OddTag()) {}
    friend Sum
    operator+(const Sum& x, const Sum& y)
    {
        return Sum(x.value + y.value, OddTag());
    }
};

// Type with limited set of operations, which includes an associative but not commutative operation.
// Not default-constructible.
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
class MonoidElement
{
    size_t a, b;

  public:
    MonoidElement(size_t a_, size_t b_, OddTag) : a(a_), b(b_) {}
    friend bool
    operator==(const MonoidElement& x, const MonoidElement& y)
    {
        return x.a == y.a && x.b == y.b;
    }
    friend std::ostream&
    operator<<(std::ostream& o, const MonoidElement& x)
    {
        return o << "[" << x.a << ".." << x.b << ")";
    }
    friend class AssocOp;
};

// Stateful associative op for MonoidElement
// It's not really a monoid since the operation is not allowed for any two elements.
// But it's good enough for testing.
class AssocOp
{
    unsigned c;

  public:
    explicit AssocOp(OddTag) : c(5) {}
    MonoidElement
    operator()(const MonoidElement& x, const MonoidElement& y)
    {
        unsigned d = 5;
        EXPECT_EQ(d, c, "state lost");
        EXPECT_EQ(x.b, y.a, "commuted?");

        return MonoidElement(x.a, y.b, OddTag());
    }
};

// Multiplication of matrix is an associative but not commutative operation
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
template <typename T>
struct Matrix2x2
{
    T a[2][2];
    Matrix2x2() : a{{1, 0}, {0, 1}} {}
    Matrix2x2(T x, T y) : a{{0, x}, {x, y}} {}
#if !__PSTL_ICL_19_VC14_VC141_TEST_SCAN_RELEASE_BROKEN
    Matrix2x2(const Matrix2x2& m) : a{{m.a[0][0], m.a[0][1]}, {m.a[1][0], m.a[1][1]}} {}
    Matrix2x2&
    operator=(const Matrix2x2& m)
    {
        a[0][0] = m.a[0][0], a[0][1] = m.a[0][1], a[1][0] = m.a[1][0], a[1][1] = m.a[1][1];
        return *this;
    }
#endif
};

template <typename T>
bool
operator==(const Matrix2x2<T>& left, const Matrix2x2<T>& right)
{
    return left.a[0][0] == right.a[0][0] && left.a[0][1] == right.a[0][1] && left.a[1][0] == right.a[1][0] &&
           left.a[1][1] == right.a[1][1];
}

template <typename T>
Matrix2x2<T>
multiply_matrix(const Matrix2x2<T>& left, const Matrix2x2<T>& right)
{
    Matrix2x2<T> result;
    for (int32_t i = 0; i < 2; ++i)
    {
        for (int32_t j = 0; j < 2; ++j)
        {
            result.a[i][j] = left.a[i][0] * right.a[0][j] + left.a[i][1] * right.a[1][j];
        }
    }
    return result;
}

// Check that Intel(R) Threading Building Blocks header files are not used when parallel policies are off
#if !__PSTL_USE_PAR_POLICIES
#if defined(TBB_INTERFACE_VERSION)
#error The parallel backend is used while it should not (__PSTL_USE_PAR_POLICIES==0)
#endif
#endif

// Invoke op(policy,rest...) for each possible policy.
template <typename Op, typename... T>
void
invoke_on_all_policies(Op op, T&&... rest)
{
    using namespace __pstl::execution;

    // Try static execution policies
    invoke_on_all_iterator_types()(seq, op, std::forward<T>(rest)...);
    invoke_on_all_iterator_types()(unseq, op, std::forward<T>(rest)...);
#if __PSTL_USE_PAR_POLICIES
    invoke_on_all_iterator_types()(par, op, std::forward<T>(rest)...);
    invoke_on_all_iterator_types()(par_unseq, op, std::forward<T>(rest)...);
#endif

}

template <typename F>
struct NonConstAdapter
{
    F my_f;
    NonConstAdapter(const F& f) : my_f(f) {}

    template <typename... Types>
    auto
    operator()(Types&&... args) -> decltype(std::declval<F>().
                                            operator()(std::forward<Types>(args)...))
    {
        return my_f(std::forward<Types>(args)...);
    }
};

template <typename F>
NonConstAdapter<F>
non_const(const F& f)
{
    return NonConstAdapter<F>(f);
}

// Wrapper for types. It's need for counting of constructing and destructing objects
template <typename T>
class Wrapper
{
  public:
    Wrapper()
    {
        my_field = std::shared_ptr<T>(new T());
        ++my_count;
    }
    Wrapper(const T& input)
    {
        my_field = std::shared_ptr<T>(new T(input));
        ++my_count;
    }
    Wrapper(const Wrapper& input)
    {
        my_field = input.my_field;
        ++my_count;
    }
    Wrapper(Wrapper&& input)
    {
        my_field = input.my_field;
        input.my_field = nullptr;
        ++move_count;
    }
    Wrapper&
    operator=(const Wrapper& input)
    {
        my_field = input.my_field;
        return *this;
    }
    Wrapper&
    operator=(Wrapper&& input)
    {
        my_field = input.my_field;
        input.my_field = nullptr;
        ++move_count;
        return *this;
    }
    bool
    operator==(const Wrapper& input) const
    {
        return my_field == input.my_field;
    }
    bool
    operator<(const Wrapper& input) const
    {
        return *my_field < *input.my_field;
    }
    bool
    operator>(const Wrapper& input) const
    {
        return *my_field > *input.my_field;
    }
    friend std::ostream&
    operator<<(std::ostream& stream, const Wrapper& input)
    {
        return stream << *(input.my_field);
    }
    ~Wrapper()
    {
        --my_count;
        if (move_count > 0)
        {
            --move_count;
        }
    }
    T*
    get_my_field() const
    {
        return my_field.get();
    };
    static size_t
    Count()
    {
        return my_count;
    }
    static size_t
    MoveCount()
    {
        return move_count;
    }
    static void
    SetCount(const size_t& n)
    {
        my_count = n;
    }
    static void
    SetMoveCount(const size_t& n)
    {
        move_count = n;
    }

  private:
    static std::atomic<size_t> my_count;
    static std::atomic<size_t> move_count;
    std::shared_ptr<T> my_field;
};

template <typename T>
std::atomic<size_t> Wrapper<T>::my_count = {0};

template <typename T>
std::atomic<size_t> Wrapper<T>::move_count = {0};

template <typename InputIterator, typename T, typename BinaryOperation, typename UnaryOperation>
T
transform_reduce_serial(InputIterator first, InputIterator last, T init, BinaryOperation binary_op,
                        UnaryOperation unary_op) noexcept
{
    for (; first != last; ++first)
    {
        init = binary_op(init, unary_op(*first));
    }
    return init;
}

static const char*
done()
{
#if __PSTL_TEST_SUCCESSFUL_KEYWORD
    return "done";
#else
    return "passed";
#endif
}

// test_algo_basic_* functions are used to execute
// f on a very basic sequence of elements of type T.

// Should be used with unary predicate
template <typename T, typename F>
static void
test_algo_basic_single(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });

    invoke_on_all_policies(f, in.begin());
}

// Should be used with binary predicate
template <typename T, typename F>
static void
test_algo_basic_double(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });
    Sequence<T> out(N, [](size_t v) -> T { return T(v); });

    invoke_on_all_policies(f, in.begin(), out.begin());
}

template <typename Policy, typename F>
static void
invoke_if(Policy&& p, F f)
{
#if __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    __pstl::internal::invoke_if_not(__pstl::internal::allow_unsequenced<Policy>(), f);
#else
    f();
#endif
}

} /* namespace TestUtils */
