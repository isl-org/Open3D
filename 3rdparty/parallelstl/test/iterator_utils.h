// -*- C++ -*-
//===-- iterator_utils.h --------------------------------------------------===//
//
// Copyright (C) 2019 Intel Corporation
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

#ifndef __iterator_utils_H
#define __iterator_utils_H

#include "pstl_test_config.h"

// File contains common utilities for testing with different types of iterators
namespace TestUtils
{
// ForwardIterator is like type Iterator, but restricted to be a forward iterator.
// Only the forward iterator signatures that are necessary for tests are present.
// Post-increment in particular is deliberatly omitted since our templates should avoid using it
// because of efficiency considerations.
template <typename Iterator, typename IteratorTag>
class ForwardIterator
{
  public:
    typedef IteratorTag iterator_category;
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
    typedef typename std::iterator_traits<Iterator>::pointer pointer;
    typedef typename std::iterator_traits<Iterator>::reference reference;

  protected:
    Iterator my_iterator;
    typedef value_type element_type;

  public:
    ForwardIterator() = default;
    explicit ForwardIterator(Iterator i) : my_iterator(i) {}
    reference operator*() const { return *my_iterator; }
    Iterator operator->() const { return my_iterator; }
    ForwardIterator
    operator++()
    {
        ++my_iterator;
        return *this;
    }
    ForwardIterator operator++(int32_t)
    {
        auto retval = *this;
        my_iterator++;
        return retval;
    }
    friend bool
    operator==(const ForwardIterator& i, const ForwardIterator& j)
    {
        return i.my_iterator == j.my_iterator;
    }
    friend bool
    operator!=(const ForwardIterator& i, const ForwardIterator& j)
    {
        return i.my_iterator != j.my_iterator;
    }

    Iterator
    iterator() const
    {
        return my_iterator;
    }
};

template <typename Iterator, typename IteratorTag>
class BidirectionalIterator : public ForwardIterator<Iterator, IteratorTag>
{
    typedef ForwardIterator<Iterator, IteratorTag> base_type;

  public:
    BidirectionalIterator() = default;
    explicit BidirectionalIterator(Iterator i) : base_type(i) {}
    BidirectionalIterator(const base_type& i) : base_type(i.iterator()) {}

    BidirectionalIterator
    operator++()
    {
        ++base_type::my_iterator;
        return *this;
    }
    BidirectionalIterator
    operator--()
    {
        --base_type::my_iterator;
        return *this;
    }
    BidirectionalIterator operator++(int32_t)
    {
        auto retval = *this;
        base_type::my_iterator++;
        return retval;
    }
    BidirectionalIterator operator--(int32_t)
    {
        auto retval = *this;
        base_type::my_iterator--;
        return retval;
    }
};

//============================================================================
// Adapters for creating different types of iterators.
//
// In this block we implemented some adapters for creating differnet types of iterators.
// It's needed for extending the unit testing of Parallel STL algorithms.
// We have adapters for iterators with different tags (forward_iterator_tag, bidirectional_iterator_tag), reverse iterators.
// The input iterator should be const or non-const, non-reverse random access iterator.
// Iterator creates in "MakeIterator":
// firstly, iterator is "packed" by "IteratorTypeAdapter" (creating forward or bidirectional iterator)
// then iterator is "packed" by "ReverseAdapter" (if it's possible)
// So, from input iterator we may create, for example, reverse bidirectional iterator.
// "Main" functor for testing iterators is named "invoke_on_all_iterator_types".

// Base adapter
template <typename Iterator>
struct BaseAdapter
{
    typedef Iterator iterator_type;
    iterator_type
    operator()(Iterator it)
    {
        return it;
    }
};

// Check if the iterator is reverse iterator
// Note: it works only for iterators that created by std::reverse_iterator
template <typename NotReverseIterator>
struct isReverse : std::false_type
{
};

template <typename Iterator>
struct isReverse<std::reverse_iterator<Iterator>> : std::true_type
{
};

// Reverse adapter
template <typename Iterator, typename IsReverse>
struct ReverseAdapter
{
    typedef std::reverse_iterator<Iterator> iterator_type;
    iterator_type
    operator()(Iterator it)
    {
#if __PSTL_CPP14_MAKE_REVERSE_ITERATOR_PRESENT
        return std::make_reverse_iterator(it);
#else
        return iterator_type(it);
#endif
    }
};

// Non-reverse adapter
template <typename Iterator>
struct ReverseAdapter<Iterator, std::false_type> : BaseAdapter<Iterator>
{
};

// Iterator adapter by type (by default std::random_access_iterator_tag)
template <typename Iterator, typename IteratorTag>
struct IteratorTypeAdapter : BaseAdapter<Iterator>
{
};

// Iterator adapter for forward iterator
template <typename Iterator>
struct IteratorTypeAdapter<Iterator, std::forward_iterator_tag>
{
    typedef ForwardIterator<Iterator, std::forward_iterator_tag> iterator_type;
    iterator_type
    operator()(Iterator it)
    {
        return iterator_type(it);
    }
};

// Iterator adapter for bidirectional iterator
template <typename Iterator>
struct IteratorTypeAdapter<Iterator, std::bidirectional_iterator_tag>
{
    typedef BidirectionalIterator<Iterator, std::bidirectional_iterator_tag> iterator_type;
    iterator_type
    operator()(Iterator it)
    {
        return iterator_type(it);
    }
};

//For creating iterator with new type
template <typename InputIterator, typename IteratorTag, typename IsReverse>
struct MakeIterator
{
    typedef IteratorTypeAdapter<InputIterator, IteratorTag> IterByType;
    typedef ReverseAdapter<typename IterByType::iterator_type, IsReverse> ReverseIter;

    typename ReverseIter::iterator_type
    operator()(InputIterator it)
    {
        return ReverseIter()(IterByType()(it));
    }
};

// Useful constant variables
constexpr std::size_t GuardSize = 5;
constexpr std::size_t sizeLimit = 1000;

template <typename Iter, typename Void = void> // local iterator_traits for non-iterators
struct iterator_traits_
{
};

template <typename Iter> // For iterators
struct iterator_traits_<Iter,
                        typename std::enable_if<!std::is_void<typename Iter::iterator_category>::value, void>::type>
{
    typedef typename Iter::iterator_category iterator_category;
};

template <typename T> // For pointers
struct iterator_traits_<T*>
{
    typedef std::random_access_iterator_tag iterator_category;
};

// is iterator Iter has tag Tag
template <typename Iter, typename Tag>
using is_same_iterator_category = std::is_same<typename iterator_traits_<Iter>::iterator_category, Tag>;

// if we run with reverse or const iterators we shouldn't test the large range
template <typename IsReverse, typename IsConst>
struct invoke_if_
{
    template <typename Op, typename... Rest>
    void
    operator()(bool is_allow, Op op, Rest&&... rest)
    {
        if (is_allow)
            op(std::forward<Rest>(rest)...);
    }
};
template <>
struct invoke_if_<std::false_type, std::false_type>
{
    template <typename Op, typename... Rest>
    void
    operator()(bool is_allow, Op op, Rest&&... rest)
    {
        op(std::forward<Rest>(rest)...);
    }
};

// Base non_const_wrapper struct. It is used to distinguish non_const testcases
// from a regular one. For non_const testcases only compilation is checked.
struct non_const_wrapper
{
};

// Generic wrapper to specify iterator type to execute callable Op on.
// The condition can be either positive(Op is executed only with IteratorTag)
// or negative(Op is executed with every type of iterators except IteratorTag)
template <typename Op, typename IteratorTag, bool IsPositiveCondition = true>
struct non_const_wrapper_tagged : non_const_wrapper
{
    template <typename Policy, typename Iterator>
    typename std::enable_if<IsPositiveCondition == is_same_iterator_category<Iterator, IteratorTag>::value, void>::type
    operator()(Policy&& exec, Iterator iter)
    {
        Op()(exec, iter);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator>
    typename std::enable_if<IsPositiveCondition == is_same_iterator_category<OutputIterator, IteratorTag>::value,
                            void>::type
    operator()(Policy&& exec, InputIterator input_iter, OutputIterator out_iter)
    {
        Op()(exec, input_iter, out_iter);
    }

    template <typename Policy, typename Iterator>
    typename std::enable_if<IsPositiveCondition != is_same_iterator_category<Iterator, IteratorTag>::value, void>::type
    operator()(Policy&& exec, Iterator iter)
    {
    }

    template <typename Policy, typename InputIterator, typename OutputIterator>
    typename std::enable_if<IsPositiveCondition != is_same_iterator_category<OutputIterator, IteratorTag>::value,
                            void>::type
    operator()(Policy&& exec, InputIterator input_iter, OutputIterator out_iter)
    {
    }
};

// These run_for_* structures specify with which types of iterators callable object Op
// should be executed.
template <typename Op>
struct run_for_rnd : non_const_wrapper_tagged<Op, std::random_access_iterator_tag>
{
};

template <typename Op>
struct run_for_rnd_bi : non_const_wrapper_tagged<Op, std::forward_iterator_tag, false>
{
};

template <typename Op>
struct run_for_rnd_fw : non_const_wrapper_tagged<Op, std::bidirectional_iterator_tag, false>
{
};

// Invoker for different types of iterators.
template <typename IteratorTag, typename IsReverse>
struct iterator_invoker
{
    template <typename Iterator>
    using make_iterator = MakeIterator<Iterator, IteratorTag, IsReverse>;
    template <typename Iterator>
    using IsConst = typename std::is_const<
        typename std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>::type>::type;
    template <typename Iterator>
    using invoke_if = invoke_if_<IsReverse, IsConst<Iterator>>;

    // A single iterator version which is used for non_const testcases
    template <typename Policy, typename Op, typename Iterator>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
                                std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, Iterator iter)
    {
        op(std::forward<Policy>(exec), make_iterator<Iterator>()(iter));
    }

    // A version with 2 iterators which is used for non_const testcases
    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value &&
                                std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator input_iter, OutputIterator out_iter)
    {
        op(std::forward<Policy>(exec), make_iterator<InputIterator>()(input_iter),
           make_iterator<OutputIterator>()(out_iter));
    }

    template <typename Policy, typename Op, typename Iterator, typename Size, typename... Rest>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
    operator()(Policy&& exec, Op op, Iterator begin, Size n, Rest&&... rest)
    {
        invoke_if<Iterator>()(n <= sizeLimit, op, exec, make_iterator<Iterator>()(begin), n,
                              std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename Iterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
                                !std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, Iterator inputBegin, Iterator inputEnd, Rest&&... rest)
    {
        invoke_if<Iterator>()(std::distance(inputBegin, inputEnd) <= sizeLimit, op, exec,
                              make_iterator<Iterator>()(inputBegin), make_iterator<Iterator>()(inputEnd),
                              std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin,
               Rest&&... rest)
    {
        invoke_if<InputIterator>()(std::distance(inputBegin, inputEnd) <= sizeLimit, op, exec,
                                   make_iterator<InputIterator>()(inputBegin), make_iterator<InputIterator>()(inputEnd),
                                   make_iterator<OutputIterator>()(outputBegin), std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin,
               OutputIterator outputEnd, Rest&&... rest)
    {
        invoke_if<InputIterator>()(std::distance(inputBegin, inputEnd) <= sizeLimit, op, exec,
                                   make_iterator<InputIterator>()(inputBegin), make_iterator<InputIterator>()(inputEnd),
                                   make_iterator<OutputIterator>()(outputBegin),
                                   make_iterator<OutputIterator>()(outputEnd), std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator1 inputBegin1, InputIterator1 inputEnd1, InputIterator2 inputBegin2,
               InputIterator2 inputEnd2, OutputIterator outputBegin, OutputIterator outputEnd, Rest&&... rest)
    {
        invoke_if<InputIterator1>()(
            std::distance(inputBegin1, inputEnd1) <= sizeLimit, op, exec, make_iterator<InputIterator1>()(inputBegin1),
            make_iterator<InputIterator1>()(inputEnd1), make_iterator<InputIterator2>()(inputBegin2),
            make_iterator<InputIterator2>()(inputEnd2), make_iterator<OutputIterator>()(outputBegin),
            make_iterator<OutputIterator>()(outputEnd), std::forward<Rest>(rest)...);
    }
};

// Invoker for reverse iterators only
// Note: if we run with reverse iterators we shouldn't test the large range
template <typename IteratorTag>
struct iterator_invoker<IteratorTag, /* IsReverse = */ std::true_type>
{

    template <typename Iterator>
    using make_iterator = MakeIterator<Iterator, IteratorTag, std::true_type>;

    // A single iterator version which is used for non_const testcases
    template <typename Policy, typename Op, typename Iterator>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
                                std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, Iterator iter)
    {
        op(std::forward<Policy>(exec), make_iterator<Iterator>()(iter));
    }

    // A version with 2 iterators which is used for non_const testcases
    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value &&
                                std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator input_iter, OutputIterator out_iter)
    {
        op(std::forward<Policy>(exec), make_iterator<InputIterator>()(input_iter),
           make_iterator<OutputIterator>()(out_iter));
    }

    template <typename Policy, typename Op, typename Iterator, typename Size, typename... Rest>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
    operator()(Policy&& exec, Op op, Iterator begin, Size n, Rest&&... rest)
    {
        if (n <= sizeLimit)
            op(exec, make_iterator<Iterator>()(begin + n), n, std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename Iterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
                                !std::is_base_of<non_const_wrapper, Op>::value,
                            void>::type
    operator()(Policy&& exec, Op op, Iterator inputBegin, Iterator inputEnd, Rest&&... rest)
    {
        if (std::distance(inputBegin, inputEnd) <= sizeLimit)
            op(exec, make_iterator<Iterator>()(inputEnd), make_iterator<Iterator>()(inputBegin),
               std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin,
               Rest&&... rest)
    {
        if (std::distance(inputBegin, inputEnd) <= sizeLimit)
            op(exec, make_iterator<InputIterator>()(inputEnd), make_iterator<InputIterator>()(inputBegin),
               make_iterator<OutputIterator>()(outputBegin + (inputEnd - inputBegin)), std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator, typename OutputIterator, typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin,
               OutputIterator outputEnd, Rest&&... rest)
    {
        if (std::distance(inputBegin, inputEnd) <= sizeLimit)
            op(exec, make_iterator<InputIterator>()(inputEnd), make_iterator<InputIterator>()(inputBegin),
               make_iterator<OutputIterator>()(outputEnd), make_iterator<OutputIterator>()(outputBegin),
               std::forward<Rest>(rest)...);
    }

    template <typename Policy, typename Op, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename... Rest>
    typename std::enable_if<is_same_iterator_category<OutputIterator, std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, Op op, InputIterator1 inputBegin1, InputIterator1 inputEnd1, InputIterator2 inputBegin2,
               InputIterator2 inputEnd2, OutputIterator outputBegin, OutputIterator outputEnd, Rest&&... rest)
    {
        if (std::distance(inputBegin1, inputEnd1) <= sizeLimit)
            op(exec, make_iterator<InputIterator1>()(inputEnd1), make_iterator<InputIterator1>()(inputBegin1),
               make_iterator<InputIterator2>()(inputEnd2), make_iterator<InputIterator2>()(inputBegin2),
               make_iterator<OutputIterator>()(outputEnd), make_iterator<OutputIterator>()(outputBegin),
               std::forward<Rest>(rest)...);
    }
};

// We can't create reverse iterator from forward iterator
template <>
struct iterator_invoker<std::forward_iterator_tag, /*isReverse=*/std::true_type>
{
    template <typename... Rest>
    void
    operator()(Rest&&... rest)
    {
    }
};

template <typename IsReverse>
struct reverse_invoker
{
    template <typename... Rest>
    void
    operator()(Rest&&... rest)
    {
        // Random-access iterator
        iterator_invoker<std::random_access_iterator_tag, IsReverse>()(std::forward<Rest>(rest)...);

        // Forward iterator
        iterator_invoker<std::forward_iterator_tag, IsReverse>()(std::forward<Rest>(rest)...);

        // Bidirectional iterator
        iterator_invoker<std::bidirectional_iterator_tag, IsReverse>()(std::forward<Rest>(rest)...);
    }
};

struct invoke_on_all_iterator_types
{
    template <typename... Rest>
    void
    operator()(Rest&&... rest)
    {
        reverse_invoker</* IsReverse = */ std::false_type>()(std::forward<Rest>(rest)...);
        reverse_invoker</* IsReverse = */ std::true_type>()(std::forward<Rest>(rest)...);
    }
};
} /* namespace TestUtils */
#endif /* __iterator_utils_H */
