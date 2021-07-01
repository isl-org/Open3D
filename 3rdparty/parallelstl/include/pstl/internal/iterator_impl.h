// -*- C++ -*-
//===-- iterator_impl.h ---------------------------------------------------===//
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

#ifndef __PSTL_iterator_impl_H
#define __PSTL_iterator_impl_H

#include <iterator>
#include <tuple>

#if __PSTL_CPP14_INTEGER_SEQUENCE_PRESENT
#    include <utility>
#endif

#include "pstl_config.h"

namespace __pstl
{
namespace __internal
{

#if __PSTL_CPP14_INTEGER_SEQUENCE_PRESENT

using std::index_sequence;
using std::make_index_sequence;

#else

template<std::size_t... S> class index_sequence {};

template<std::size_t N, std::size_t... S>
struct make_index_sequence_impl : make_index_sequence_impl < N - 1, N - 1, S... > {};

template<std::size_t... S>
struct make_index_sequence_impl <0, S...> {
    using type = index_sequence<S...>;
};

template<std::size_t N>
using make_index_sequence = typename __pstl::__internal::make_index_sequence_impl<N>::type;

#endif /* __PSTL_CPP14_INTEGER_SEQUENCE_PRESENT */

template<size_t N>
struct zip_forward_iterator_util {
    template<typename TupleType>
    static void increment(TupleType& it) {
        ++std::get<N - 1>(it);
        zip_forward_iterator_util<N - 1>::increment(it);
    }
};

template<>
struct  zip_forward_iterator_util<0> {
    template<typename TupleType>
    static void increment(TupleType&) {};
};


template <typename TupleReturnType>
struct make_references {
    template <typename TupleType, std::size_t... Is>
    TupleReturnType operator()(const TupleType& t, index_sequence<Is...>) {
        return std::tie(*std::get<Is>(t)...);
    }
};

//zip_iterator version for forward iterator
//== and != comparison is performed only on the first element of the tuple
template <typename ...Types>
class zip_forward_iterator {
    static const std::size_t num_types = sizeof...(Types);
    typedef typename std::tuple<Types...> it_types;
public:
    typedef typename std::make_signed<std::size_t>::type difference_type;
    typedef std::tuple<typename std::iterator_traits<Types>::value_type...> value_type;
    typedef std::tuple<typename std::iterator_traits<Types>::reference...> reference;
    typedef std::tuple<typename std::iterator_traits<Types>::pointer...> pointer;
    typedef std::forward_iterator_tag iterator_category;

    zip_forward_iterator() : my_it() {}
    explicit zip_forward_iterator(Types ...args) : my_it(std::make_tuple(args...)) {}
    zip_forward_iterator(const zip_forward_iterator& input) : my_it(input.my_it) {}
    zip_forward_iterator& operator=(const zip_forward_iterator& input) {
        my_it = input.my_it;
        return *this;
    }

    reference operator*() const {
        return make_references<reference>()(my_it, make_index_sequence<num_types>());
    }

    zip_forward_iterator& operator++() {
        zip_forward_iterator_util<num_types>::increment(my_it);
        return *this;
    }
    zip_forward_iterator operator++(int) {
        zip_forward_iterator it(*this);
        ++(*this);
        return it;
    }

    bool operator==(const zip_forward_iterator& it) const {
        return std::get<0>(my_it) == std::get<0>(it.my_it);
    }
    bool operator!=(const zip_forward_iterator& it) const {
        return !(*this == it);
    }

    it_types base() const {
        return my_it;
    }

private:
    it_types my_it;
};

} // namespace __internal
} // namespace __pstl

#endif /* __PSTL_iterator_impl_H */
