/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_GENERAL_H_
#define FLANN_GENERAL_H_

#include "defines.h"
#include <stdexcept>
#include <cassert>
#include <limits.h>

namespace flann
{

class FLANNException : public std::runtime_error
{
public:
    FLANNException(const char* message) : std::runtime_error(message) { }

    FLANNException(const std::string& message) : std::runtime_error(message) { }
};


template <typename T>
struct flann_datatype_value
{
	static const flann_datatype_t value = FLANN_NONE;
};

template<>
struct flann_datatype_value<char>
{
	static const flann_datatype_t value = FLANN_INT8;
};

template<>
struct flann_datatype_value<short>
{
	static const flann_datatype_t value = FLANN_INT16;
};

template<>
struct flann_datatype_value<int>
{
	static const flann_datatype_t value = FLANN_INT32;
};

#ifdef LLONG_MAX
template<>
struct flann_datatype_value<long long>
{
	static const flann_datatype_t value = FLANN_INT64;
};
#endif

template<>
struct flann_datatype_value<unsigned char>
{
	static const flann_datatype_t value = FLANN_UINT8;
};

template<>
struct flann_datatype_value<unsigned short>
{
	static const flann_datatype_t value = FLANN_UINT16;
};

template<>
struct flann_datatype_value<unsigned int>
{
	static const flann_datatype_t value = FLANN_UINT32;
};

#ifdef ULLONG_MAX
template<>
struct flann_datatype_value<unsigned long long>
{
	static const flann_datatype_t value = FLANN_UINT64;
};
#endif


template<>
struct flann_datatype_value<float>
{
	static const flann_datatype_t value = FLANN_FLOAT32;
};

template<>
struct flann_datatype_value<double>
{
	static const flann_datatype_t value = FLANN_FLOAT64;
};



template <flann_datatype_t datatype>
struct flann_datatype_type
{
	typedef void type;
};

template<>
struct flann_datatype_type<FLANN_INT8>
{
	typedef char type;
};

template<>
struct flann_datatype_type<FLANN_INT16>
{
	typedef short type;
};

template<>
struct flann_datatype_type<FLANN_INT32>
{
	typedef int type;
};

#ifdef LLONG_MAX
template<>
struct flann_datatype_type<FLANN_INT64>
{
	typedef long long type;
};
#endif

template<>
struct flann_datatype_type<FLANN_UINT8>
{
	typedef unsigned char type;
};


template<>
struct flann_datatype_type<FLANN_UINT16>
{
	typedef unsigned short type;
};

template<>
struct flann_datatype_type<FLANN_UINT32>
{
	typedef unsigned int type;
};

#ifdef ULLONG_MAX
template<>
struct flann_datatype_type<FLANN_UINT64>
{
	typedef unsigned long long type;
};
#endif

template<>
struct flann_datatype_type<FLANN_FLOAT32>
{
	typedef float type;
};

template<>
struct flann_datatype_type<FLANN_FLOAT64>
{
	typedef double type;
};


inline size_t flann_datatype_size(flann_datatype_t type)
{
	switch (type) {
	case FLANN_INT8:
		return sizeof(flann_datatype_type<FLANN_INT8>::type);
	case FLANN_INT16:
		return sizeof(flann_datatype_type<FLANN_INT16>::type);
	case FLANN_INT32:
		return sizeof(flann_datatype_type<FLANN_INT32>::type);
	case FLANN_INT64:
		return sizeof(flann_datatype_type<FLANN_INT64>::type);
	case FLANN_UINT8:
		return sizeof(flann_datatype_type<FLANN_UINT8>::type);
	case FLANN_UINT16:
		return sizeof(flann_datatype_type<FLANN_UINT16>::type);
	case FLANN_UINT32:
		return sizeof(flann_datatype_type<FLANN_UINT32>::type);
	case FLANN_UINT64:
		return sizeof(flann_datatype_type<FLANN_UINT64>::type);
	case FLANN_FLOAT32:
		return sizeof(flann_datatype_type<FLANN_FLOAT32>::type);
	case FLANN_FLOAT64:
		return sizeof(flann_datatype_type<FLANN_FLOAT64>::type);
	default:
		return 0;
	}
}

}


#endif  /* FLANN_GENERAL_H_ */
