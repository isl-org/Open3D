/*
Copyright (c) 2011, Michael Kazhdan and Ming Chuang
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include <string.h>
#include <stdio.h>
#include <emmintrin.h>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif // _WIN32
#include <stddef.h>
#include <type_traits>
#include <cstddef>

template< class C >
class Array
{
	template< class D > friend class Array;
	void _assertBounds( std::ptrdiff_t idx ) const
	{
		if( idx<min || idx>=max )
		{
			StackTracer::Trace();
			ERROR_OUT( "Array index out-of-bounds: " , min , " <= " , idx , " < " , max );
		}
	}
protected:
	C *data , *_data;
	std::ptrdiff_t min , max;

public:
	std::ptrdiff_t minimum( void ) const { return min; }
	std::ptrdiff_t maximum( void ) const { return max; }

	static inline Array New( size_t size , const char* name=NULL )
	{
		Array a;
		a._data = a.data = new C[size];
		a.min = 0;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Casting unsigned to signed" )
#endif // SHOW_WARNINGS
		a.max = (std::ptrdiff_t)size;
		return a;
	}
	static inline Array Alloc( size_t size , bool clear , const char* name=NULL )
	{
		Array a;
		a._data = a.data = ( C* ) malloc( size * sizeof( C ) );
		if( clear ) memset( a.data ,  0 , size * sizeof( C ) );
		a.min = 0;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Casting unsigned to signed" )
#endif // SHOW_WARNINGS
		a.max = (std::ptrdiff_t)size;
		return a;
	}

	static inline Array AlignedAlloc( size_t size , size_t alignment , bool clear , const char* name=NULL )
	{
		Array a;
		a.data = ( C* ) aligned_malloc( sizeof(C) * size , alignment );
		a._data = ( C* )( ( ( void** )a.data )[-1] );
		if( clear ) memset( a.data ,  0 , size * sizeof( C ) );
		a.min = 0;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Casting unsigned to signed" )
#endif // SHOW_WARNINGS
		a.max = (std::ptrdiff_t)size;
		return a;
	}

	static inline Array ReAlloc( Array& a , size_t size , bool clear , const char* name=NULL )
	{
		Array _a;
		_a._data = _a.data = ( C* ) realloc( a.data , size * sizeof( C ) );
		if( clear ) memset( _a.data ,  0 , size * sizeof( C ) );
		a._data = NULL;
		_a.min = 0;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Casting unsigned to signed" )
#endif // SHOW_WARNINGS
		_a.max = (std::ptrdiff_t)size;
		return _a;
	}

	Array( void )
	{
		data = _data = NULL;
		min = max = 0;
	}
	template< class D >
	Array( Array< D >& a )
	{
		_data = NULL;
		if( !a )
		{
			data =  NULL;
			min = max = 0;
		}
		else
		{
			// [WARNING] Changing szC and szD to size_t causes some really strange behavior.
			std::ptrdiff_t szC = (std::ptrdiff_t)sizeof( C );
			std::ptrdiff_t szD = (std::ptrdiff_t)sizeof( D );
			data = (C*)a.data;
			min = ( a.minimum() * szD ) / szC;
			max = ( a.maximum() * szD ) / szC;
			if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD ) ERROR_OUT( "Could not convert array [ " , a.minimum() , " , " , a.maximum() , " ] * " , szD , " => [ " , min , " , " , max , " ] * " , szC );
		}
	}
	static Array FromPointer( C* data , std::ptrdiff_t max )
	{
		Array a;
		a._data = NULL;
		a.data = data;
		a.min = 0;
		a.max = max;
		return a;
	}
	static Array FromPointer( C* data , std::ptrdiff_t min , std::ptrdiff_t max )
	{
		Array a;
		a._data = NULL;
		a.data = data;
		a.min = min;
		a.max = max;
		return a;
	}
	inline bool operator == ( const Array< C >& a ) const { return data==a.data; }
	inline bool operator != ( const Array< C >& a ) const { return data!=a.data; }
	inline bool operator == ( const C* c ) const { return data==c; }
	inline bool operator != ( const C* c ) const { return data!=c; }
	inline C* operator -> ( void )
	{
		_assertBounds( 0 );
		return data;
	}
	inline const C* operator -> ( void ) const
	{
		_assertBounds( 0 );
		return data;
	}
	inline C &operator * ( void )
	{
		_assertBounds( 0 );
		return data[0];
	}
	inline const C &operator * ( void ) const
	{
		_assertBounds( 0 );
		return data[0];
	}
	template< typename Int >
	inline C& operator[]( Int idx )
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		_assertBounds( idx );
		return data[idx];
	}

	template< typename Int >
	inline const C& operator[]( Int idx ) const
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		_assertBounds( idx );
		return data[idx];
	}

	template< typename Int >
	inline Array operator + ( Int idx ) const
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		Array a;
		a._data = _data;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	template< typename Int >
	inline Array& operator += ( Int idx  )
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	template< typename Int > Array  operator -  ( Int idx ) const { return (*this) +  (-idx); }
	template< typename Int > Array& operator -= ( Int idx )       { return (*this) += (-idx); }

	inline Array& operator ++ ( void  ) { return (*this) += 1; }
	inline Array operator++( int ){ Array< C > temp = (*this) ; (*this) +=1 ; return temp; }
	inline Array operator--( int ){ Array< C > temp = (*this) ; (*this) -=1 ; return temp; }
	std::ptrdiff_t operator - ( const Array& a ) const { return data - a.data; }

	void Free( void )
	{
		if( _data )
		{
			free( _data );
		}
		(*this) = Array( );
	}
	void Delete( void )
	{
		if( _data )
		{
			delete[] _data;
		}
		(*this) = Array( );
	}
	C* pointer( void ){ return data; }
	const C* pointer( void ) const { return data; }
	bool operator !( void ) const { return data==NULL; }
	operator bool( ) const { return data!=NULL; }
};

template< class C >
class ConstArray
{
	template< class D > friend class ConstArray;
	void _assertBounds( std::ptrdiff_t idx ) const
	{
		if( idx<min || idx>=max ) ERROR_OUT( "ConstArray index out-of-bounds: " , min , " <= " , idx , " < " , max );
	}
protected:
	const C *data;
	std::ptrdiff_t min , max;
public:
	std::ptrdiff_t minimum( void ) const { return min; }
	std::ptrdiff_t maximum( void ) const { return max; }

	inline ConstArray( void )
	{
		data = NULL;
		min = max = 0;
	}
	inline ConstArray( const Array< C >& a )
	{
		// [WARNING] Changing szC and szD to size_t causes some really strange behavior.
		data = ( const C* )a.pointer( );
		min = a.minimum();
		max = a.maximum();
	}
	template< class D >
	inline ConstArray( const Array< D >& a )
	{
		// [WARNING] Changing szC and szD to size_t causes some really strange behavior.
		std::ptrdiff_t szC = (std::ptrdiff_t)sizeof( C );
		std::ptrdiff_t szD = (std::ptrdiff_t)sizeof( D );
		data = ( const C* )a.pointer( );
		min = ( a.minimum() * szD ) / szC;
		max = ( a.maximum() * szD ) / szC;
		if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD ) ERROR_OUT( "Could not convert const array [ " , a.minimum() , " , " , a.maximum() , " ] * " , szD , " => [ " , min , " , " , max , " ] * " , szC );
	}
	template< class D >
	inline ConstArray( const ConstArray< D >& a )
	{
		// [WARNING] Chaning szC and szD to size_t causes some really strange behavior.
		std::ptrdiff_t szC = (std::ptrdiff_t)sizeof( C );
		std::ptrdiff_t szD = (std::ptrdiff_t)sizeof( D );
		data = ( const C*)a.pointer( );
		min = ( a.minimum() * szD ) / szC;
		max = ( a.maximum() * szD ) / szC;
		if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD ) ERROR_OUT( "Could not convert array [ " , a.minimum() , " , " , a.maximum() , " ] * " , szD , " => [ " , min , " , " , max , " ] * " , szC );
	}
	static ConstArray FromPointer( const C* data , std::ptrdiff_t max )
	{
		ConstArray a;
		a.data = data;
		a.min = 0;
		a.max = max;
		return a;
	}

	static ConstArray FromPointer( const C* data , std::ptrdiff_t min , std::ptrdiff_t max )
	{
		ConstArray a;
		a.data = data;
		a.min = min;
		a.max = max;
		return a;
	}

	inline bool operator == ( const ConstArray< C >& a ) const { return data==a.data; }
	inline bool operator != ( const ConstArray< C >& a ) const { return data!=a.data; }
	inline bool operator == ( const C* c ) const { return data==c; }
	inline bool operator != ( const C* c ) const { return data!=c; }
	inline const C* operator -> ( void )
	{
		_assertBounds( 0 );
		return data;
	}
	inline const C &operator * ( void )
	{
		_assertBounds( 0 );
		return data[0];
	}
	template< typename Int >
	inline const C& operator[]( Int idx ) const
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		_assertBounds( idx );
		return data[idx];
	}
	template< typename Int >
	inline ConstArray operator + ( Int idx ) const
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		ConstArray a;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	template< typename  Int >
	inline ConstArray& operator += ( Int idx  )
	{
		static_assert( std::is_integral< Int >::value , "Integral type expected" );
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	template< typename Int > ConstArray  operator -  ( Int idx ) const { return (*this) +  (-idx); }
	template< typename Int > ConstArray& operator -= ( Int idx )       { return (*this) += (-idx); }
	inline ConstArray& operator ++ ( void ) { return (*this) += 1; }
	inline ConstArray operator++( int ){ ConstArray< C > temp = (*this) ; (*this) +=1 ; return temp; }
	inline ConstArray operator--( int ){ ConstArray< C > temp = (*this) ; (*this) -=1 ; return temp; }
	std::ptrdiff_t operator - ( const ConstArray& a ) const { return data - a.data; }
	std::ptrdiff_t operator - ( const Array< C >& a ) const { return data - a.pointer(); }

	const C* pointer( void ) const { return data; }
	bool operator !( void ) { return data==NULL; }
	operator bool( ) { return data!=NULL; }
};

template< class C >
Array< C > memcpy( Array< C > destination , const void* source , size_t size )
{
	if( size>destination.maximum()*sizeof(C) ) ERROR_OUT( "Size of copy exceeds destination maximum: " , size , " > " , destination.maximum()*sizeof( C ) );
	if( size ) memcpy( &destination[0] , source , size );
	return destination;
}
template< class C , class D >
Array< C > memcpy( Array< C > destination , Array< D > source , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) ) ERROR_OUT( "Size of copy exceeds destination maximum: " , size , " > " , destination.maximum()*sizeof( C ) );
	if( size>source.maximum()*sizeof( D ) ) ERROR_OUT( "Size of copy exceeds source maximum: " , size , " > " , source.maximum()*sizeof( D ) );
	if( size ) memcpy( &destination[0] , &source[0] , size );
	return destination;
}
template< class C , class D >
Array< C > memcpy( Array< C > destination , ConstArray< D > source , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) ) ERROR_OUT( "Size of copy exceeds destination maximum: " , size , " > " , destination.maximum()*sizeof( C ) );
	if( size>source.maximum()*sizeof( D ) ) ERROR_OUT( "Size of copy exceeds source maximum: " , size , " > " , source.maximum()*sizeof( D ) );
	if( size ) memcpy( &destination[0] , &source[0] , size );
	return destination;
}
template< class D >
void* memcpy( void* destination , Array< D > source , size_t size )
{
	if( size>source.maximum()*sizeof( D ) ) ERROR_OUT( "Size of copy exceeds source maximum: " , size , " > " , source.maximum()*sizeof( D ) );
	if( size ) memcpy( destination , &source[0] , size );
	return destination;
}
template< class D >
void* memcpy( void* destination , ConstArray< D > source , size_t size )
{
	if( size>source.maximum()*sizeof( D ) ) ERROR_OUT( "Size of copy exceeds source maximum: " , size , " > " , source.maximum()*sizeof( D ) );
	if( size ) memcpy( destination , &source[0] , size );
	return destination;
}
template< class C >
Array< C > memset( Array< C > destination , int value , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) ) ERROR_OUT( "Size of set exceeds destination maximum: " , size , " > " , destination.maximum()*sizeof( C ) );
	if( size ) memset( &destination[0] , value , size );
	return destination;
}

template< class C >
size_t fread( Array< C > destination , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>destination.maximum()*sizeof( C ) ) ERROR_OUT( "Size of read exceeds source maximum: " , count*eSize , " > " , destination.maximum()*sizeof( C ) );
	return fread( &destination[0] , eSize , count , fp );
}
template< class C >
size_t fwrite( Array< C > source , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>source.maximum()*sizeof( C ) ) ERROR_OUT( "Size of write exceeds source maximum: " , count*eSize , " > " , source.maximum()*sizeof( C ) );
	return fwrite( &source[0] , eSize , count , fp );
}
template< class C >
size_t fwrite( ConstArray< C > source , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>source.maximum()*sizeof( C ) ) ERROR_OUT( "Size of write exceeds source maximum: " , count*eSize , " > " , source.maximum()*sizeof( C ) );
	return fwrite( &source[0] , eSize , count , fp );
}
template< class C >
void qsort( Array< C > base , size_t numElements , size_t elementSize , int (*compareFunction)( const void* , const void* ) )
{
	if( sizeof(C)!=elementSize ) ERROR_OUT( "Element sizes differ: " , sizeof(C) , " != " , elementSize );
	if( base.minimum()>0 || base.maximum()<numElements ) ERROR_OUT( "Array access out of bounds: " , base.minimum() , " <= 0 <= " , base.maximum() , " <= " , numElements );
	qsort( base.pointer() , numElements , elementSize , compareFunction );
}
