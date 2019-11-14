/*
Copyright (c) 2016, Michael Kazhdan
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

#ifndef WINDOW_INCLUDED
#define WINDOW_INCLUDED

#include <functional>
#include "MyMiscellany.h"
#include "Allocator.h"
#include "Array.h"

//////////////////////////////////////////////////////////
// Some basic functionality for integer parameter packs //
//////////////////////////////////////////////////////////

// A wrapper class for passing unsigned integer parameter packs
template< unsigned int  ... Values > struct UIntPack{};
template< unsigned int _Value , unsigned int ... _Values > struct UIntPack< _Value , _Values ... >
{
	static const unsigned int First = _Value;
	typedef UIntPack< _Values ... > Rest;

	static const unsigned int Size = 1 + sizeof ... ( _Values );
	template< unsigned int __Value > using  Append = UIntPack< _Value , _Values ... , __Value >;
	template< unsigned int __Value > using Prepend = UIntPack< __Value , _Value , _Values ... >;
	static const unsigned int Values[];
	static constexpr unsigned int Min( void ){ return _Value < Rest::Min() ? _Value : Rest::Min(); }
	static constexpr unsigned int Max( void ){ return _Value > Rest::Max() ? _Value : Rest::Max(); }

	template< typename T > struct Plus{};
	template< typename T > struct Minus{};
	template< typename T > struct Compare{};
	template< unsigned int __Value , unsigned int ... __Values > struct Plus < UIntPack< __Value , __Values ... > >{ typedef typename Rest::template Plus < UIntPack< __Values ... > >::type::template Prepend< _Value + __Value > type; };
	template< unsigned int __Value , unsigned int ... __Values > struct Minus< UIntPack< __Value , __Values ... > >{ typedef typename Rest::template Minus< UIntPack< __Values ... > >::type::template Prepend< _Value - __Value > type; };
	template< unsigned int __Value , unsigned int ... __Values > struct Compare< UIntPack< __Value , __Values ... > >
	{
		static const bool              Equal = _Value==__Value && Rest::template Compare< UIntPack< __Values ... > >::             Equal;
		static const bool           NotEqual = _Value!=__Value || Rest::template Compare< UIntPack< __Values ... > >::          NotEqual;
		static const bool    LessThan        = _Value< __Value && Rest::template Compare< UIntPack< __Values ... > >::   LessThan       ;
		static const bool    LessThanOrEqual = _Value<=__Value && Rest::template Compare< UIntPack< __Values ... > >::   LessThanOrEqual;
		static const bool GreaterThan        = _Value> __Value && Rest::template Compare< UIntPack< __Values ... > >::GreaterThan       ;
		static const bool GreaterThanOrEqual = _Value>=__Value && Rest::template Compare< UIntPack< __Values ... > >::GreaterThanOrEqual;
	};

	static void Print( FILE* fp=stdout , bool leadingSpace=false ){ if( leadingSpace ) fprintf( fp , " " ) ; fprintf( fp , "%d" , _Value ) ; Rest::Print( fp , true ); }

	template< unsigned int I > constexpr static typename std::enable_if< I==0 , unsigned int >::type Get( void ){ return _Value; }
	template< unsigned int I > constexpr static typename std::enable_if< I!=0 , unsigned int >::type Get( void ){ return Rest::template Get< I-1 >(); }

	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator <  ( UIntPack< __Value , __Values ... > ) const { return _Value< __Value && Rest()< UIntPack< __Values ... >(); }
	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator <= ( UIntPack< __Value , __Values ... > ) const { return _Value<=__Value && Rest()<=UIntPack< __Values ... >(); }
	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator >  ( UIntPack< __Value , __Values ... > ) const { return _Value> __Value && Rest()> UIntPack< __Values ... >(); }
	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator >= ( UIntPack< __Value , __Values ... > ) const { return _Value>=__Value && Rest()>=UIntPack< __Values ... >(); }
	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator == ( UIntPack< __Value , __Values ... > ) const { return _Value==__Value && Rest()==UIntPack< __Values ... >(); }
	template< unsigned int __Value , unsigned int ... __Values > constexpr bool operator != ( UIntPack< __Value , __Values ... > ) const { return _Value!=__Value && Rest()!=UIntPack< __Values ... >(); }
};
template< unsigned int _Value > struct UIntPack< _Value >
{
	static const unsigned int First = _Value;

	static const unsigned int Size = 1;
	template< unsigned int __Value > using  Append = UIntPack< _Value , __Value >;
	template< unsigned int __Value > using Prepend = UIntPack< __Value , _Value >;
	static const unsigned int Values[];
	static constexpr unsigned int Min( void ){ return _Value; }
	static constexpr unsigned int Max( void ){ return _Value; }

	template< typename T > struct Plus{};
	template< typename T > struct Minus{};
	template< typename T > struct Compare{};
	template< unsigned int __Value > struct Plus < UIntPack< __Value > >{ typedef UIntPack< _Value + __Value > type; };
	template< unsigned int __Value > struct Minus< UIntPack< __Value > >{ typedef UIntPack< _Value - __Value > type; };
	template< unsigned int __Value > struct Compare< UIntPack< __Value > >
	{
		static const bool              Equal = _Value==__Value;
		static const bool           NotEqual = _Value!=__Value;
		static const bool    LessThan        = _Value< __Value;
		static const bool    LessThanOrEqual = _Value<=__Value;
		static const bool GreaterThan        = _Value> __Value;
		static const bool GreaterThanOrEqual = _Value>=__Value;
	};

	static void Print( FILE* fp=stdout , bool leadingSpace=false ){ if( leadingSpace ) fprintf( fp , " " ) ; fprintf( fp , "%d" , _Value ); }
	template< unsigned int I > constexpr static unsigned int Get( void ){ static_assert( I==0 , "[ERROR] UIntPack< Value >::Get called with non-zero index" ) ; return _Value; }

	template< unsigned int __Value > constexpr bool operator <  ( UIntPack< __Value > ) const { return _Value< __Value; }
	template< unsigned int __Value > constexpr bool operator <= ( UIntPack< __Value > ) const { return _Value<=__Value; }
	template< unsigned int __Value > constexpr bool operator >  ( UIntPack< __Value > ) const { return _Value> __Value; }
	template< unsigned int __Value > constexpr bool operator >= ( UIntPack< __Value > ) const { return _Value>=__Value; }
	template< unsigned int __Value > constexpr bool operator == ( UIntPack< __Value > ) const { return _Value==__Value; }
	template< unsigned int __Value > constexpr bool operator != ( UIntPack< __Value > ) const { return _Value!=__Value; }
};
template< unsigned int _Value , unsigned int ... _Values > const unsigned int UIntPack< _Value , _Values ... >::Values[] = { _Value , _Values ... };
template< unsigned int _Value > const unsigned int UIntPack< _Value >::Values[] = { _Value };
template< unsigned int ... V1 , unsigned int ... V2 > typename UIntPack< V1 ... >::template Plus < UIntPack< V2 ... > >::type operator + ( UIntPack< V1 ... > , UIntPack< V2 ... > ){ return typename UIntPack< V1 ... >::template Plus < UIntPack< V2 ... > >::type(); }
template< unsigned int ... V1 , unsigned int ... V2 > typename UIntPack< V1 ... >::template Minus< UIntPack< V2 ... > >::type operator - ( UIntPack< V1 ... > , UIntPack< V2 ... > ){ return typename UIntPack< V1 ... >::template Minus< UIntPack< V2 ... > >::type(); }

template< int ... Values > struct IntPack{};
template< int _Value , int ... _Values > struct IntPack< _Value , _Values ... >
{
	static const int First = _Value;
	typedef IntPack< _Values ... > Rest;

	static const unsigned int Size = 1 + sizeof ... ( _Values );
	template< int __Value > using  Append = IntPack< _Value , _Values ... , __Value >;
	template< int __Value > using Prepend = IntPack< __Value , _Value , _Values ... >;
	static const int Values[];
	static constexpr int Min( void ){ return _Value < Rest::Min ? _Value : Rest::Min; }
	static constexpr int Max( void ){ return _Value > Rest::Max ? _Value : Rest::Max; }

	template< typename T > struct Plus{};
	template< typename T > struct Minus{};
	template< typename T > struct Compare{};
	template< int __Value , int ... __Values > struct Plus < IntPack< __Value , __Values ... > >{ typedef typename Rest::template Plus < IntPack< __Values ... > >::type::template Prepend< _Value + __Value > type; };
	template< int __Value , int ... __Values > struct Minus< IntPack< __Value , __Values ... > >{ typedef typename Rest::template Minus< IntPack< __Values ... > >::type::template Prepend< _Value - __Value > type; };
	template< int __Value , int ... __Values > struct Compare< IntPack< __Value , __Values ... > >
	{
		static const bool              Equal = _Value==__Value && Rest::template Compare< IntPack< __Values ... > >::             Equal;
		static const bool           NotEqual = _Value!=__Value || Rest::template Compare< IntPack< __Values ... > >::          NotEqual;
		static const bool    LessThan        = _Value< __Value && Rest::template Compare< IntPack< __Values ... > >::   LessThan       ;
		static const bool    LessThanOrEqual = _Value<=__Value && Rest::template Compare< IntPack< __Values ... > >::   LessThanOrEqual;
		static const bool GreaterThan        = _Value> __Value && Rest::template Compare< IntPack< __Values ... > >::GreaterThan       ;
		static const bool GreaterThanOrEqual = _Value>=__Value && Rest::template Compare< IntPack< __Values ... > >::GreaterThanOrEqual;
	};

	static void Print( FILE* fp=stdout , bool leadingSpace=false ){ if( leadingSpace ) fprintf( fp , " " ) ; fprintf( fp , "%d" , _Value ) ; Rest::Print( fp , true ); }

	template< unsigned int I > constexpr static typename std::enable_if< I==0 , unsigned int >::type Get( void ){ return _Value; }
	template< unsigned int I > constexpr static typename std::enable_if< I!=0 , unsigned int >::type Get( void ){ return Rest::template Get< I-1 >(); }

	template< int __Value , int ... __Values > constexpr bool operator <  ( IntPack< __Value , __Values ... > ) const { return _Value< __Value && Rest()< IntPack< __Values ... >(); }
	template< int __Value , int ... __Values > constexpr bool operator <= ( IntPack< __Value , __Values ... > ) const { return _Value<=__Value && Rest()<=IntPack< __Values ... >(); }
	template< int __Value , int ... __Values > constexpr bool operator >  ( IntPack< __Value , __Values ... > ) const { return _Value> __Value && Rest()> IntPack< __Values ... >(); }
	template< int __Value , int ... __Values > constexpr bool operator >= ( IntPack< __Value , __Values ... > ) const { return _Value>=__Value && Rest()>=IntPack< __Values ... >(); }
	template< int __Value , int ... __Values > constexpr bool operator == ( IntPack< __Value , __Values ... > ) const { return _Value==__Value && Rest()==IntPack< __Values ... >(); }
	template< int __Value , int ... __Values > constexpr bool operator != ( IntPack< __Value , __Values ... > ) const { return _Value!=__Value && Rest()!=IntPack< __Values ... >(); }
};
template< int _Value > struct IntPack< _Value >
{
	static const int First = _Value;

	static const unsigned int Size = 1;
	template< int __Value > using  Append = IntPack< _Value , __Value >;
	template< int __Value > using Prepend = IntPack< __Value , _Value >;
	static const int Values[];
	static constexpr int Min( void ){ return _Value; }
	static constexpr int Max( void ){ return _Value; }

	template< typename T > struct Plus{};
	template< typename T > struct Minus{};
	template< typename T > struct Compare{};
	template< int __Value > struct Plus < IntPack< __Value > >{ typedef IntPack< _Value + __Value > type; };
	template< int __Value > struct Minus< IntPack< __Value > >{ typedef IntPack< _Value - __Value > type; };
	template< int __Value > struct Compare< IntPack< __Value > >
	{
		static const bool              Equal = _Value==__Value;
		static const bool           NotEqual = _Value!=__Value;
		static const bool    LessThan        = _Value< __Value;
		static const bool    LessThanOrEqual = _Value<=__Value;
		static const bool GreaterThan        = _Value> __Value;
		static const bool GreaterThanOrEqual = _Value>=__Value;
	};

	static void Print( FILE* fp=stdout , bool leadingSpace=false ){ if( leadingSpace ) fprintf( fp , " " ) ; fprintf( fp , "%d" , _Value ); }
	template< unsigned int I > constexpr static unsigned int Get( void ){ static_assert( I==0 , "[ERROR] IntPack< Value >::Get called with non-zero index" ) ; return _Value; }

	template< int __Value > constexpr bool operator <  ( IntPack< __Value > ) const { return _Value< __Value; }
	template< int __Value > constexpr bool operator <= ( IntPack< __Value > ) const { return _Value<=__Value; }
	template< int __Value > constexpr bool operator >  ( IntPack< __Value > ) const { return _Value> __Value; }
	template< int __Value > constexpr bool operator >= ( IntPack< __Value > ) const { return _Value>=__Value; }
	template< int __Value > constexpr bool operator == ( IntPack< __Value > ) const { return _Value==__Value; }
	template< int __Value > constexpr bool operator != ( IntPack< __Value > ) const { return _Value!=__Value; }
};
template< int _Value , int ... _Values > const int IntPack< _Value , _Values ... >::Values[] = { _Value , _Values ... };
template< int _Value > const int IntPack< _Value >::Values[] = { _Value };
template< int ... V1 , int ... V2 > typename IntPack< V1 ... >::template Plus < IntPack< V2 ... > >::type operator + ( IntPack< V1 ... > , IntPack< V2 ... > ){ return typename IntPack< V1 ... >::template Plus < IntPack< V2 ... > >::type(); }
template< int ... V1 , int ... V2 > typename IntPack< V1 ... >::template Minus< IntPack< V2 ... > >::type operator - ( IntPack< V1 ... > , IntPack< V2 ... > ){ return typename IntPack< V1 ... >::template Minus< IntPack< V2 ... > >::type(); }

///////////////////////////
// The isotropic variant //
///////////////////////////
template< unsigned int Dim , unsigned int Value > struct _IsotropicUIntPack             { typedef typename _IsotropicUIntPack< Dim-1 , Value >::type::template Append< Value > type; };
template<                    unsigned int Value > struct _IsotropicUIntPack< 1 , Value >{ typedef UIntPack< Value > type; };
template<                    unsigned int Value > struct _IsotropicUIntPack< 0 , Value >{ typedef UIntPack< > type; };
template< unsigned int Dim , unsigned int Value > using IsotropicUIntPack = typename _IsotropicUIntPack< Dim , Value >::type;
template< unsigned int Dim > using ZeroUIntPack = IsotropicUIntPack< Dim , 0 >;

template< int Dim , int Value > struct _IsotropicIntPack             { typedef typename _IsotropicUIntPack< Dim-1 , Value >::type::template Append< Value > type; };
template<           int Value > struct _IsotropicIntPack< 1 , Value >{ typedef IntPack< Value > type; };
template<           int Value > struct _IsotropicIntPack< 0 , Value >{ typedef IntPack< > type; };
template< int Dim , int Value > using IsotropicIntPack = typename _IsotropicIntPack< Dim , Value >::type;
template< int Dim > using ZeroIntPack = IsotropicIntPack< Dim , 0 >;
/////////////////////////////
// And now for the windows //
/////////////////////////////
template< typename T > struct WindowSize{};
template< typename T1 , typename T2 > struct WindowIndex{};

template< unsigned int Res , unsigned int ... Ress > struct WindowSize< UIntPack< Res , Ress ... > >{ static const unsigned int Size = WindowSize< UIntPack< Ress ... > >::Size * Res; };
template< unsigned int Res                         > struct WindowSize< UIntPack< Res            > >{ static const unsigned int Size = Res; };

template< unsigned int Res , unsigned int ... Ress , unsigned int Idx , unsigned int ... Idxs > struct WindowIndex< UIntPack< Res , Ress ... > , UIntPack< Idx , Idxs ... > >{ static const unsigned int Index = Idx * WindowSize< UIntPack< Ress ... > >::Size + WindowIndex< UIntPack< Ress ... > , UIntPack< Idxs ... > >::Index; };
template< unsigned int Res                         , unsigned int Idx                         > struct WindowIndex< UIntPack< Res            > , UIntPack< Idx            > >{ static const unsigned int Index = Idx; };

template< unsigned int Res , unsigned int ... Ress > typename std::enable_if< (sizeof...(Ress)!=0) , unsigned int >::type GetWindowIndex( UIntPack< Res , Ress ... > , const unsigned int idx[] ){ return idx[0] * WindowSize< UIntPack< Ress ... > >::Size + GetWindowIndex( UIntPack< Ress ... >() , idx+1 ); };
template< unsigned int Res                         > unsigned int GetWindowIndex( UIntPack< Res > , const unsigned int idx[] ){ return idx[0]; }

template< unsigned int Res , unsigned int ... Ress > typename std::enable_if< (sizeof...(Ress)!=0) , unsigned int >::type GetWindowIndex( UIntPack< Res , Ress ... > , const int idx[] ){ return idx[0] * WindowSize< UIntPack< Ress ... > >::Size + GetWindowIndex( UIntPack< Ress ... >() , idx+1 ); };
template< unsigned int Res                         > unsigned int GetWindowIndex( UIntPack< Res > , const int idx[] ){ return idx[0]; }

template< typename Data , typename Pack > struct   ConstWindowSlice{};
template< typename Data , typename Pack > struct        WindowSlice{};
template< typename Data , typename Pack > struct  StaticWindow     {};
template< typename Data , typename Pack > struct DynamicWindow     {};


template< class Data , unsigned int ... Ress >
struct ConstWindowSlice< Data , UIntPack< Ress ... > >
{
	typedef UIntPack< Ress ... > Pack;
	static const unsigned int Size = WindowSize< Pack >::Size;
	typedef Data data_type;
	typedef const Data& data_reference_type;
	typedef const Data& const_data_reference_type;
	ConstWindowSlice( Pointer( Data ) d ) : data(d) { ; }
	ConstWindowSlice( ConstPointer( Data ) d ) : data(d) { ; }
	ConstWindowSlice< Data , typename Pack::Rest > operator[]( int idx ) const { return ConstWindowSlice< Data , typename Pack::Rest >( data + WindowSize< typename Pack::Rest >::Size * idx ); }
	data_reference_type operator()( const          int idx[sizeof...(Ress)] ) const { return data[ GetWindowIndex( UIntPack< Ress ... >() , idx ) ]; }
	data_reference_type operator()( const unsigned int idx[sizeof...(Ress)] ) const { return data[ GetWindowIndex( UIntPack< Ress ... >() , idx ) ]; }
	ConstPointer( Data ) data;
};
template< class Data , unsigned int Res >
struct ConstWindowSlice< Data , UIntPack< Res > >
{
	typedef UIntPack< Res > Pack;
	static const unsigned int Size = Res;
	typedef Data data_type;
	typedef const Data& data_reference_type;
	typedef const Data& const_data_reference_type;
	ConstWindowSlice( Pointer( Data ) d ) : data(d) { ; }
	ConstWindowSlice( ConstPointer( Data ) d ) : data(d) { ; }
	inline data_reference_type operator[]( int idx ) const { return data[idx]; }
	data_reference_type operator()( const          int idx[1] ) const { return data[ idx[0] ]; }
	data_reference_type operator()( const unsigned int idx[1] ) const { return data[ idx[0] ]; }
	ConstPointer( Data ) data;
};
template< class Data , unsigned int ... Ress >
struct WindowSlice< Data , UIntPack< Ress ... > >
{
	typedef UIntPack< Ress ... > Pack;
	static const unsigned int Size = WindowSize< Pack >::Size;
	typedef Data data_type;
	typedef Data& data_reference_type;
	typedef const Data& const_data_reference_type;
	WindowSlice( Pointer( Data ) d ) : data(d) { ; }
	WindowSlice< Data , typename Pack::Rest > operator[]( int idx ){ return WindowSlice< Data , typename Pack::Rest >( data + WindowSize< typename Pack::Rest >::Size * idx ); }
	inline data_reference_type operator()( const int idx[sizeof...(Ress)] ){ return (*this)[ idx[0] ]( idx+1 ); }
	const_data_reference_type operator()( const int idx[sizeof...(Ress)] ) const { return (*this)[ idx[0] ]( idx+1 ); }
	operator ConstWindowSlice< Data , Pack >() const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )data ); }
	Pointer( Data ) data;
};
template< class Data , unsigned int Res >
struct WindowSlice< Data , UIntPack< Res > >
{
	typedef UIntPack< Res > Pack;
	static const unsigned int Size = Res;
	typedef Data data_type;
	typedef Data& data_reference_type;
	typedef const Data& const_data_reference_type;
	WindowSlice( Pointer( Data ) d ) : data(d) { ; }
	inline data_reference_type operator[]( int idx ){ return data[idx]; }
	inline const_data_reference_type operator[]( int idx ) const { return data[idx]; }
	data_reference_type operator()( const int idx[1] ){ return (*this)[ idx[0] ]; }
	const_data_reference_type operator()( const int idx[1] ) const { return (*this)[ idx[0] ]; }
	operator ConstWindowSlice< Data , Pack >() const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )data ); }
	Pointer( Data ) data;
};

template< class Data , unsigned int ... Ress >
struct StaticWindow< Data , UIntPack< Ress ... > >
{
	typedef UIntPack< Ress ... > Pack;
#if defined( __GNUC__ ) && defined( DEBUG )
#warning "you've got me gcc"
	static const unsigned int Size;
#else // !( __GNUC__ && DEBUG )
	static const unsigned int Size = WindowSize< Pack >::Size;
#endif // ( __GNUC__ && DEBUG )
	typedef ConstWindowSlice< Data , Pack > const_window_slice_type;
	typedef WindowSlice< Data , Pack > window_slice_type;
	typedef Data data_type;
	WindowSlice< Data , typename Pack::Rest > operator[]( int idx ){ return WindowSlice< Data , typename Pack::Rest >( GetPointer( data , WindowSize< Pack >::Size ) + WindowSize< typename Pack::Rest >::Size * idx ); }
	ConstWindowSlice< Data , typename Pack::Rest > operator[]( int idx ) const { return ConstWindowSlice< Data , typename Pack::Rest >( ( ConstPointer( Data ) )GetPointer( data , WindowSize< Pack >::Size ) + WindowSize< typename Pack::Rest >::Size * idx ); }
	WindowSlice< Data , Pack > operator()( void ){ return WindowSlice< Data , Pack >( GetPointer( data , WindowSize< Pack >::Size ) ); }
	ConstWindowSlice< Data , Pack > operator()( void ) const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )GetPointer( data , WindowSize< Pack >::Size ) ); }
	Data& operator()( const int idx[sizeof...(Ress)] ){ return (*this)()( idx ); }
	const Data& operator()( const unsigned int idx[sizeof...(Ress)] ) const { return data[ GetWindowIndex( UIntPack< Ress ... >() , idx ) ]; }
	const Data& operator()( const          int idx[sizeof...(Ress)] ) const { return data[ GetWindowIndex( UIntPack< Ress ... >() , idx ) ]; }
	Data data[ WindowSize< Pack >::Size ];
};
#if defined( __GNUC__ ) && defined( DEBUG )
template< class Data , unsigned int ... Ress >
const unsigned int StaticWindow< Data , UIntPack< Ress ... > >::Size = WindowSize< UIntPack< Ress ... > >::Size;
#endif // ( __GNUC__ && DEBUG )
template< class Data , unsigned int Res >
struct StaticWindow< Data , UIntPack< Res > >
{
	typedef UIntPack< Res > Pack;
#if defined( __GNUC__ ) && defined( DEBUG )
#warning "you've got me gcc"
	static const unsigned int Size;
#else // !( __GNUC__ && DEBUG )
	static const unsigned int Size = Res;
#endif // ( __GNUC__ && DEBUG )
	typedef Data data_type;
	Data& operator[]( int idx ){ return data[idx]; };
	const Data& operator[]( int idx ) const { return data[idx]; };
	WindowSlice< Data , Pack > operator()( void ){ return WindowSlice< Data , Pack >( GetPointer( data , WindowSize< Pack >::Size ) ); }
	ConstWindowSlice< Data , Pack > operator()( void ) const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )GetPointer( data , WindowSize< Pack >::Size ) ); }
	Data& operator()( const int idx[1] ){ return (*this)()( idx ); }
	const Data& operator()( const unsigned int idx[1] ) const { return data[ idx[0] ]; }
	const Data& operator()( const          int idx[1] ) const { return data[ idx[0] ]; }
	Data data[ Res ];
};
#if defined( __GNUC__ ) && defined( DEBUG )
template< class Data , unsigned int Res >
const unsigned int StaticWindow< Data , UIntPack< Res > >::Size = Res;
#endif // ( __GNUC__ && DEBUG )

template< class Data , unsigned int ... Ress >
struct DynamicWindow< Data , UIntPack< Ress ... > >
{
	typedef UIntPack< Ress ... > Pack;
	static const unsigned int Size = WindowSize< Pack >::Size;
	typedef ConstWindowSlice< Data , Pack > const_window_slice_type;
	typedef WindowSlice< Data , Pack > window_slice_type;
	typedef Data data_type;
	WindowSlice< Data , typename Pack::Rest > operator[]( int idx ){ return WindowSlice< Data , typename Pack::Rest >( data + WindowSize< typename Pack::Rest >::Size * idx ); }
	ConstWindowSlice< Data , typename Pack::Rest > operator[]( int idx ) const { return ConstWindowSlice< Data , typename Pack::Rest >( ( ConstPointer( Data ) )( data + WindowSize< typename Pack::Rest >::Size * idx ) ); }
	WindowSlice< Data , Pack > operator()( void ){ return WindowSlice< Data , Pack >( data ); }
	ConstWindowSlice< Data , Pack > operator()( void ) const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )data ); }
	Data& operator()( const int idx[sizeof...(Ress)+1] ){ return (*this)()( idx ); }
	const Data& operator()( const int idx[sizeof...(Ress)+1] ) const { return (*this)()( idx ); }

	DynamicWindow( void ){ data = NewPointer< Data >( WindowSize< Pack >::Size ); }
	~DynamicWindow( void ){ DeletePointer( data ); }
	Pointer( Data ) data;
};
template< class Data , unsigned int Res >
struct DynamicWindow< Data , UIntPack< Res > >
{
	typedef UIntPack< Res > Pack;
	static const unsigned int Size = Res;
	typedef Data data_type;
	Data& operator[]( int idx ){ return data[idx]; };
	const Data& operator[]( int idx ) const { return data[idx]; };
	WindowSlice< Data , Pack > operator()( void ) { return WindowSlice< Data , Pack >( data ); }
	ConstWindowSlice< Data , Pack > operator()( void ) const { return ConstWindowSlice< Data , Pack >( ( ConstPointer( Data ) )data ); }
	Data& operator()( const int idx[1] ){ return (*this)()( idx ); }
	const Data& operator()( const int idx[1] ) const { return (*this)()( idx ); }

	DynamicWindow( void ){ data = NewPointer< Data >( Res ); }
	~DynamicWindow( void ){ DeletePointer( data ); }
	Pointer( Data ) data;
};

// Recursive loop iterations for processing window slices
//		WindowDimension: the the window slice
//		IterationDimensions: the number of dimensions to process
//		Res: the resolution of the window

template< unsigned int WindowDimension , unsigned int IterationDimensions , unsigned int CurrentIteration > struct _WindowLoop;
template< unsigned int WindowDimension , unsigned int IterationDimensions=WindowDimension >
struct WindowLoop
{
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
	{
		_WindowLoop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
	}
};

#include "Window.inl"

#endif // WINDOW_INCLUDED
