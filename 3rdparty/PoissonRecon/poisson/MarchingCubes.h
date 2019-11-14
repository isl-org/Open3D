/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
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

#ifndef MARCHING_CUBES_INCLUDED
#define MARCHING_CUBES_INCLUDED

#include <stdio.h>
#include <type_traits>
#include "Geometry.h"

#include "Window.h"

namespace HyperCube
{
	enum Direction{ BACK , CROSS , FRONT };
	inline Direction Opposite( Direction dir ){ return dir==BACK ? FRONT : ( dir==FRONT ? BACK : CROSS ); }

	// The number of k-dimensional elements in a d-dimensional cube is equal to
	// the number of (k-1)-dimensional elements in a (d-1)-dimensional hypercube plus twice the number of k-dimensional elements in a (d-1)-dimensional hypercube
	
	// Number of elements of dimension K in a cube of dimension D
	template< unsigned int D , unsigned int K > struct ElementNum         { static const unsigned int Value = 2 * ElementNum< D-1 , K >::Value + ElementNum< D-1 , K-1 >::Value; };
	template< unsigned int D                  > struct ElementNum< D , 0 >{ static const unsigned int Value = 2 * ElementNum< D-1 , 0 >::Value; };
	template< unsigned int D                  > struct ElementNum< D , D >{ static const unsigned int Value = 1; };
	template<                                 > struct ElementNum< 0 , 0 >{ static const unsigned int Value = 1; };
	// [WARNING] This shouldn't really happen, but we need to support the definition of OverlapElementNum
	template<                  unsigned int K > struct ElementNum< 0 , K >{ static const unsigned int Value = K==0 ? 1 : 0; };

	template< unsigned int D , unsigned int K1 , unsigned int K2 > struct OverlapElementNum             { static const unsigned int Value = K1>=K2 ? ElementNum< K1 , K2 >::Value : OverlapElementNum< D-1 , K1 , K2 >::Value + OverlapElementNum< D-1 , K1 , K2-1 >::Value; };
	template< unsigned int D ,                   unsigned int K  > struct OverlapElementNum< D , D , K >{ static const unsigned int Value = ElementNum< D , K >::Value; };
	template< unsigned int D                                     > struct OverlapElementNum< D , D , 0 >{ static const unsigned int Value = ElementNum< D , 0 >::Value; };
	template< unsigned int D , unsigned int K                    > struct OverlapElementNum< D , K , 0 >{ static const unsigned int Value = ElementNum< K , 0 >::Value; };
	template< unsigned int D , unsigned int K                    > struct OverlapElementNum< D , K , K >{ static const unsigned int Value = 1; };
	template< unsigned int D , unsigned int K                    > struct OverlapElementNum< D , K , D >{ static const unsigned int Value = 1; };
	template< unsigned int D                                     > struct OverlapElementNum< D , D , D >{ static const unsigned int Value = 1; };
	template< unsigned int D                                     > struct OverlapElementNum< D , 0 , 0 >{ static const unsigned int Value = 1; };

	template< unsigned int D >
	struct Cube
	{
		// Corner index (x,y,z,...) -> x + 2*y + 4*z + ...
		// CROSS -> the D-th axis 

		// Representation of a K-dimensional element of the cube
		template< unsigned int K >
		struct Element
		{
			static_assert( D>=K , "[ERROR] Element dimension exceeds cube dimension" );

			// The index of the element, sorted as:
			// 1. All K-dimensional elements contained in the back face
			// 2. All K-dimensional elements spanning the D-th axis
			// 3. All K-dimensional elements contained in the front face
			unsigned int index;

			// Initialize by index
			Element( unsigned int idx=0 );

			// Initialize by co-index:
			// 1. A K-dimensional element in either BACK or FRONT
			// 2. A (K-1)-dimensional element extruded across the D-th axis
			Element( Direction dir , unsigned int coIndex );

			// Given a K-Dimensional sub-element living inside a DK-dimensional sub-cube, get the element relative to the D-dimensional cube
			template< unsigned int DK >
			Element( Element< DK > subCube , typename Cube< DK >::template Element< K > subElement );

			// Initialize by setting the directions
			Element( const Direction dirs[D] );

			// Print the element to the specified stream
			void print( FILE* fp=stdout ) const;

			// Sets the direction and co-index of the element
			void factor( Direction& dir , unsigned int& coIndex ) const;

			// Returns the direction along which the element lives
			Direction direction( void ) const;

			// Returns the co-index of the element
			unsigned int coIndex( void ) const;

			// Compute the directions of the element
			void directions( Direction* dirs ) const;

			// Returns the antipodal element
			typename Cube< D >::template Element< K > antipodal( void ) const;

			// Comparison operators
			bool operator <  ( Element e ) const { return index< e.index; }
			bool operator <= ( Element e ) const { return index<=e.index; }
			bool operator >  ( Element e ) const { return index> e.index; }
			bool operator >= ( Element e ) const { return index>=e.index; }
			bool operator == ( Element e ) const { return index==e.index; }
			bool operator != ( Element e ) const { return index!=e.index; }
			bool operator <  ( unsigned int i ) const { return index< i; }
			bool operator <= ( unsigned int i ) const { return index<=i; }
			bool operator >  ( unsigned int i ) const { return index> i; }
			bool operator >= ( unsigned int i ) const { return index>=i; }
			bool operator == ( unsigned int i ) const { return index==i; }
			bool operator != ( unsigned int i ) const { return index!=i; }

			// Increment operators
			Element& operator ++ ( void ) { index++ ; return *this; }
			Element  operator ++ ( int ) { index++ ; return Element(index-1); }
		protected:
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D!=0 && _K!=0 >::type _setElement( Direction dir , unsigned int coIndex );
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D!=0 && _K==0 >::type _setElement( Direction dir , unsigned int coIndex );
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D==0 && _K==0 >::type _setElement( Direction dir , unsigned int coIndex );

			template< unsigned int KD > typename std::enable_if< (D> KD) && (KD>K) && K!=0 >::type _setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement );
			template< unsigned int KD > typename std::enable_if< (D> KD) && (KD>K) && K==0 >::type _setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement );
			template< unsigned int KD > typename std::enable_if< (D==KD) && (KD>K)         >::type _setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement );
			template< unsigned int KD > typename std::enable_if< (KD==K)                   >::type _setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement );

			template< unsigned int _D=D > typename std::enable_if< _D!=0 >::type _setElement( const Direction* dirs );
			template< unsigned int _D=D > typename std::enable_if< _D==0 >::type _setElement( const Direction* dirs );

			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D==_K          >::type _factor( Direction& dir , unsigned int& coIndex ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D!=_K && _K!=0 >::type _factor( Direction& dir , unsigned int& coIndex ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D!=_K && _K==0 >::type _factor( Direction& dir , unsigned int& coIndex ) const;

			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< (_D>_K) && _K!=0 >::type _directions( Direction* dirs ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< (_D>_K) && _K==0 >::type _directions( Direction* dirs ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D==_K           >::type _directions( Direction* dirs ) const;

			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< (_D>_K) && _K!=0 , Element >::type _antipodal( void ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< (_D>_K) && _K==0 , Element >::type _antipodal( void ) const;
			template< unsigned int _D=D , unsigned int _K=K > typename std::enable_if< _D==_K           , Element >::type _antipodal( void ) const;
		};
		// A way of indexing the cubes incident on an element
		template< unsigned int K > using IncidentCubeIndex = typename Cube< D-K >::template Element< 0 >;

		// Number of elements of dimension K
		template< unsigned int K > static constexpr unsigned int ElementNum( void ){ return HyperCube::ElementNum< D , K >::Value; }
		// Number of cubes incident to an element of dimension K
		template< unsigned int K > static constexpr unsigned int IncidentCubeNum( void ){ return HyperCube::ElementNum< D-K , 0 >::Value; }
		// Number of overlapping elements of dimension K1 / K2
		template< unsigned int K1 , unsigned int K2 > static constexpr unsigned int OverlapElementNum( void ){ return HyperCube::OverlapElementNum< D , K1 , K2 >::Value; }

		// Is the face outward-facing
		static bool IsOriented( Element< D-1 > e );

		// Is one element contained in the other?
		template< unsigned int K1 , unsigned int K2 >
		static bool Overlap( Element< K1 > e1 , Element< K2 > e2 );

		// If K1>K2: returns all elements contained in e
		// Else:     returns all elements containing e
		template< unsigned int K1 , unsigned int K2 >
		static void OverlapElements( Element< K1 > e , Element< K2 >* es );

		// Returns the marching-cubes index for the set of values
		template< typename Real >
		static unsigned int MCIndex( const Real values[ Cube::ElementNum< 0 >() ] , Real iso );

		// Extracts the marching-cubes sub-index for the associated element
		template< unsigned int K >
		static unsigned int ElementMCIndex( Element< K > element , unsigned int mcIndex );

		// Does the marching cubes index have a zero-crossing
		static bool HasMCRoots( unsigned int mcIndex );

		// Sets the offset of the incident cube relative to the center cube, x[i] \in {-1,0,1}
		template< unsigned int K >
		static void CellOffset( Element< K > e , IncidentCubeIndex< K > d , int x[D] );

		// Returns the linearized offset of the incident cube relative to the center cube, \in [0,3^D)
		template< unsigned int K >
		static unsigned int CellOffset( Element< K > e , IncidentCubeIndex< K > d );

		// Returns the index of the incident cube that is the source
		template< unsigned int K >
		static typename Cube< D >::template IncidentCubeIndex< K > IncidentCube( Element< K > e );

		// Returns the corresponding element in the incident cube
		template< unsigned int K >
		static typename Cube< D >::template Element< K > IncidentElement( Element< K > e , IncidentCubeIndex< K > d );

	protected:
		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1>=K2) , bool >::type _Overlap( Element< K1 > e1 , Element< K2 > e2 );
		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1< K2) , bool >::type _Overlap( Element< K1 > e1 , Element< K2 > e2 );

		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1>=K2)                   >::type _OverlapElements( Element< K1 > e , Element< K2 >* es );
		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1< K2) && D==K2          >::type _OverlapElements( Element< K1 > e , Element< K2 >* es );
		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1< K2) && D!=K2 && K1!=0 >::type _OverlapElements( Element< K1 > e , Element< K2 >* es );
		template< unsigned int K1 , unsigned int K2 > static typename std::enable_if< (K1< K2) && D!=K2 && K1==0 >::type _OverlapElements( Element< K1 > e , Element< K2 >* es );

		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K                  , IncidentCubeIndex< K > >::type _IncidentCube( Element< K > e );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && _D!=0 && K!=0 , IncidentCubeIndex< K > >::type _IncidentCube( Element< K > e );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && _D!=0 && K==0 , IncidentCubeIndex< K > >::type _IncidentCube( Element< K > e );

		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K         >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d , int* x );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K!=0 >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d , int* x );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K==0 >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d , int* x );

		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K && K==0 , unsigned int >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K && K!=0 , unsigned int >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K!=0 , unsigned int >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K==0 , unsigned int >::type _CellOffset( Element< K > e , IncidentCubeIndex< K > d );

		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K                  , Element< K > >::type _IncidentElement( Element< K > e , IncidentCubeIndex< K > d );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && _D!=0 && K!=0 , Element< K > >::type _IncidentElement( Element< K > e , IncidentCubeIndex< K > d );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && _D!=0 && K==0 , Element< K > >::type _IncidentElement( Element< K > e , IncidentCubeIndex< K > d );

		template< unsigned int _D=D > static typename std::enable_if< _D!=1 >::type _FactorOrientation( Element< D-1 > e , unsigned int& dim , Direction& dir );
		template< unsigned int _D=D > static typename std::enable_if< _D==1 >::type _FactorOrientation( Element< D-1 > e , unsigned int& dim , Direction& dir );

		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K!=0 , unsigned int >::type _ElementMCIndex( Element< K > element , unsigned int mcIndex );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D!=K && K==0 , unsigned int >::type _ElementMCIndex( Element< K > element , unsigned int mcIndex );
		template< unsigned int K , unsigned int _D=D > static typename std::enable_if< _D==K         , unsigned int >::type _ElementMCIndex( Element< K > element , unsigned int mcIndex );

		template< unsigned int DD > friend struct Cube;
	};

	// Specialized class for extracting iso-curves from a square
	struct MarchingSquares
	{
		const static unsigned int MAX_EDGES=2;
		static const int edges[1<<HyperCube::Cube< 2 >::ElementNum< 0 >()][2*MAX_EDGES+1];
		static int AddEdgeIndices( unsigned char mcIndex , int* edges);
	};

	///////////////////
	// Cube::Element //
	///////////////////
	template< unsigned int D > template< unsigned int K >
	Cube< D >::Element< K >::Element( unsigned int idx ) : index( idx ){}
	template< unsigned int D > template< unsigned int K >
	Cube< D >::Element< K >::Element( Direction dir , unsigned int coIndex ){ _setElement( dir , coIndex ); }
	template< unsigned int D > template< unsigned int K > template< unsigned int DK >
	Cube< D >::Element< K >::Element( Element< DK > subCube , typename Cube< DK >::template Element< K > subElement )
	{
		static_assert( DK>=K , "[ERROR] Element::Element: sub-cube dimension cannot be smaller than the sub-element dimension" );
		static_assert( DK<=D , "[ERROR] Element::Element: sub-cube dimension cannot be larger than the cube dimension" );
		_setElement( subCube , subElement );
	}
	template< unsigned int D > template< unsigned int K >
	Cube< D >::Element< K >::Element( const Direction dirs[D] ){ _setElement( dirs ); }

	template< unsigned int D > template< unsigned int K > template< unsigned int KD >
	typename std::enable_if< (D>KD) && (KD>K) && K!=0 >::type Cube< D >::Element< K >::_setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement )
	{
		Direction dir ; unsigned int coIndex;
		subCube.factor( dir , coIndex );
		// If the sub-cube lies entirely in the back/front, we can compute the element in the smaller cube.
		if( dir==BACK || dir==FRONT )
		{
			typename Cube< D-1 >::template Element< KD > _subCube( coIndex );
			typename Cube< D-1 >::template Element< K > _element( _subCube , subElement );
			*this = Element( dir , _element.index );
		}
		else
		{
			typename Cube< D-1 >::template Element< KD-1 > _subCube( coIndex );

			Direction _dir ; unsigned int _coIndex;
			subElement.factor( _dir , _coIndex );
			// If the sub-element lies entirely in the back/front, we can compute the element in the smaller cube.
			if( _dir==BACK || _dir==FRONT )
			{
				typename Cube< KD-1 >::template Element< K > _subElement( _coIndex );
				typename Cube< D-1 >::template Element< K > _element( _subCube , _subElement );
				*this = Element( _dir , _element.index );
			}
			// Otherwise
			else
			{
				typename Cube< KD-1 >::template Element< K-1 > _subElement( _coIndex );
				typename Cube< D-1 >::template Element< K-1 > _element( _subCube , _subElement );
				*this = Element( _dir , _element.index );
			}
		}
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int KD >
	typename std::enable_if< (D>KD) && (KD>K) && K==0 >::type Cube< D >::Element< K >::_setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement )
	{
		Direction dir ; unsigned int coIndex;
		subCube.factor( dir , coIndex );
		// If the sub-cube lies entirely in the back/front, we can compute the element in the smaller cube.
		if( dir==BACK || dir==FRONT )
		{
			typename Cube< D-1 >::template Element< KD > _subCube( coIndex );
			typename Cube< D-1 >::template Element< K > _element( _subCube , subElement );
			*this = Element( dir , _element.index );
		}
		else
		{
			typename Cube< D-1 >::template Element< KD-1 > _subCube( coIndex );

			Direction _dir ; unsigned int _coIndex;
			subElement.factor( _dir , _coIndex );
			// If the sub-element lies entirely in the back/front, we can compute the element in the smaller cube.
			if( _dir==BACK || _dir==FRONT )
			{
				typename Cube< KD-1 >::template Element< K > _subElement( _coIndex );
				typename Cube< D-1 >::template Element< K > _element( _subCube , _subElement );
				*this = Element( _dir , _element.index );
			}
		}
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int KD >
	typename std::enable_if< (D==KD) && (KD>K) >::type Cube< D >::Element< K >::_setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement ){ *this = subElement; }
	template< unsigned int D > template< unsigned int K > template< unsigned int KD >
	typename std::enable_if< (KD==K) >::type Cube< D >::Element< K >::_setElement( typename Cube< D >::template Element< KD > subCube , typename Cube< KD >::template Element< K > subElement ){ *this = subCube; }

	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D!=0 && _K!=0 >::type Cube< D >::Element< K >::_setElement( Direction dir , unsigned int coIndex )
	{
		switch( dir )
		{
			case BACK:  index = coIndex ; break;
			case CROSS: index = coIndex + HyperCube::ElementNum< D-1 , K >::Value ; break;
			case FRONT: index = coIndex + HyperCube::ElementNum< D-1 , K >::Value + HyperCube::ElementNum< D-1 , K-1 >::Value ; break;
			default: ERROR_OUT( "Bad direction: " , dir );
		}
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D!=0 && _K==0 >::type Cube< D >::Element< K >::_setElement( Direction dir , unsigned int coIndex )
	{
		switch( dir )
		{
			case BACK:  index = coIndex ; break;
			case FRONT: index = coIndex + HyperCube::ElementNum< D-1 , K >::Value ; break;
			default: ERROR_OUT( "Bad direction: " , dir );
		}
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D==0 && _K==0 >::type Cube< D >::Element< K >::_setElement( Direction dir , unsigned int coIndex ){ index = coIndex; }

	template< unsigned int D > template< unsigned int K > template< unsigned int _D >
	typename std::enable_if< _D!=0 >::type Cube< D >::Element< K>::_setElement( const Direction* dirs )
	{
		if( dirs[D-1]==CROSS ) *this = Element( dirs[D-1] , typename Cube< D-1 >::template Element< K-1 >( dirs ).index );
		else                   *this = Element( dirs[D-1] , typename Cube< D-1 >::template Element< K >( dirs ).index );
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D >
	typename std::enable_if< _D==0 >::type Cube< D >::Element< K>::_setElement( const Direction* dirs ){}

	template< unsigned int D > template< unsigned int  K >
	void Cube< D >::Element< K >::print( FILE* fp ) const
	{
		Direction dirs[D==0?1:D];
		directions( dirs );
		for( int d=0 ; d<D ; d++ ) fprintf( fp , "%c" , dirs[d]==BACK ? 'B' : ( dirs[d]==CROSS ? 'C' : 'F' ) );
	}

	template< unsigned int D > template< unsigned int K >
	void Cube< D >::Element< K >::factor( Direction& dir , unsigned int& coIndex ) const { _factor( dir , coIndex ); }

	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D!=_K && _K!=0 >::type Cube< D >::Element< K >::_factor( Direction& dir , unsigned int& coIndex ) const
	{
		if     ( index<HyperCube::ElementNum< D-1 , K >::Value )                                             dir = BACK  , coIndex = index;
		else if( index<HyperCube::ElementNum< D-1 , K >::Value + HyperCube::ElementNum< D-1 , K-1 >::Value ) dir = CROSS , coIndex = index - HyperCube::ElementNum< D-1 , K >::Value;
		else                                                                                                 dir = FRONT , coIndex = index - HyperCube::ElementNum< D-1 , K >::Value - HyperCube::ElementNum< D-1 , K-1 >::Value;
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D!=_K && _K==0 >::type Cube< D >::Element< K >::_factor( Direction& dir , unsigned int& coIndex ) const
	{
		if     ( index<HyperCube::ElementNum< D-1 , K >::Value ) dir = BACK  , coIndex = index;
		else                                                     dir = FRONT , coIndex = index - HyperCube::ElementNum< D-1 , K >::Value;
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D==_K >::type Cube< D >::Element< K >::_factor( Direction& dir , unsigned int& coIndex ) const { dir=CROSS , coIndex=0; }

	template< unsigned int D > template< unsigned int K >
	Direction Cube< D >::Element< K >::direction( void ) const
	{
		Direction dir ; unsigned int coIndex;
		factor( dir , coIndex );
		return dir;
	}
	template< unsigned int D > template< unsigned int K >
	unsigned int Cube< D >::Element< K >::coIndex( void ) const
	{
		Direction dir ; unsigned int coIndex;
		factor( dir , coIndex );
		return coIndex;
	}

	template< unsigned int D > template< unsigned int K >
	void Cube< D >::Element< K >::directions( Direction* dirs ) const { _directions( dirs ); }
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< (_D>_K) && _K!=0 >::type Cube< D >::Element< K >::_directions( Direction* dirs ) const
	{
		unsigned int coIndex;
		factor( dirs[D-1] , coIndex );
		if( dirs[D-1]==CROSS ) typename Cube< D-1 >::template Element< K-1 >( coIndex ).directions( dirs );
		else                   typename Cube< D-1 >::template Element< K   >( coIndex ).directions( dirs );
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< (_D>_K) && _K==0 >::type Cube< D >::Element< K >::_directions( Direction* dirs ) const
	{
		unsigned int coIndex;
		factor( dirs[D-1] , coIndex );
		if( dirs[D-1]==FRONT || dirs[D-1]==BACK ) typename Cube< D-1 >::template Element< K >( coIndex ).directions( dirs );
	}

	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
	typename std::enable_if< _D==_K >::type Cube< D >::Element< K >::_directions( Direction* dirs ) const { for( int d=0 ; d<D ; d++ ) dirs[d] = CROSS; }

	template< unsigned int D > template< unsigned int K >
	typename Cube< D >::template Element< K > Cube< D >::Element< K >::antipodal( void ) const { return _antipodal(); }
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
#ifdef _MSC_VER
	typename std::enable_if< (_D>_K) && _K!=0 , typename Cube< D >::Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const
#else // !_MSC_VER
	typename std::enable_if< (_D>_K) && _K!=0 , typename Cube< D >::template Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const
#endif // _MSC_VER
	{
		Direction dir ; unsigned int coIndex;
		factor( dir , coIndex );
		if     ( dir==CROSS ) return Element< K >( CROSS , typename Cube< D-1 >::template Element< K-1 >( coIndex ).antipodal().index );
		else if( dir==FRONT ) return Element< K >( BACK  , typename Cube< D-1 >::template Element< K   >( coIndex ).antipodal().index );
		else if( dir==BACK  ) return Element< K >( FRONT , typename Cube< D-1 >::template Element< K   >( coIndex ).antipodal().index );
		return Element< K >();
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
#ifdef _MSC_VER
	typename std::enable_if< (_D>_K) && _K==0 , typename Cube< D >::Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const
#else // !_MSC_VER
	typename std::enable_if< (_D>_K) && _K==0 , typename Cube< D >::template Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const
#endif // _MSC_VER
	{
		Direction dir ; unsigned int coIndex;
		factor( dir , coIndex );
		if     ( dir==FRONT ) return Element< K >( BACK  , typename Cube< D-1 >::template Element< K >( coIndex ).antipodal().index );
		else if( dir==BACK  ) return Element< K >( FRONT , typename Cube< D-1 >::template Element< K >( coIndex ).antipodal().index );
		return Element< K >();
	}
	template< unsigned int D > template< unsigned int K > template< unsigned int _D , unsigned int _K >
#ifdef _MSC_VER
	typename std::enable_if< _D==_K , typename Cube< D >::Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const { return *this; }
#else // !_MSC_VER
	typename std::enable_if< _D==_K , typename Cube< D >::template Element< K > >::type Cube< D >::Element< K >::_antipodal( void ) const { return *this; }
#endif // _MSC_VER

	//////////
	// Cube //
	//////////
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	bool Cube< D >::Overlap( Element< K1 > e1 , Element< K2 > e2 ){ return _Overlap( e1 , e2 ); }
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1>=K2) , bool >::type Cube< D >::_Overlap( Element< K1 > e1 , Element< K2 > e2 )
	{
		Direction dir1[ D ] , dir2[ D ];
		e1.directions( dir1 ) , e2.directions( dir2 );
		for( int d=0 ; d<D ; d++ ) if( dir1[d]!=CROSS && dir1[d]!=dir2[d] ) return false;
		return true;
	}
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1< K2) , bool >::type Cube< D >::_Overlap( Element< K1 > e1 , Element< K2 > e2 ){ return _Overlap( e2 , e1 ); }

	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	void Cube< D >::OverlapElements( Element< K1 > e , Element< K2 >* es ){ _OverlapElements( e , es ); }
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1>=K2) >::type Cube< D >::_OverlapElements( Element< K1 > e , Element< K2 >* es )
	{
		for( typename Cube< K1 >::template Element< K2 > _e ; _e<Cube< K1 >::template ElementNum< K2 >() ; _e++ ) es[_e.index] = Element< K2 >( e , _e );
	}
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1< K2) && D==K2 >::type Cube< D >::_OverlapElements( Element< K1 > e , Element< K2 >* es )
	{
		es[0] = Element< D >();
	}
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1< K2) && D!=K2 && K1!=0 >::type Cube< D >::_OverlapElements( Element< K1 > e , Element< K2 >* es )
	{
		Direction dir = e.direction() ; unsigned int coIndex;
		e.factor( dir , coIndex );
		if( dir==FRONT || dir==BACK )
		{
			typename Cube< D-1 >::template Element< K2   > _es1[ HyperCube::OverlapElementNum< D-1 , K1 , K2   >::Value ];
			typename Cube< D-1 >::template Element< K2-1 > _es2[ HyperCube::OverlapElementNum< D-1 , K1 , K2-1 >::Value ];
			Cube< D-1 >::OverlapElements( typename Cube< D-1 >::template Element< K1 >( coIndex ) , _es1 );
			Cube< D-1 >::OverlapElements( typename Cube< D-1 >::template Element< K1 >( coIndex ) , _es2 );
			for( unsigned int i=0 ; i<HyperCube::OverlapElementNum< D-1 , K1 , K2   >::Value ; i++ ) es[i] = typename Cube< D >::template Element< K2 >( dir   , _es1[i].index );
			es += HyperCube::OverlapElementNum< D-1 , K1 , K2 >::Value;
			for( unsigned int i=0 ; i<HyperCube::OverlapElementNum< D-1 , K1 , K2-1 >::Value ; i++ ) es[i] = typename Cube< D >::template Element< K2 >( CROSS , _es2[i].index );
		}
		else if( dir==CROSS )
		{
			typename Cube< D-1 >::template Element< K2-1 > _es1[ HyperCube::OverlapElementNum< D-1 , K1-1 , K2-1 >::Value ];
			Cube< D-1 >::OverlapElements( typename Cube< D-1 >::template Element< K1-1 >( coIndex ) , _es1 );
			for( unsigned int i=0 ; i<HyperCube::OverlapElementNum< D-1 , K1-1 , K2-1 >::Value ; i++ ) es[i] = typename Cube< D >::template Element< K2 >( CROSS , _es1[i].index );
		}
	}
	template< unsigned int D > template< unsigned int K1 , unsigned int K2 >
	typename std::enable_if< (K1< K2) && D!=K2 && K1==0 >::type Cube< D >::_OverlapElements( Element< K1 > e , Element< K2 >* es )
	{
		Direction dir = e.direction() ; unsigned int coIndex;
		e.factor( dir , coIndex );
		if( dir==FRONT || dir==BACK )
		{
			typename Cube< D-1 >::template Element< K2   > _es1[ HyperCube::OverlapElementNum< D-1 , K1 , K2   >::Value ];
			typename Cube< D-1 >::template Element< K2-1 > _es2[ HyperCube::OverlapElementNum< D-1 , K1 , K2-1 >::Value ];
			Cube< D-1 >::OverlapElements( typename Cube< D-1 >::template Element< K1 >( coIndex ) , _es1 );
			Cube< D-1 >::OverlapElements( typename Cube< D-1 >::template Element< K1 >( coIndex ) , _es2 );
			for( unsigned int i=0 ; i<HyperCube::OverlapElementNum< D-1 , K1 , K2   >::Value ; i++ ) es[i] = typename Cube< D >::template Element< K2 >( dir   , _es1[i].index );
			es += HyperCube::OverlapElementNum< D-1 , K1 , K2 >::Value;
			for( unsigned int i=0 ; i<HyperCube::OverlapElementNum< D-1 , K1 , K2-1 >::Value ; i++ ) es[i] = typename Cube< D >::template Element< K2 >( CROSS , _es2[i].index );
		}
	}

	template< unsigned int D > template< unsigned int K >
	typename Cube< D >::template IncidentCubeIndex< K > Cube< D >::IncidentCube( Element< K > e ){ return _IncidentCube( e ); }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D==K , typename Cube< D >::IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e ){ return IncidentCubeIndex< D >(); }
#else // !_MSC_VER
	typename std::enable_if< _D==K , typename Cube< D >::template IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e ){ return IncidentCubeIndex< D >(); }
#endif // _MSC_VER
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K!=0 , typename Cube< D >::IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e )
#else // !_MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K!=0 , typename Cube< D >::template IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e )
#endif // _MSC_VER
	{
		Direction dir ; unsigned int coIndex;
		e.factor( dir , coIndex );
		if     ( dir==CROSS ) return                                 Cube< D-1 >::IncidentCube( typename Cube< D-1 >::template Element< K-1 >( coIndex ) );
		else if( dir==FRONT ) return IncidentCubeIndex< K >( BACK  , Cube< D-1 >::IncidentCube( typename Cube< D-1 >::template Element< K   >( coIndex ) ).index );
		else                  return IncidentCubeIndex< K >( FRONT , Cube< D-1 >::IncidentCube( typename Cube< D-1 >::template Element< K   >( coIndex ) ).index );
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K==0 , typename Cube< D >::IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e )
#else // !_MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K==0 , typename Cube< D >::template IncidentCubeIndex< K > >::type Cube< D >::_IncidentCube( Element< K > e )
#endif // _MSC_VER
	{
		Direction dir ; unsigned int coIndex;
		e.factor( dir , coIndex );
		if( dir==FRONT ) return IncidentCubeIndex< K >( BACK  , Cube< D-1 >::IncidentCube( typename Cube< D-1 >::template Element< K >( coIndex ) ).index );
		else             return IncidentCubeIndex< K >( FRONT , Cube< D-1 >::IncidentCube( typename Cube< D-1 >::template Element< K >( coIndex ) ).index );
	}

	template< unsigned int D >
	bool Cube< D >::IsOriented( Element< D-1 > e )
	{
		unsigned int dim ; Direction dir;
		_FactorOrientation( e , dim , dir );
		return (dir==FRONT) ^ ((D-dim-1)&1);
	}
	template< unsigned int D > template< unsigned int _D >
	typename std::enable_if< _D!=1 >::type Cube< D >::_FactorOrientation( Element< D-1 > e , unsigned int& dim , Direction& dir )
	{
		unsigned int coIndex;
		e.factor( dir , coIndex );
		if( dir==CROSS ) Cube< D-1 >::template _FactorOrientation( typename Cube< D-1 >::template Element< D-2 >( coIndex ) , dim , dir );
		else dim = D-1;
	}
	template< unsigned int D > template< unsigned int _D >
	typename std::enable_if< _D==1 >::type Cube< D >::_FactorOrientation( Element< D-1 > e , unsigned int& dim , Direction& dir )
	{
		unsigned int coIndex;
		e.factor( dir , coIndex );
		dim = 0;
	}

	template< unsigned int D > template< typename Real >
	unsigned int Cube< D >::MCIndex( const Real values[ Cube< D >::ElementNum< 0 >() ] , Real iso )
	{
		unsigned int mcIdx = 0;
		for( unsigned int c=0 ; c<ElementNum< 0 >() ; c++ ) if( values[c]<iso ) mcIdx |= (1<<c);
		return mcIdx;
	}

	template< unsigned int D > template< unsigned int K >
	unsigned int Cube< D >::ElementMCIndex( Element< K > element , unsigned int mcIndex ){ return _ElementMCIndex( element , mcIndex ); }

	template< unsigned int D >
	bool Cube< D >::HasMCRoots( unsigned int mcIndex )
	{
		static const unsigned int Mask = (1<<(1<<D)) - 1;
		return mcIndex!=0 && ( mcIndex & Mask )!=Mask;
	}

	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K!=0 , unsigned int >::type Cube< D >::_ElementMCIndex( Element< K > element , unsigned int mcIndex )
	{
		static const unsigned int Mask = ( 1<<( ElementNum< 0 >() / 2 ) ) - 1;
		static const unsigned int Shift = ElementNum< 0 >() / 2 , _Shift = Cube< K >::template ElementNum< 0 >() / 2;
		unsigned int mcIndex0 = mcIndex & Mask , mcIndex1 = ( mcIndex>>Shift ) & Mask;
		Direction dir ; unsigned int coIndex;
		element.factor( dir , coIndex );
		if( dir==CROSS ) return Cube< D-1 >::template ElementMCIndex< K-1 >( coIndex , mcIndex0 ) | ( Cube< D-1 >::template ElementMCIndex< K-1 >( coIndex , mcIndex1 )<<_Shift );
		else             return Cube< D-1 >::template ElementMCIndex< K   >( coIndex , dir==BACK ? mcIndex0 : mcIndex1 );
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K==0 , unsigned int >::type Cube< D >::_ElementMCIndex( Element< K > element , unsigned int mcIndex )
	{
		static const unsigned int Mask = ( 1<<( ElementNum< 0 >() / 2 ) ) - 1;
		static const unsigned int Shift = ElementNum< 0 >() / 2 , _Shift = Cube< K >::template ElementNum< 0 >() / 2;
		unsigned int mcIndex0 = mcIndex & Mask , mcIndex1 = ( mcIndex>>Shift ) & Mask;
		Direction dir ; unsigned int coIndex;
		element.factor( dir , coIndex );
		return Cube< D-1 >::template ElementMCIndex< K >( typename Cube< D-1 >::template Element< K >( coIndex ) , dir==BACK ? mcIndex0 : mcIndex1 );
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D==K , unsigned int >::type Cube< D >::_ElementMCIndex( Element< K > element , unsigned int mcIndex ){ return mcIndex; }

	template< unsigned int D > template< unsigned int K >
	void Cube< D >::CellOffset( Element< K > e , IncidentCubeIndex< K > d , int x[D] ){ _CellOffset( e , d , x ); }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D==K >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d , int *x ){ for( int d=0 ; d<D ; d++ ) x[d] = 0; }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K!=0 >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d , int *x )
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if     ( eDir==CROSS ){ x[D-1] =  0                          ; Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K-1 >( eCoIndex ) , d                                                                 , x ); }
		else if( eDir==BACK  ){ x[D-1] = -1 + ( dDir==BACK ? 0 : 1 ) ; Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) , x ); }
		else if( eDir==FRONT ){ x[D-1] =  0 + ( dDir==BACK ? 0 : 1 ) ; Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) , x ); }
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K==0 >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d , int *x )
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if     ( eDir==BACK  ){ x[D-1] = -1 + ( dDir==BACK ? 0 : 1 ) ; Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) , x ); }
		else if( eDir==FRONT ){ x[D-1] =  0 + ( dDir==BACK ? 0 : 1 ) ; Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) , x ); }
	}

	template< unsigned int D > template< unsigned int K >
	unsigned int Cube< D >::CellOffset( Element< K > e , IncidentCubeIndex< K > d ){ return _CellOffset( e , d ); }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D==K && K==0 , unsigned int >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d ){ return 0; }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D==K && K!=0 , unsigned int >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d ){ return WindowIndex< IsotropicUIntPack< D , 3 > , IsotropicUIntPack< D , 1 > >::Index; }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K!=0 , unsigned int >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d )
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if     ( eDir==CROSS ){ return 1                          + Cube< D-1 >::template CellOffset( typename Cube< D-1 >::template Element< K-1 >( eCoIndex ) , d                                                                 ) * 3; }
		else if( eDir==BACK  ){ return 0 + ( dDir==BACK ? 0 : 1 ) + Cube< D-1 >::template CellOffset( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ) * 3; }
		else if( eDir==FRONT ){ return 1 + ( dDir==BACK ? 0 : 1 ) + Cube< D-1 >::template CellOffset( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ) * 3; }
		return 0;
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
	typename std::enable_if< _D!=K && K==0 , unsigned int >::type Cube< D >::_CellOffset( Element< K > e , IncidentCubeIndex< K > d )
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if     ( eDir==BACK  ){ return 0 + ( dDir==BACK ? 0 : 1 ) + Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ) * 3; }
		else if( eDir==FRONT ){ return 1 + ( dDir==BACK ? 0 : 1 ) + Cube< D-1 >::CellOffset( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ) * 3; }
		return 0;
	}


	template< unsigned int D > template< unsigned int K >
	typename Cube< D >::template Element< K > Cube< D >::IncidentElement( Element< K > e , IncidentCubeIndex< K > d ){ return _IncidentElement( e , d ); }
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D==K , typename Cube< D >::Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d ){ return e; }
#else // !_MSC_VER
	typename std::enable_if< _D==K , typename Cube< D >::template Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d ){ return e; }
#endif // _MSC_VER
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K!=0 , typename Cube< D >::Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d )
#else // !_MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K!=0 , typename Cube< D >::template Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d )
#endif // _MSC_VER
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if     ( eDir==CROSS ) return Element< K >(           eDir   , Cube< D-1 >::template IncidentElement( typename Cube< D-1 >::template Element< K-1 >( eCoIndex ) , d                                                                 ).index );
		else if( eDir==dDir  ) return Element< K >( Opposite( eDir ) , Cube< D-1 >::template IncidentElement( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ).index );
		else                   return Element< K >(           eDir   , Cube< D-1 >::template IncidentElement( typename Cube< D-1 >::template Element< K   >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ).index );
	}
	template< unsigned int D > template< unsigned int K , unsigned int _D >
#ifdef _MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K==0 , typename Cube< D >::Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d )
#else // !_MSC_VER
	typename std::enable_if< _D!=K && _D!=0 && K==0 , typename Cube< D >::template Element< K > >::type Cube< D >::_IncidentElement( Element< K > e , IncidentCubeIndex< K > d )
#endif // _MSC_VER
	{
		Direction eDir , dDir ; unsigned int eCoIndex , dCoIndex;
		e.factor( eDir , eCoIndex ) , d.factor( dDir , dCoIndex );
		if( eDir==dDir ) return Element< K >( Opposite( eDir ) , Cube< D-1 >::template IncidentElement( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ).index );
		else             return Element< K >(           eDir   , Cube< D-1 >::template IncidentElement( typename Cube< D-1 >::template Element< K >( eCoIndex ) , typename Cube< D-1 >::template IncidentCubeIndex< K >( dCoIndex ) ).index );
	}


	/////////////////////
	// MarchingSquares //
	/////////////////////
	const int MarchingSquares::edges[][MAX_EDGES*2+1] =
	{
		// Positive to the right
		// Positive in center
		/////////////////////////////////// (0,0) (1,0) (0,1) (1,1)
		{ -1 ,  -1 ,  -1 ,  -1 ,  -1 } , //   -     -     -     -  //
		{  1 ,   0 ,  -1 ,  -1 ,  -1 } , //   +     -     -     -  // (0,0) - (0,1) | (0,0) - (1,0)
		{  0 ,   2 ,  -1 ,  -1 ,  -1 } , //   -     +     -     -  // (0,0) - (1,0) | (1,0) - (1,1)
		{  1 ,   2 ,  -1 ,  -1 ,  -1 } , //   +     +     -     -  // (0,0) - (0,1) | (1,0) - (1,1)
		{  3 ,   1 ,  -1 ,  -1 ,  -1 } , //   -     -     +     -  // (0,1) - (1,1) | (0,0) - (0,1)
		{  3 ,   0 ,  -1 ,  -1 ,  -1 } , //   +     -     +     -  // (0,1) - (1,1) | (0,0) - (1,0)
		{  0 ,   1 ,   3 ,   2 ,  -1 } , //   -     +     +     -  // (0,0) - (1,0) | (0,0) - (0,1) & (0,1) - (1,1) | (1,0) - (1,1)
		{  3 ,   2 ,  -1 ,  -1 ,  -1 } , //   +     +     +     -  // (0,1) - (1,1) | (1,0) - (1,1)
		{  2 ,   3 ,  -1 ,  -1 ,  -1 } , //   -     -     -     +  // (1,0) - (1,1) | (0,1) - (1,1)
		{  1 ,   3 ,   2 ,   0 ,  -1 } , //   +     -     -     +  // (0,0) - (0,1) | (0,1) - (1,1) & (1,0) - (1,1) | (0,0) - (1,0)
		{  0 ,   3 ,  -1 ,  -1 ,  -1 } , //   -     +     -     +  // (0,0) - (1,0) | (0,1) - (1,1)
		{  1 ,   3 ,  -1 ,  -1 ,  -1 } , //   +     +     -     +  // (0,0) - (0,1) | (0,1) - (1,1)
		{  2 ,   1 ,  -1 ,  -1 ,  -1 } , //   -     -     +     +  // (1,0) - (1,1) | (0,0) - (0,1)
		{  2 ,   0 ,  -1 ,  -1 ,  -1 } , //   +     -     +     +  // (1,0) - (1,1) | (0,0) - (1,0)
		{  0 ,   1 ,  -1 ,  -1 ,  -1 } , //   -     +     +     +  // (0,0) - (1,0) | (0,0) - (0,1)
		{ -1 ,  -1 ,  -1 ,  -1 ,  -1 } , //   +     +     +     +  //
	};
	inline int MarchingSquares::AddEdgeIndices( unsigned char mcIndex , int* isoIndices )
	{
		int nEdges = 0;
		/* Square is entirely in/out of the surface */
		if( mcIndex==0 || mcIndex==15 ) return 0;

		/* Create the edges */
		for( int i=0 ; edges[mcIndex][i]!=-1 ; i+=2 )
		{
			for( int j=0 ; j<2 ; j++ ) isoIndices[i+j] = edges[mcIndex][i+j];
			nEdges++;
		}
		return nEdges;
	}
}

#endif //MARCHING_CUBES_INCLUDED
