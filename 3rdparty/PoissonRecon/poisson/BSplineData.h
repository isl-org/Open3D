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

#ifndef BSPLINE_DATA_INCLUDED
#define BSPLINE_DATA_INCLUDED

#include <string.h>

#include "BinaryNode.h"
#include "PPolynomial.h"
#include "Array.h"

enum BoundaryType
{
	BOUNDARY_FREE ,
	BOUNDARY_DIRICHLET ,
	BOUNDARY_NEUMANN ,
	BOUNDARY_COUNT
};
const char* BoundaryNames[] = { "free" , "Dirichlet" , "Neumann" };
template< BoundaryType BType > inline bool HasPartitionOfUnity( void ){ return BType!=BOUNDARY_DIRICHLET; }
inline bool HasPartitionOfUnity( BoundaryType bType ){ return bType!=BOUNDARY_DIRICHLET; }
template< BoundaryType BType , unsigned int D > struct DerivativeBoundary{};
template< unsigned int D > struct DerivativeBoundary< BOUNDARY_FREE      , D >{ static const BoundaryType BType = BOUNDARY_FREE; };
template< unsigned int D > struct DerivativeBoundary< BOUNDARY_DIRICHLET , D >{ static const BoundaryType BType = DerivativeBoundary< BOUNDARY_NEUMANN   , D-1 >::BType; };
template< unsigned int D > struct DerivativeBoundary< BOUNDARY_NEUMANN   , D >{ static const BoundaryType BType = DerivativeBoundary< BOUNDARY_DIRICHLET , D-1 >::BType; };
template< > struct DerivativeBoundary< BOUNDARY_FREE      , 0 >{ static const BoundaryType BType = BOUNDARY_FREE; };
template< > struct DerivativeBoundary< BOUNDARY_DIRICHLET , 0 >{ static const BoundaryType BType = BOUNDARY_DIRICHLET; };
template< > struct DerivativeBoundary< BOUNDARY_NEUMANN   , 0 >{ static const BoundaryType BType = BOUNDARY_NEUMANN; };


// Generate a single signature that combines the degree, boundary type, and number of supported derivatives
template< unsigned int Degree , BoundaryType BType=BOUNDARY_FREE > struct FEMDegreeAndBType { static const unsigned int Signature = Degree * BOUNDARY_COUNT + BType; };

// Extract the degree and boundary type from the signaure
template< unsigned int Signature > struct FEMSignature
{
	static const unsigned int Degree = ( Signature / BOUNDARY_COUNT );
	static const BoundaryType BType = (BoundaryType)( Signature % BOUNDARY_COUNT );
	template< unsigned int D=1 >
	static constexpr typename std::enable_if< (Degree>=D) , unsigned int >::type DSignature( void ){ return FEMDegreeAndBType< Degree-D , DerivativeBoundary< BType , D >::BType >::Signature; }
};

unsigned int FEMSignatureDegree( unsigned int signature ){ return signature / BOUNDARY_COUNT; }
BoundaryType FEMSignatureBType ( unsigned int signature ){ return (BoundaryType)( signature % BOUNDARY_COUNT ); }

static const unsigned int FEMTrivialSignature = FEMDegreeAndBType< 0 , BOUNDARY_FREE >::Signature;

// This class represents a function that is a linear combination of B-spline elements,
// with the coeff member indicating how much of each element is present.
// [WARNING] The ordering of B-spline elements is in the opposite order from that returned by Polynomial::BSplineComponent
template< unsigned int Degree >
struct BSplineElementCoefficients
{
	int coeffs[Degree+1];
	BSplineElementCoefficients( void ){ memset( coeffs , 0 , sizeof( coeffs ) ); }
	int& operator[]( int idx ){ return coeffs[idx]; }
	const int& operator[]( int idx ) const { return coeffs[idx]; }
};

// This class represents a function on the the interval, partitioned into "res" blocks.
// On each block, the function is a degree-Degree polynomial, represented by the coefficients
// in the associated BSplineElementCoefficients.
// [NOTE] This representation of a function is agnostic to the type of boundary conditions (though the constructor is not).
template< unsigned int Degree >
struct BSplineElements : public std::vector< BSplineElementCoefficients< Degree > >
{
	static const bool _Primal = (Degree&1)==1;
	static const int _Off = (Degree+1)/2;
	static int _ReflectLeft ( int offset , int res );
	static int _ReflectRight( int offset , int res );
	static int _RotateLeft  ( int offset , int res );
	static int _RotateRight ( int offset , int res );
	template< bool Left > void _addPeriodic( int offset , bool negate );
public:
	// Coefficients are ordered as "/" "-" "\"
	// [WARNING] This is the opposite of the order in Polynomial::BSplineComponent
	int denominator;

	BSplineElements( void ) { denominator = 1; }
	BSplineElements( int res , int offset , BoundaryType bType );

	void upSample( BSplineElements& high ) const;
	template< unsigned int D >
	void differentiate( BSplineElements< Degree-D >& d ) const;

	void print( FILE* fp=stdout ) const
	{
		for( int i=0 ; i<std::vector< BSplineElementCoefficients< Degree > >::size() ; i++ )
		{
			printf( "%d]" , i );
			for( int j=0 ; j<=Degree ; j++ ) printf( " %d" , (*this)[i][j] );
			printf( " (%d)\n" , denominator );
		}
	}
	Polynomial< Degree > polynomial( int idx ) const
	{
		int res = (int)std::vector< BSplineElementCoefficients< Degree > >::size();
		Polynomial< Degree > P;
		if( idx>=0 && idx<res ) for( int d=0 ; d<=Degree ; d++ ) P += Polynomial< Degree >::BSplineComponent( Degree-d ).scale( 1./res ).shift( (idx+0.)/res ) * ( (*this)[idx][d] );
		return P / denominator;
	}
	PPolynomial< Degree > pPolynomial( void ) const
	{
		int res = (int)std::vector< BSplineElementCoefficients< Degree > >::size();
		PPolynomial< Degree > P;
		P.polyCount = res + 1;
		P.polys = AllocPointer< StartingPolynomial< Degree > >( P.polyCount );
		for( int i=0 ; i<P.polyCount ; i++ ) P.polys[i].start = (i+0.) / res , P.polys[i].p = polynomial(i);
		for( int i=res ; i>=1 ; i-- ) P.polys[i].p -= P.polys[i-1].p;
		return P.compress(0);
	}
};

template< unsigned int Degree , unsigned int DDegree > struct Differentiator                   { static void Differentiate( const BSplineElements< Degree >& bse , BSplineElements< DDegree >& dbse ); };
template< unsigned int Degree >                        struct Differentiator< Degree , Degree >{ static void Differentiate( const BSplineElements< Degree >& bse , BSplineElements<  Degree >& dbse ); };

#define BSPLINE_SET_BOUNDS( name , s , e ) \
	static const int name ## Start = (s); \
	static const int name ## End   = (e); \
	static const unsigned int name ## Size  = (e)-(s)+1

// Assumes that x is non-negative
#define _FLOOR_OF_HALF( x ) (   (x)    >>1 )
#define  _CEIL_OF_HALF( x ) ( ( (x)+1 )>>1 )
// Done with the assumption
#define FLOOR_OF_HALF( x ) ( (x)<0 ? -  _CEIL_OF_HALF( -(x) ) : _FLOOR_OF_HALF( x ) )
#define  CEIL_OF_HALF( x ) ( (x)<0 ? - _FLOOR_OF_HALF( -(x) ) :  _CEIL_OF_HALF( x ) )
#define SMALLEST_INTEGER_LARGER_THAN_HALF( x ) (  CEIL_OF_HALF( (x)+1 ) )
#define LARGEST_INTEGER_SMALLER_THAN_HALF( x ) ( FLOOR_OF_HALF( (x)-1 ) )
#define SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( x ) (  CEIL_OF_HALF( x ) )
#define LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( x ) ( FLOOR_OF_HALF( x ) )

template< unsigned int Degree >
struct BSplineSupportSizes
{
protected:
	static const int _Degree = Degree;
public:
	inline static int Nodes( int depth ){ return ( 1<<depth ) + ( Degree&1 ); }
	inline static bool OutOfBounds( int depth , int offset ){ return offset>=0 || offset<Nodes(depth); }
	// An index is interiorly supported if its support is in the range [0,1<<depth)
	inline static void InteriorSupportedSpan( int depth , int& begin , int& end ){ begin = -SupportStart , end = (1<<depth)-SupportEnd; }
	inline static bool IsInteriorlySupported( int depth , int offset ){ return offset+SupportStart>=0 && offset+SupportEnd<(1<<depth); }

	// If the degree is even, we use a dual basis and functions are centered at the center of the interval
	// It the degree is odd, we use a primal basis and functions are centered at the left end of the interval
	// The function at index I is supported in:
	//	Support( I ) = [ I - (Degree+1-Inset)/2 , I + (Degree+1+Inset)/2 ]
	// [NOTE] The value of ( Degree + 1 +/- Inset ) is always even
	static const int Inset = (Degree&1) ? 0 : 1;
	BSPLINE_SET_BOUNDS(      Support , -( (_Degree+1)/2 ) , _Degree/2          );
	BSPLINE_SET_BOUNDS( ChildSupport ,     2*SupportStart , 2*(SupportEnd+1)-1 );
	BSPLINE_SET_BOUNDS(       Corner ,     SupportStart+1 , SupportEnd         );
	BSPLINE_SET_BOUNDS(  ChildCorner ,   2*SupportStart+1 , 2*SupportEnd + 1   );
	BSPLINE_SET_BOUNDS(      BCorner ,      CornerStart-1 ,      CornerEnd+1 );
	BSPLINE_SET_BOUNDS( ChildBCorner , ChildCornerStart-1 , ChildCornerEnd+1 );

	// Setting I=0, we are looking for the smallest/largest integers J such that:
	//		Support( 0 ) CONTAINS Support( J )
	// <=>	[-(Degree+1-Inset) , (Degree+1+Inset) ] CONTAINS [ J-(Degree+1-Inset)/2 , J+(Degree+1+Inset)/2 ]
	// Which is the same as the smallest/largest integers J such that:
	//		J - (Degree+1-Inset)/2 >= -(Degree+1-Inset)	| J + (Degree+1+Inset)/2 <= (Degree+1+Inset)
	// <=>	J >= -(Degree+1-Inset)/2					| J <= (Degree+1+Inset)/2
	BSPLINE_SET_BOUNDS( UpSample , - ( _Degree + 1 - Inset ) / 2 , ( _Degree + 1 + Inset ) /2 );

	// Setting I=0/1, we are looking for the smallest/largest integers J such that:
	//		Support( J ) CONTAINS Support( 0/1 )
	// <=>	[ 2*J - (Degree+1-Inset) , 2*J + (Degree+1+Inset) ] CONTAINS [ 0/1 - (Degree+1-Inset)/2 , 0/1 + (Degree+1+Inset)/2 ]
	// Which is the same as the smallest/largest integers J such that:
	//		2*J + (Degree+1+Inset) >= 0/1 + (Degree+1+Inset)/2	| 2*J - (Degree+1-Inset) <= 0/1 - (Degree+1-Inset)/2
	// <=>	2*J >= 0/1 - (Degree+1+Inset)/2						| 2*J <= 0/1 + (Degree+1-Inset)/2
	BSPLINE_SET_BOUNDS( DownSample0 , SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( 0 - ( _Degree + 1 + Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( 0 + ( _Degree + 1 - Inset ) / 2 ) );
	BSPLINE_SET_BOUNDS( DownSample1 , SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( 1 - ( _Degree + 1 + Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( 1 + ( _Degree + 1 - Inset ) / 2 ) );
	static const int DownSampleStart[] , DownSampleEnd[];
	static const unsigned int DownSampleSize[];
};
template< unsigned int Degree > const int BSplineSupportSizes< Degree >::DownSampleStart[] = { DownSample0Start , DownSample1Start };
template< unsigned int Degree > const int BSplineSupportSizes< Degree >::DownSampleEnd  [] = { DownSample0End   , DownSample1End   };
template< unsigned int Degree > const unsigned int BSplineSupportSizes< Degree >::DownSampleSize [] = { DownSample0Size  , DownSample1Size  };

template< unsigned int Degree1 , unsigned int Degree2=Degree1 >
struct BSplineOverlapSizes
{
protected:
	static const int _Degree1 = Degree1;
	static const int _Degree2 = Degree2;
public:
	typedef BSplineSupportSizes< Degree1 > EData1;
	typedef BSplineSupportSizes< Degree2 > EData2;
	BSPLINE_SET_BOUNDS(             Overlap , EData1::     SupportStart - EData2::SupportEnd , EData1::     SupportEnd - EData2::SupportStart );
	BSPLINE_SET_BOUNDS(        ChildOverlap , EData1::ChildSupportStart - EData2::SupportEnd , EData1::ChildSupportEnd - EData2::SupportStart );
	BSPLINE_SET_BOUNDS(      OverlapSupport ,      OverlapStart + EData2::SupportStart ,      OverlapEnd + EData2::SupportEnd );
	BSPLINE_SET_BOUNDS( ChildOverlapSupport , ChildOverlapStart + EData2::SupportStart , ChildOverlapEnd + EData2::SupportEnd );

	// Setting I=0/1, we are looking for the smallest/largest integers J such that:
	//		Support( 2*J ) * 2 INTERSECTION Support( 0/1 ) NON-EMPTY
	// <=>	[ 2*J - (Degree2+1-Inset2) , 2*J + (Degree2+1+Inset2) ] INTERSECTION [ 0/1 - (Degree1+1-Inset1)/2 , 0/1 + (Degree1+1+Inset1)/2 ] NON-EMPTY
	// Which is the same as the smallest/largest integers J such that:
	//		0/1 - (Degree1+1-Inset1)/2 < 2*J + (Degree2+1+Inset2)			| 0/1 + (Degree1+1+Inset1)/2 > 2*J - (Degree2+1-Inset2)	
	// <=>	2*J > 0/1 - ( 2*Degree2 + Degree1 + 3 + 2*Inset2 - Inset1 ) / 2	| 2*J < 0/1 + ( 2*Degree2 + Degree1 + 3 - 2*Inset2 + Inset1 ) / 2
	BSPLINE_SET_BOUNDS( ParentOverlap0 , SMALLEST_INTEGER_LARGER_THAN_HALF( 0 - ( 2*_Degree2 + _Degree1 + 3 + 2*EData2::Inset - EData1::Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_HALF( 0 + ( 2*_Degree2 + _Degree1 + 3 - 2*EData2::Inset + EData1::Inset ) / 2 ) );
	BSPLINE_SET_BOUNDS( ParentOverlap1 , SMALLEST_INTEGER_LARGER_THAN_HALF( 1 - ( 2*_Degree2 + _Degree1 + 3 + 2*EData2::Inset - EData1::Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_HALF( 1 + ( 2*_Degree2 + _Degree1 + 3 - 2*EData2::Inset + EData1::Inset ) / 2 ) );
	static const int ParentOverlapStart[] , ParentOverlapEnd[] , ParentOverlapSize[];
};
template< unsigned int Degree1 , unsigned int Degree2 > const int BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapStart[] = { ParentOverlap0Start , ParentOverlap1Start };
template< unsigned int Degree1 , unsigned int Degree2 > const int BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapEnd  [] = { ParentOverlap0End   , ParentOverlap1End   };
template< unsigned int Degree1 , unsigned int Degree2 > const int BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapSize [] = { ParentOverlap0Size  , ParentOverlap1Size  };

struct EvaluationData
{
	struct CornerEvaluator
	{
		virtual double value( int fIdx , int cIdx , int d ) const = 0;
		virtual void set( int depth ) = 0;
		virtual ~CornerEvaluator( void ){}
	};
	struct CenterEvaluator
	{
		virtual double value( int fIdx , int cIdx , int d ) const = 0;
		virtual void set( int depth ) = 0;
		virtual ~CenterEvaluator( void ){}
	};
	struct UpSampleEvaluator
	{
		virtual double value( int pIdx , int cIdx ) const = 0;
		virtual void set( int depth ) = 0;
		virtual ~UpSampleEvaluator( void ){}
	};
};

template< unsigned int FEMSig >
class BSplineEvaluationData
{
public:
	static const unsigned int Degree = FEMSignature< FEMSig >::Degree;
	static const int Pad = (FEMSignature< FEMSig >::BType==BOUNDARY_FREE ) ? BSplineSupportSizes< Degree >::SupportEnd : ( (Degree&1) && FEMSignature< FEMSig >::BType==BOUNDARY_DIRICHLET ) ? -1 : 0;
	inline static int Begin( int depth ){ return -Pad; }
	inline static int End  ( int depth ){ return (1<<depth) + (Degree&1) + Pad; }
	inline static bool OutOfBounds( int depth , int offset ){ return offset<Begin(depth) || offset>=End(depth); }

	static const int OffsetStart = -BSplineSupportSizes< Degree >::SupportStart , OffsetStop = BSplineSupportSizes< Degree >::SupportEnd + ( Degree&1 ) , IndexSize = OffsetStart + OffsetStop + 1 + 2 * Pad;
	static int OffsetToIndex( int depth , int offset )
	{
		int dim = BSplineSupportSizes< Degree >::Nodes( depth );
		if     ( offset<OffsetStart )     return Pad + offset;
		else if( offset>=dim-OffsetStop ) return Pad + OffsetStart + 1 + offset - ( dim-OffsetStop );
		else                              return Pad + OffsetStart;
	}
	static inline int IndexToOffset( int depth , int idx ){ return ( idx-Pad<=OffsetStart ? idx - Pad : ( BSplineSupportSizes< Degree >::Nodes(depth) + Pad - IndexSize + idx ) ); }

	BSplineEvaluationData( void );
	static double Value( int depth , int off , double s , int d );
	static double Integral( int depth , int off , double b , double e , int d );

	struct BSplineUpSamplingCoefficients
	{
	protected:
		int _coefficients[ BSplineSupportSizes< Degree >::UpSampleSize ];
	public:
		BSplineUpSamplingCoefficients( void ){ ; }
		BSplineUpSamplingCoefficients( int depth , int offset );
		double operator[] ( int idx ){ return (double)_coefficients[idx] / (1<<Degree); }
	};

	template< unsigned int D >
	struct CenterEvaluator
	{
		struct Evaluator : public EvaluationData::CenterEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _depth;
			double _ccValues[D+1][IndexSize][BSplineSupportSizes< Degree >::SupportSize];
		public:
			Evaluator( void ){ _depth = 0 ; memset( _ccValues , 0 , sizeof(_ccValues) ); }
			double value( int fIdx , int cIdx , int d ) const;
			int depth( void ) const { return _depth; }
			void set( int depth ){ BSplineEvaluationData< FEMSig >::template SetCenterEvaluator< D >( *this , depth ); }
		};
		struct ChildEvaluator : public EvaluationData::CenterEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _parentDepth;
			double _pcValues[D+1][IndexSize][BSplineSupportSizes< Degree >::ChildSupportSize];
		public:
			ChildEvaluator( void ){ _parentDepth = 0 ; memset( _pcValues , 0 , sizeof(_pcValues) ); }
			double value( int fIdx , int cIdx , int d ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
			void set( int parentDepth ){ BSplineEvaluationData< FEMSig >::template SetChildCenterEvaluator< D >( *this , parentDepth ); }
		};
	};
	template< unsigned int D > static void SetCenterEvaluator( typename CenterEvaluator< D >::Evaluator& evaluator , int depth );
	template< unsigned int D > static void SetChildCenterEvaluator( typename CenterEvaluator< D >::ChildEvaluator& evaluator , int parentDepth );

	template< unsigned int D >
	struct CornerEvaluator
	{
		struct Evaluator : public EvaluationData::CornerEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _depth;
			double _ccValues[D+1][IndexSize][BSplineSupportSizes< Degree >::BCornerSize];
		public:
			Evaluator( void ){ _depth = 0 ; memset( _ccValues , 0 , sizeof( _ccValues ) ); }
			double value( int fIdx , int cIdx , int d ) const;
			int depth( void ) const { return _depth; }
			void set( int depth ){ BSplineEvaluationData< FEMSig >::template SetCornerEvaluator< D >( *this , depth ); }
		};
		struct ChildEvaluator : public EvaluationData::CornerEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _parentDepth;
			double _pcValues[D+1][IndexSize][BSplineSupportSizes< Degree >::ChildBCornerSize];
		public:
			ChildEvaluator( void ){ _parentDepth = 0 ; memset( _pcValues , 0 , sizeof( _pcValues ) ); }
			double value( int fIdx , int cIdx , int d ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
			void set( int parentDepth ){ BSplineEvaluationData< FEMSig >::template SetChildCornerEvaluator< D >( *this , parentDepth ); }
		};
	};
	template< unsigned int D > static void SetCornerEvaluator( typename CornerEvaluator< D >::Evaluator& evaluator , int depth );
	template< unsigned int D > static void SetChildCornerEvaluator( typename CornerEvaluator< D >::ChildEvaluator& evaluator , int parentDepth );

	template< unsigned int D >
	struct Evaluator
	{
		typename CenterEvaluator< D >::Evaluator centerEvaluator;
		typename CornerEvaluator< D >::Evaluator cornerEvaluator;
		double centerValue( int fIdx , int cIdx , int d ) const { return centerEvaluator.value( fIdx , cIdx , d ); }
		double cornerValue( int fIdx , int cIdx , int d ) const { return cornerEvaluator.value( fIdx , cIdx , d ); }
	};
	template< unsigned int D > static void SetEvaluator( Evaluator< D >& evaluator , int depth ){ SetCenterEvaluator< D >( evaluator.centerEvaluator , depth ) , SetCornerEvaluator< D >( evaluator.cornerEvaluator , depth ); }
	template< unsigned int D >
	struct ChildEvaluator
	{
		typename CenterEvaluator< D >::ChildEvaluator centerEvaluator;
		typename CornerEvaluator< D >::ChildEvaluator cornerEvaluator;
		double centerValue( int fIdx , int cIdx , int d ) const { return centerEvaluator.value( fIdx , cIdx , d ); }
		double cornerValue( int fIdx , int cIdx , int d ) const { return cornerEvaluator.value( fIdx , cIdx , d ); }
	};
	template< unsigned int D > static void SetChildEvaluator( ChildEvaluator< D >& evaluator , int depth ){ SetChildCenterEvaluator< D >( evaluator.centerEvaluator , depth ) , SetChildCornerEvaluator< D >( evaluator.cornerEvaluator , depth ); }

	struct UpSampleEvaluator : public EvaluationData::UpSampleEvaluator
	{
	protected:
		friend BSplineEvaluationData;
		int _lowDepth;
		double _pcValues[IndexSize][BSplineSupportSizes< Degree >::UpSampleSize];
	public:
		UpSampleEvaluator( void ){ _lowDepth = 0 ; memset( _pcValues , 0 , sizeof( _pcValues ) ); }
		double value( int pIdx , int cIdx ) const;
		int lowDepth( void ) const { return _lowDepth; }
		void set( int lowDepth ){ BSplineEvaluationData::SetUpSampleEvaluator( *this , lowDepth ); }
	};
	static void SetUpSampleEvaluator( UpSampleEvaluator& evaluator , int lowDepth );
};

template< unsigned int FEMSig1 , unsigned int FEMSig2 >
class BSplineIntegrationData
{
public:
	static const unsigned int Degree1 = FEMSignature< FEMSig1 >::Degree;
	static const unsigned int Degree2 = FEMSignature< FEMSig2 >::Degree;
	static const int OffsetStart = - BSplineOverlapSizes< Degree1 , Degree2 >::OverlapSupportStart;
	static const int OffsetStop  =   BSplineOverlapSizes< Degree1 , Degree2 >::OverlapSupportEnd + ( Degree1&1 );
	static const int IndexSize = OffsetStart + OffsetStop + 1 + 2 * BSplineEvaluationData< FEMSig1 >::Pad;
	static int OffsetToIndex( int depth , int offset )
	{
		int dim = BSplineSupportSizes< Degree1 >::Nodes( depth );
		if     ( offset<OffsetStart )     return BSplineEvaluationData< FEMSig1 >::Pad + offset;
		else if( offset>=dim-OffsetStop ) return BSplineEvaluationData< FEMSig1 >::Pad + OffsetStart + 1 + offset - ( dim-OffsetStop );
		else                              return BSplineEvaluationData< FEMSig1 >::Pad + OffsetStart;
	}
	static inline int IndexToOffset( int depth , int idx ){ return ( idx-BSplineEvaluationData< FEMSig1 >::Pad<=OffsetStart ? idx-BSplineEvaluationData< FEMSig1 >::Pad : ( BSplineSupportSizes< Degree1 >::Nodes(depth) + BSplineEvaluationData< FEMSig1 >::Pad - IndexSize + idx ) ); }

	template< unsigned int D1 , unsigned int D2 > static double Dot( int depth1 , int off1 , int depth2 , int off2 );
	// An index is interiorly overlapped if the support of its overlapping neighbors is in the range [0,1<<depth)
	inline static void InteriorOverlappedSpan( int depth , int& begin , int& end ){ begin = -BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart-BSplineSupportSizes< Degree2 >::SupportStart , end = (1<<depth)-BSplineOverlapSizes< Degree1 , Degree2 >::OverlapEnd-BSplineSupportSizes< Degree2 >::SupportEnd; }

	struct FunctionIntegrator
	{
		template< unsigned int D1=Degree1 , unsigned int D2=Degree2 >
		struct Integrator
		{
		protected:
			friend BSplineIntegrationData;
			int _depth;
			double _ccIntegrals[D1+1][D2+1][IndexSize][BSplineOverlapSizes< Degree1 , Degree2 >::OverlapSize];
		public:
			Integrator( void )
			{
				_depth = 0;
				memset(_ccIntegrals, 0, sizeof(_ccIntegrals));
			}
			double dot( int fIdx1 , int fidx2 , int d1 , int d2 ) const;
			int depth( void ) const { return _depth; }
			void set( int depth ){ BSplineIntegrationData::SetIntegrator( *this , depth ); }
		};
		template< unsigned int D1=Degree1 , unsigned int D2=Degree2 >
		struct ChildIntegrator
		{
		protected:
			friend BSplineIntegrationData;
			int _parentDepth;
			double _pcIntegrals[D1+1][D2+1][IndexSize][BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapSize];
		public:
			ChildIntegrator( void )
			{
				_parentDepth = 0;
				memset( _pcIntegrals , 0 , sizeof( _pcIntegrals ) ); 
			}
			double dot( int fIdx1 , int fidx2 , int d1 , int d2 ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
			void set( int depth ){ BSplineIntegrationData::SetChildIntegrator( *this , depth ); }
		};
	};
	// D1 and D2 indicate the number of derivatives that should be taken
	template< unsigned int D1 , unsigned int D2 >
	static void SetIntegrator( typename FunctionIntegrator::template Integrator< D1 , D2 >& integrator , int depth );
	template< unsigned int D1 , unsigned int D2 >
	static void SetChildIntegrator( typename FunctionIntegrator::template ChildIntegrator< D1 , D2 >& integrator , int parentDepth );

protected:
	// _D1 and _D2 indicate the total number of derivatives the integrator will be storing
	template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 >
	struct _IntegratorSetter
	{
		static void Set( typename FunctionIntegrator::template      Integrator< _D1 , _D2 >& integrator , int depth );
		static void Set( typename FunctionIntegrator::template ChildIntegrator< _D1 , _D2 >& integrator , int depth );
	};

	template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
	struct IntegratorSetter
	{
		static void Set2D( Integrator& integrator , int depth );
		static void Set1D( Integrator& integrator , int depth );
	};
	template< unsigned int D1 , unsigned int _D1 , unsigned int _D2 , class Integrator >
	struct IntegratorSetter< D1 , 0 , _D1 , _D2 , Integrator >
	{
		static void Set2D( Integrator& integrator , int  depth );
		static void Set1D( Integrator& integrator , int  depth );
	};
	template< unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
	struct IntegratorSetter< 0 , D2 , _D1 , _D2 , Integrator >
	{
		static void Set2D( Integrator& integrator , int  depth );
		static void Set1D( Integrator& integrator , int  depth );
	};
	template< unsigned int _D1 , unsigned int _D2 , class Integrator >
	struct IntegratorSetter< 0 , 0 , _D1 , _D2 , Integrator >
	{
		static void Set2D( Integrator& integrator , int  depth );
		static void Set1D( Integrator& integrator , int  depth );
	};
};
#undef BSPLINE_SET_BOUNDS
#undef _FLOOR_OF_HALF
#undef  _CEIL_OF_HALF
#undef FLOOR_OF_HALF
#undef  CEIL_OF_HALF
#undef SMALLEST_INTEGER_LARGER_THAN_HALF
#undef LARGEST_INTEGER_SMALLER_THAN_HALF
#undef SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF
#undef LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF


template< unsigned int FEMSig , unsigned int D=0 >
struct BSplineData
{
	static const unsigned int Degree = FEMSignature< FEMSig >::Degree;
	static const int _Degree = Degree;
	// Note that this struct stores the components in left-to-right order
	struct BSplineComponents
	{
		BSplineComponents( void ){ ; }
		BSplineComponents( int depth , int offset );
		const Polynomial< Degree >* operator[] ( int idx ) const { return _polys[idx]; }
	protected:
		Polynomial< Degree > _polys[Degree+1][D+1];
	};
	struct SparseBSplineEvaluator
	{
		void init( unsigned int depth )
		{
			_depth = depth , _width = 1./(1<<depth);
			// _preStart + BSplineSupportSizes< _Degree >::SupportEnd >=0
			_preStart = -BSplineSupportSizes< _Degree >::SupportEnd;
			// _postStart + BSplineSupportSizes< _Degree >::SupportEnd <= (1<<depth)-1
			_postStart = (1<<depth) - 1 - BSplineSupportSizes< _Degree >::SupportEnd;
			_preEnd = _preStart + _Degree + 1;
			_postEnd = _postStart + _Degree + 1;
			_centerIndex = ( ( _preStart + _Degree + 1 ) + ( _postStart - 1 ) ) / 2;
			_centerComponents = BSplineComponents( depth , _centerIndex );
			for( int i=0 ; i<=Degree ; i++ ) _preComponents[i] = BSplineComponents( depth , _preStart+i ) , _postComponents[i] = BSplineComponents( depth , _postStart+i );
		}
		double value( double p ,            int fIdx , int d ) const { return value( p , (int)( p * (1<<_depth ) ) , fIdx , d ); }
		double value( double p , int pIdx , int fIdx , int d ) const
		{
			if     ( fIdx<_preStart  ) return 0;
			else if( fIdx<_preEnd    ) return _preComponents [fIdx-_preStart ][pIdx-fIdx+_LeftSupportRadius][d]( p );
			else if( fIdx<_postStart ) return _centerComponents               [pIdx-fIdx+_LeftSupportRadius][d]( p+_width*(_centerIndex-fIdx) );
			else if( fIdx<_postEnd   ) return _postComponents[fIdx-_postStart][pIdx-fIdx+_LeftSupportRadius][d]( p );
			else                       return 0;
		}
		const Polynomial< _Degree >* polynomialsAndOffset( double& p ,            int fIdx ) const { return polynomialsAndOffset( p , (int)( p * (1<<_depth ) ) , fIdx ); }
		const Polynomial< _Degree >* polynomialsAndOffset( double& p , int pIdx , int fIdx ) const
		{
			if     ( fIdx<_preEnd    ){                                   return _preComponents [fIdx-_preStart ][pIdx-fIdx+_LeftSupportRadius]; }
			else if( fIdx<_postStart ){ p += _width*(_centerIndex-fIdx) ; return _centerComponents               [pIdx-fIdx+_LeftSupportRadius]; }
			else                      {                                   return _postComponents[fIdx-_postStart][pIdx-fIdx+_LeftSupportRadius]; }
		}
	protected:
		static const int _LeftSupportRadius = -BSplineSupportSizes< _Degree >::SupportStart;
		BSplineComponents _preComponents[_Degree+1] , _postComponents[_Degree+1] ,_centerComponents;
		int _preStart , _preEnd , _postStart , _postEnd , _centerIndex;
		unsigned int _depth;
		double _width;
	};
	const SparseBSplineEvaluator& operator[]( int depth ) const { return _evaluators[depth]; }

	inline static int RemapOffset( int depth , int idx , bool& reflect );

	BSplineData( void );
	void reset( int maxDepth );
	BSplineData( int maxDepth );
	~BSplineData( void );

protected:
	unsigned int _maxDepth;
	Pointer( SparseBSplineEvaluator ) _evaluators;
};

template< unsigned int Degree1 , unsigned int Degree2 > void SetBSplineElementIntegrals( double integrals[Degree1+1][Degree2+1] );


#include "BSplineData.inl"
#endif // BSPLINE_DATA_INCLUDED