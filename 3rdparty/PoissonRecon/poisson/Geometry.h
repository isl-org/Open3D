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

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

#include <stdio.h>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <unordered_map>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#endif // _WIN32

template< class Real > Real Random( void );

template< class Real , unsigned int Dim > struct XForm;

template< class Real , unsigned int Dim >
struct Point
{
	void _init( unsigned int d )
	{
		if( !d ) memset( coords , 0 , sizeof(Real)*Dim );
		else ERROR_OUT( "Should never be called" );
	}
	template< class _Real , class ... _Reals > void _init( unsigned int d , _Real v , _Reals ... values )
	{
		coords[d] = (Real)v;
		if( d+1<Dim ) _init( d+1 , values... );
	}
	template< class ... Points >
	static void _AddColumnVector( XForm< Real , Dim >& x , unsigned int c , Point point , Points ... points )
	{
		for( unsigned int r=0 ; r<Dim ; r++ ) x( c , r ) = point[r];
		_AddColumnVector( x , c+1 , points ... );
	}
	static void _AddColumnVector( XForm< Real , Dim >& x , unsigned int c ){ ; }
public:
	Real coords[Dim];
	Point( void ) { memset( coords , 0 , sizeof(Real)*Dim ); }
	Point( const Point& p ){ memcpy( coords , p.coords , sizeof(Real)*Dim ); }
	template< class ... _Reals > Point( _Reals ... values ){ static_assert( sizeof...(values)==Dim || sizeof...(values)==0 , "[ERROR] Point::Point: Invalid number of coefficients" ) ; _init( 0 , values... ); }
	template< class _Real > Point( const Point< _Real , Dim >& p ){ for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] = (Real) p.coords[d]; }
	inline       Real& operator[] ( unsigned int i )       { return coords[i]; }
	inline const Real& operator[] ( unsigned int i ) const { return coords[i]; }
	inline Point  operator - ( void ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = - coords[d] ; return q; }


	template< class _Real > inline Point& operator += ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] += (Real)p.coords[d] ; return *this; }
	template< class _Real > inline Point  operator +  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] + (Real)p.coords[d] ; return q; }
	template< class _Real > inline Point& operator -= ( Point< _Real , Dim > p )       { return (*this)+=(-p); }
	template< class _Real > inline Point  operator -  ( Point< _Real , Dim > p ) const { return (*this)+ (-p); }
	template< class Scalar > inline Point& operator *= ( Scalar r )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] *= r ; return *this; }
	template< class Scalar > inline Point  operator *  ( Scalar r ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] * r ; return q; }
	template< class Scalar > inline Point& operator /= ( Scalar r )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] /= r ; return *this; }
	template< class Scalar > inline Point  operator /  ( Scalar r ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] / r ; return q; }
	template< class _Real > inline Point& operator *= ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] *= p.coords[d] ; return *this; }
	template< class _Real > inline Point  operator *  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] * p.coords[d] ; return q; }
	template< class _Real > inline Point& operator /= ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] /= p.coords[d] ; return *this; }
	template< class _Real > inline Point  operator /  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] / p.coords[d] ; return q; }

	static Real Dot( const Point& p1 , const Point& p2 ){ Real dot = {} ; for( unsigned int d=0 ; d<Dim ; d++ ) dot += p1.coords[d] * p2.coords[d] ; return dot; }
	static Real SquareNorm( const Point& p ){ return Dot( p , p ); }
	template< class ... Points > static Point CrossProduct( Points ... points )
	{
		static_assert( sizeof ... ( points )==Dim-1 , "Number of points in cross-product must be one less than the dimension" );
		XForm< Real , Dim > x;
		_AddColumnVector( x , 0 , points ... );
		Point p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( d&1 ) ? -x.subDeterminant( Dim-1 , d ) : x.subDeterminant( Dim-1 , d );
		return p;
	}
	static Point CrossProduct( const Point* points )
	{
		XForm< Real , Dim > x;
		for( unsigned int d=0 ; d<Dim-1 ; d++ ) for( unsigned int c=0 ; c<Dim ; c++ ) x(d,c) = points[d][c];
		Point p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( d&1 ) ? -x.subDeterminant( Dim-1 , d ) : x.subDeterminant( Dim-1 , d );
		return p;
	}
	static Point CrossProduct( Point* points ){ return CrossProduct( ( const Point* )points ); }
};
template< class Real , unsigned int Dim > Point< Real , Dim > operator * ( Real r , Point< Real , Dim > p ){ return p*r; }
template< class Real , unsigned int Dim > Point< Real , Dim > operator / ( Real r , Point< Real , Dim > p ){ return p/r; }

template< class Real , unsigned int _Columns , unsigned int _Rows >
struct Matrix
{
	static const unsigned int Columns = _Columns;
	static const unsigned int Rows = _Rows;
	Real coords[Columns][Rows];
	Matrix( void ) { memset( coords , 0 , sizeof(coords) ); }
	inline       Real& operator() ( unsigned int c , unsigned int r )       { return coords[c][r]; }
	inline const Real& operator() ( unsigned int c , unsigned int r ) const { return coords[c][r]; }
	inline       Real* operator[] ( unsigned int c                  )       { return coords[c]   ; }
	inline const Real* operator[] ( unsigned int c                  ) const { return coords[c]   ; }

	inline Matrix  operator - ( void ) const { Matrix m ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) m.coords[c][r] = - coords[c][r] ; return m; }

	inline Matrix& operator += ( const Matrix& m ){ for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) coords[c][r] += m.coords[c][r] ; return *this; }
	inline Matrix  operator +  ( const Matrix& m ) const { Matrix n ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) n.coords[c][r] = coords[c][r] + m.coords[c][r] ; return n; }
	inline Matrix& operator *= ( Real s ) { for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) coords[c][r] *= s ; return *this; }
	inline Matrix  operator *  ( Real s ) const { Matrix n ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) n.coords[c][r] = coords[c][r] * s ; return n; }

	inline Matrix& operator -= ( const Matrix& m ){ return ( (*this)+=(-m) ); }
	inline Matrix  operator -  ( const Matrix& m ) const { return (*this)+(-m); }
	inline Matrix& operator /= ( Real s ){ return ( (*this)*=(Real)(1./s) ); }
	inline Matrix  operator /  ( Real s ) const { return (*this) * ( (Real)(1./s) ); }

	static Real Dot( const Matrix& m1 , const Matrix& m2 ){ Real dot = (Real)0 ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) dot += m1.coords[c][r] * m2.coords[c][r] ; return dot; }

	template< typename T >
	inline Point< T , Rows > operator* ( const Point< T , Columns >& p ) const { Point< T , Rows > q ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) q[r] += (T)( p[c] * coords[c][r] ) ; return q; }
};

template< class Real , unsigned int Dim >
struct XForm
{
	Real coords[Dim][Dim];
	XForm( void ) { memset( coords , 0 , sizeof(Real) * Dim * Dim ); }
	static XForm Identity( void )
	{
		XForm xForm;
		for( unsigned int d=0 ; d<Dim ; d++ ) xForm(d,d) = (Real)1.;
		return xForm;
	}
	Real& operator() ( unsigned int i , unsigned int j ){ return coords[i][j]; }
	const Real& operator() ( unsigned int i , unsigned int j ) const { return coords[i][j]; }
	template< class _Real > Point< _Real , Dim-1 > operator * ( const Point< _Real , Dim-1 >& p ) const
	{
		Point< _Real , Dim-1 > q;
		for( unsigned int i=0 ; i<Dim-1 ; i++ )
		{
			for( unsigned int j=0 ; j<Dim-1 ; j++ ) q[i] += (_Real)( coords[j][i] * p[j] );
			q[i] += (_Real)coords[Dim-1][i];
		}
		return q;
	}
	template< class _Real > Point< _Real , Dim > operator * ( const Point< _Real , Dim >& p ) const
	{
		Point< _Real , Dim > q;
		for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) q[i] += (_Real)( coords[j][i] * p[j] );
		return q;
	}
	XForm operator * ( const XForm& m ) const
	{
		XForm n;
		for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) for( unsigned int k=0 ; k<Dim ; k++ ) n.coords[i][j] += m.coords[i][k]*coords[k][j];
		return n;
	}
	XForm transpose( void ) const
	{
		XForm xForm;
		for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) xForm( i , j ) = coords[j][i];
		return xForm;
	}
	Real determinant( void ) const
	{
		Real det = (Real)0.;
		for( unsigned int d=0 ; d<Dim ; d++ )
			if( d&1 ) det -= coords[d][0] * subDeterminant( d , 0 );
			else      det += coords[d][0] * subDeterminant( d , 0 );
		return det;
	}
	XForm inverse( void ) const
	{
		XForm xForm;
		Real d = determinant();
		for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ )
			if( (i+j)%2==0 ) xForm.coords[j][i] =  subDeterminant( i , j ) / d;
			else             xForm.coords[j][i] = -subDeterminant( i , j ) / d;
		return xForm;
	}
	Real subDeterminant( unsigned int i , unsigned int j ) const
	{
		XForm< Real , Dim-1 > xForm;
		unsigned int ii[Dim-1] , jj[Dim-1];
		for( unsigned int a=0 , _i=0 , _j=0 ; a<Dim ; a++ )
		{
			if( a!=i ) ii[_i++] = a;
			if( a!=j ) jj[_j++] = a;
		}
		for( unsigned int _i=0 ; _i<Dim-1 ; _i++ ) for( unsigned int _j=0 ; _j<Dim-1 ; _j++ ) xForm( _i , _j ) = coords[ ii[_i] ][ jj[_j] ];
		return xForm.determinant();
	}

	inline XForm  operator - ( void ) const { XForm m ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) m.coords[c][r] = - coords[c][r] ; return m; }

	inline XForm& operator += ( const XForm& m ){ for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) coords[c][r] += m.coords[c][r] ; return *this; }
	inline XForm  operator +  ( const XForm& m ) const { XForm n ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) n.coords[c][r] = coords[c][r] + m.coords[c][r] ; return n; }
	inline XForm& operator *= ( Real s ) { for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) coords[c][r] *= s ; return *this; }
	inline XForm  operator *  ( Real s ) const { XForm n ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) n.coords[c][r] = coords[c][r] * s ; return n; }

	inline XForm& operator -= ( const XForm& m ){ return ( (*this)+=(-m) ); }
	inline XForm  operator -  ( const XForm& m ) const { return (*this)+(-m); }
	inline XForm& operator /= ( Real s ){ return ( (*this)*=(Real)(1./s) ); }
	inline XForm  operator /  ( Real s ) const { return (*this) * ( (Real)(1./s) ); }
};
template<>
inline XForm< float , 1 > XForm< float , 1 >::inverse( void ) const
{
	XForm< float , 1 > x;
	x.coords[0][0] = (float)(1./coords[0][0] );
	return x;
}
template<>
inline XForm< double , 1 > XForm< double , 1 >::inverse( void ) const
{
	XForm< double , 1 > x;
	x.coords[0][0] = (double)(1./coords[0][0] );
	return x;
}
template<> inline float  XForm< float  , 1 >::determinant( void ) const { return coords[0][0]; }
template<> inline double XForm< double , 1 >::determinant( void ) const { return coords[0][0]; }

template< class Real , unsigned int Dim >
struct OrientedPoint
{
	Point< Real , Dim > p , n;
	OrientedPoint( Point< Real , Dim > pp = Point< Real , Dim >() , Point< Real , Dim > nn=Point< Real , Dim >() ) : p(pp) , n(nn) { ; }
	template< class _Real > OrientedPoint( const OrientedPoint< _Real , Dim>& p ) : OrientedPoint( Point< Real , Dim >( p.p ) , Point< Real , Dim >( p.n ) ){ ; }

	template< class _Real > inline OrientedPoint& operator += ( OrientedPoint< _Real , Dim > _p ){ p += _p.p , n += _p.n ; return *this; }
	template< class _Real > inline OrientedPoint  operator +  ( OrientedPoint< _Real , Dim > _p ) const { return OrientedPoint< Real , Dim >( p+_p.p , n+_p.n ); }
	template< class _Real > inline OrientedPoint& operator *= ( _Real r ) { p *= r , n *= r ; return *this; }
	template< class _Real > inline OrientedPoint  operator *  ( _Real r ) const { return OrientedPoint< Real , Dim >( p*r , n*r ); }

	template< class _Real > inline OrientedPoint& operator -= ( OrientedPoint< _Real , Dim > p ){ return ( (*this)+=(-p) ); }
	template< class _Real > inline OrientedPoint  operator -  ( OrientedPoint< _Real , Dim > p ) const { return (*this)+(-p); }
	template< class _Real > inline OrientedPoint& operator /= ( _Real r ){ return ( (*this)*=Real(1./r) ); }
	template< class _Real > inline OrientedPoint  operator /  ( _Real r ) const { return (*this) * ( Real(1.)/r ); }
};


template< class Data , class Real >
struct ProjectiveData
{
	Data data;
	Real weight;
	ProjectiveData( Data d=Data() , Real w=(Real)0 ) : data(d) , weight(w) { ; }
	operator Data (){ return weight!=0 ? data/weight : data*weight; }
	Data value( void ) const { return weight!=0 ? data * (1/weight) : data*weight; }
	ProjectiveData& operator += ( const ProjectiveData& p ){ data += p.data , weight += p.weight ; return *this; }
	ProjectiveData& operator -= ( const ProjectiveData& p ){ data -= p.data , weight -= p.weight ; return *this; }
	ProjectiveData& operator *= ( Real s ){ data *= s , weight *= s ; return *this; }
	ProjectiveData& operator /= ( Real s ){ data /= s , weight /= s ; return *this; }
	ProjectiveData  operator +  ( const ProjectiveData& p ) const { return ProjectiveData( data+p.data , weight+p.weight ); }
	ProjectiveData  operator -  ( const ProjectiveData& p ) const { return ProjectiveData( data-p.data , weight-p.weight ); }
	ProjectiveData  operator *  ( Real s ) const { return ProjectiveData( data*s , weight*s ); }
	ProjectiveData  operator /  ( Real s ) const { return ProjectiveData( data/s , weight/s ); }
};

template< class Real , unsigned int Dim > Point< Real , Dim > RandomBallPoint( void );
template< class Real , unsigned int Dim > Point< Real , Dim > RandomSpherePoint( void );
template< class Real , unsigned int Dim > Real Length( Point< Real , Dim > p ){ return (Real)sqrt( Point< Real , Dim >::SquareNorm( p ) ); }
template< class Real , unsigned int Dim > Real SquareLength( Point< Real , Dim > p ){ return Point< Real , Dim >::SquareNorm( p ); }
template< class Real , unsigned int Dim > Real Distance( Point< Real , Dim > p1 , Point< Real , Dim > p2 ){ return Length(p1-p2); }
template< class Real , unsigned int Dim > Real SquareDistance( Point< Real , Dim > p1 , Point< Real , Dim > p2 ){ return SquareLength( p1-p2 ); }
template< class Real > Point< Real , 3 > CrossProduct( Point< Real , 3 > p1 , Point< Real , 3 > p2 ){ return Point< Real , 3 >::CrossProduct( p1 , p2 ); }

template< class Real , unsigned int Dim > Real SquareArea( Point< Real , Dim > p1 , Point< Real , Dim > p2 , Point< Real , Dim > p3 )
{
	Point< Real , Dim > v1 = p2-p1 , v2 = p3-p1;
	// Area^2 = ( |v1|^2 * |v2|^2 * sin^2( < v1 ,v2 ) ) / 4
	//        = ( |v1|^2 * |v2|^2 * ( 1 - cos^2( < v1 ,v2 ) ) ) / 4
	//        = ( |v1|^2 * |v2|^2 * ( 1 - < v1 , v2 >^2 / ( |v1|^2 * |v2|^2 ) ) ) / 4
	//        = ( |v1|^2 * |v2|^2 - < v1 , v2 >^2 ) / 4
	Real dot = Point< Real , Dim >::Dot( v1 , v2 );
	Real l1 = Point< Real , Dim >::SquareNorm( v1 ) , l2 = Point< Real , Dim >::SquareNorm( v2 );
	return ( l1 * l2 - dot * dot ) / 4;
}
template< class Real , unsigned int Dim > Real Area( Point< Real , Dim > p1 , Point< Real , Dim > p2 , Point< Real , Dim > p3 ){ return (Real)sqrt( SquareArea( p1 , p2 , p3 ) ); }

template< unsigned int K > struct Factorial{ static const unsigned long long Value = Factorial< K-1 >::Value * K; };
template<> struct Factorial< 0 >{ static const unsigned long long Value = 1; };

template< class Real , unsigned int Dim , unsigned int K >
struct Simplex
{
	Point< Real , Dim > p[K+1];
	Simplex( void ){ static_assert( K<=Dim , "[ERROR] Bad simplex dimension" ); }
	Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
	const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
	Real measure( void ) const { return (Real)sqrt( squareMeasure() ); }
	Real squareMeasure( void ) const
	{
		XForm< Real , K > mass;
		for( unsigned int i=1 ; i<=K ; i++ ) for( unsigned int j=1 ; j<=K ; j++ ) mass(i-1,j-1) = Point< Real , Dim >::Dot( p[i]-p[0] , p[j]-p[0] );
		return mass.determinant() / ( Factorial< K >::Value * Factorial< K >::Value );
	}
	Point< Real , Dim > center( void ) const
	{
		Point< Real , Dim > c;
		for( unsigned int k=0 ; k<=K ; k++ ) c += p[k];
		return c / (K+1);
	}
	void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const;
};
template< class Real , unsigned int Dim >
struct Simplex< Real , Dim , 0 >
{
	Point< Real , Dim > p[1];
	Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
	const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
	Real squareMeasure( void ) const { return (Real)1.; }
	Real measure( void ) const { return (Real)1.; }
	Point< Real , Dim > center( void ) const { return p[0]; }
	void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
	{
		if( Point< Real , Dim >::Dot( p[0] , pNormal ) < pOffset ) back.push_back( *this );
		else                                                       front.push_back( *this );
	}
};
template< class Real , unsigned int Dim > using Edge = Simplex< Real , Dim , 1 >;
template< class Real , unsigned int Dim > using Triangle = Simplex< Real , Dim , 2 >;

template< unsigned int K , typename Index >
struct SimplexIndex
{
	Index idx[K+1];
	template< class ... Ints >
	SimplexIndex( Ints ... values ){ static_assert( sizeof...(values)==K+1 || sizeof...(values)==0 , "[ERROR] Invalid number of coefficients" ) ; _init( 0 , values ... ); }
	Index &operator[] ( unsigned int i ) { return idx[i] ;}
	const Index &operator[] ( unsigned int i ) const { return idx[i]; }
protected:
	void _init( unsigned int k )
	{
		if( !k ) memset( idx , 0 , sizeof(idx) );
		else ERROR_OUT( "Should never be called" );
	}
	template< class ... Ints > void _init( unsigned int k , Index v , Ints ... values )
	{
		idx[k] = v;
		if( k<=K ) _init( k+1 , values ... );
	}
};
template< typename Index > using EdgeIndex = SimplexIndex< 1 , Index >;
template< typename Index > using TriangleIndex = SimplexIndex< 2 , Index >;

template< typename Index >
class CoredPointIndex
{
public:
	Index index;
	char inCore;

	bool operator == (const CoredPointIndex& cpi) const {return (index==cpi.index) && (inCore==cpi.inCore);};
	bool operator != (const CoredPointIndex& cpi) const {return (index!=cpi.index) || (inCore!=cpi.inCore);};
};
template< typename Index > struct CoredEdgeIndex{ CoredPointIndex< Index > idx[2]; };

template< typename Index >
class TriangulationEdge
{
public:
	TriangulationEdge( void ){ pIndex[0] = pIndex[1] = tIndex[0] = tIndex[1] = -1; }
	Index pIndex[2] , tIndex[2];
};

template< typename Index >
class TriangulationTriangle
{
public:
	TriangulationTriangle( void ){ eIndex[0] = eIndex[1] = eIndex[2] = -1; }
	Index eIndex[3];
};

template< typename Index >
struct CoredVertexIndex
{
	Index idx;
	bool inCore;
};

template< class Vertex , typename Index >
class CoredCurveData
{
public:
	std::vector< Vertex > inCorePoints;
	virtual void resetIterator( void ) = 0;

	virtual Index addOutOfCorePoint( const Vertex& p ) = 0;
	virtual Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p ) = 0;
	virtual void addEdge_s( unsigned int thread , CoredVertexIndex< Index > v1 , CoredVertexIndex< Index > v2 ) = 0;
	virtual void addEdge_s( unsigned int thread , Index v1 , Index v2 ) = 0;

	virtual Index nextOutOfCorePoint( Vertex& p )=0;
	virtual Index nextEdge( CoredVertexIndex< Index >& v1 , CoredVertexIndex< Index >& v2 ) = 0;

	virtual size_t outOfCorePointCount(void)=0;
	virtual size_t edgeCount( void ) = 0;
};

template< class Vertex , typename Index >
class CoredMeshData
{
public:
	virtual ~CoredMeshData( void ){}
	std::vector< Vertex > inCorePoints;
	virtual void resetIterator( void ) = 0;

	virtual Index addOutOfCorePoint( const Vertex& p ) = 0;
	virtual Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p ) = 0;
	virtual void addPolygon_s( unsigned int thread , const std::vector< CoredVertexIndex< Index > >& vertices ) = 0;
	virtual void addPolygon_s( unsigned int thread , const std::vector< Index >& vertices ) = 0;

	virtual Index nextOutOfCorePoint( Vertex& p )=0;
	virtual Index nextPolygon( std::vector< CoredVertexIndex< Index > >& vertices ) = 0;

	virtual size_t outOfCorePointCount( void )=0;
	virtual size_t polygonCount( void ) = 0;
};

template< class Vertex , typename Index >
class CoredVectorCurveData : public CoredCurveData< Vertex , Index >
{
	std::vector< Vertex > oocPoints;
	std::vector< std::pair< Index , Index > > edges;
	unsigned int threadIndex;
	Index edgeIndex;
	Index oocPointIndex;
public:
	CoredVectorCurveData( void );

	void resetIterator( void );

	Index addOutOfCorePoint( const Vertex& p );
	Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p );
	void addEdge_s( unsigned int thread , CoredVertexIndex< Index > v1 , CoredVertexIndex< Index > v2 );
	void addEdge_s( unsigned int thread , Index v1 , Index v2 );

	Index nextOutOfCorePoint( Vertex& p );
	Index nextEdge( CoredVertexIndex< Index > &v1 , CoredVertexIndex< Index > &v2 );

	size_t outOfCorePointCount( void );
	size_t edgeCount( void );
};
template< class Vertex , typename Index >
class CoredVectorMeshData : public CoredMeshData< Vertex , Index >
{
	std::vector< Vertex > oocPoints;
	std::vector< std::vector< std::vector< Index > > > polygons;
	unsigned int threadIndex;
	Index polygonIndex;
	Index oocPointIndex;
public:
	CoredVectorMeshData( void );

	void resetIterator( void );

	Index addOutOfCorePoint( const Vertex& p );
	Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p );
	void addPolygon_s( unsigned int thread , const std::vector< CoredVertexIndex< Index > >& vertices );
	void addPolygon_s( unsigned int thread , const std::vector< Index >& vertices );

	Index nextOutOfCorePoint( Vertex& p );
	Index nextPolygon( std::vector< CoredVertexIndex< Index > >& vertices );

	size_t outOfCorePointCount( void );
	size_t polygonCount( void );
};
class BufferedReadWriteFile
{
	bool tempFile;
	FILE* _fp;
	char *_buffer , _fileName[1024];
	size_t _bufferIndex , _bufferSize;
public:
	BufferedReadWriteFile( const char* fileName=NULL , const char* fileHeader="" , unsigned int bufferSize=(1<<20) )
	{
		_bufferIndex = 0;
		_bufferSize = bufferSize;
		if( fileName ) strcpy( _fileName , fileName ) , tempFile = false , _fp = fopen( _fileName , "w+b" );
		else
		{
			if( fileHeader && strlen(fileHeader) ) sprintf( _fileName , "%sXXXXXX" , fileHeader );
			else strcpy( _fileName , "XXXXXX" );
#ifdef _WIN32
			_mktemp( _fileName );
			_fp = fopen( _fileName , "w+b" );
#else // !_WIN32
			_fp = fdopen( mkstemp( _fileName ) , "w+b" );
#endif // _WIN32
			tempFile = true;
		}
		if( !_fp ) ERROR_OUT( "Failed to open file: " , _fileName );
		_buffer = (char*) malloc( _bufferSize );
	}
	~BufferedReadWriteFile( void )
	{
		free( _buffer );
		fclose( _fp );
		if( tempFile ) remove( _fileName );
	}
	bool write( const void* data , size_t size )
	{
		if( !size ) return true;
		const char* _data = (char*) data;
		size_t sz = _bufferSize - _bufferIndex;
		while( sz<=size )
		{
			memcpy( _buffer+_bufferIndex , _data , sz );
			fwrite( _buffer , 1 , _bufferSize , _fp );
			_data += sz;
			size -= sz;
			_bufferIndex = 0;
			sz = _bufferSize;
		}
		if( size )
		{
			memcpy( _buffer+_bufferIndex , _data , size );
			_bufferIndex += size;
		}
		return true;
	}
	bool read( void* data , size_t size )
	{
		if( !size ) return true;
		char *_data = (char*) data;
		size_t sz = _bufferSize - _bufferIndex;
		while( sz<=size )
		{
			if( size && !_bufferSize ) return false;
			memcpy( _data , _buffer+_bufferIndex , sz );
			_bufferSize = fread( _buffer , 1 , _bufferSize , _fp );
			_data += sz;
			size -= sz;
			_bufferIndex = 0;
			if( !size ) return true;
			sz = _bufferSize;
		}
		if( size )
		{
			if( !_bufferSize ) return false;
			memcpy( _data , _buffer+_bufferIndex , size );
			_bufferIndex += size;
		}
		return true;
	}
	void reset( void )
	{
		if( _bufferIndex ) fwrite( _buffer , 1 , _bufferIndex , _fp );
		_bufferIndex = 0;
		fseek( _fp , 0 , SEEK_SET );
		_bufferIndex = 0;
		_bufferSize = fread( _buffer , 1 , _bufferSize , _fp );
	}
};
template< class Vertex , typename Index >
class CoredFileCurveData : public CoredCurveData< Vertex , Index >
{
	BufferedReadWriteFile *oocPointFile;
	Index oocPoints;
	std::vector< BufferedReadWriteFile* > edgeFiles;
	unsigned int threadIndex;
public:
	CoredFileCurveData( const char* fileHeader="" );
	~CoredFileCurveData( void );

	void resetIterator( void );

	Index addOutOfCorePoint( const Vertex& p );
	Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p );
	void addEdge_s( unsigned int thread , CoredVertexIndex< Index > v1 , CoredVertexIndex< Index > v2 );
	void addEdge_s( unsigned int thread , Index v1 , Index v2 );

	Index nextOutOfCorePoint( Vertex& p );
	Index nextEdge( CoredVertexIndex< Index > &v1 , CoredVertexIndex< Index > &v2 );

	size_t outOfCorePointCount( void );
	size_t edgeCount( void );
};

template< class Vertex , typename Index >
class CoredFileMeshData : public CoredMeshData< Vertex , Index >
{
	BufferedReadWriteFile *oocPointFile;
	Index oocPoints;
	std::vector< Index > polygons;
	std::vector< BufferedReadWriteFile* > polygonFiles;
	unsigned int threadIndex;
public:
	CoredFileMeshData( const char* fileHeader="" );
	~CoredFileMeshData( void );

	void resetIterator( void );

	Index addOutOfCorePoint( const Vertex& p );
	Index addOutOfCorePoint_s( unsigned int thread , const Vertex& p );
	void addPolygon_s( unsigned int thread , const std::vector< CoredVertexIndex< Index > >& vertices );
	void addPolygon_s( unsigned int thread , const std::vector< Index >& vertices );

	Index nextOutOfCorePoint( Vertex& p );
	Index nextPolygon( std::vector< CoredVertexIndex< Index > >& vertices );

	size_t outOfCorePointCount( void );
	size_t polygonCount( void );
};
#include "Geometry.inl"

#endif // GEOMETRY_INCLUDED
