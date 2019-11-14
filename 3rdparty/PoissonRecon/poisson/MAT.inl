/*
Copyright (c) 2007, Michael Kazhdan
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

//////////////////////////////
// MinimalAreaTriangulation //
//////////////////////////////
template< typename Index , class Real , unsigned int Dim >
_MinimalAreaTriangulation< Index , Real , Dim >::_MinimalAreaTriangulation( ConstPointer( Point< Real , Dim > ) vertices , size_t vCount ) : _vertices( vertices ) , _vCount( vCount )
{
	_bestTriangulation = NullPointer( Real );
	_midpoint = NullPointer( Index );
}
template< typename Index , class Real , unsigned int Dim >
_MinimalAreaTriangulation< Index , Real , Dim >::~_MinimalAreaTriangulation( void )
{
	FreePointer( _bestTriangulation );
	FreePointer( _midpoint );
}
template< typename Index , class Real , unsigned int Dim >
std::vector< TriangleIndex< Index > > _MinimalAreaTriangulation< Index , Real , Dim >::getTriangulation( void )
{
	std::vector< TriangleIndex< Index > > triangles;
	if( _vCount==3 )
	{
		triangles.resize(1);
		triangles[0].idx[0] = 0;
		triangles[0].idx[1] = 1;
		triangles[0].idx[2] = 2;
		return triangles;
	}
	else if( _vCount==4 )
	{
		TriangleIndex< Index > tIndex[2][2];
		Real area[] = { 0 , 0 };

		triangles.resize(2);

		tIndex[0][0].idx[0]=0;
		tIndex[0][0].idx[1]=1;
		tIndex[0][0].idx[2]=2;
		tIndex[0][1].idx[0]=2;
		tIndex[0][1].idx[1]=3;
		tIndex[0][1].idx[2]=0;

		tIndex[1][0].idx[0]=0;
		tIndex[1][0].idx[1]=1;
		tIndex[1][0].idx[2]=3;
		tIndex[1][1].idx[0]=3;
		tIndex[1][1].idx[1]=1;
		tIndex[1][1].idx[2]=2;

		Point< Real , Dim > p1 , p2;
		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) area[i] = SquareArea( _vertices[ tIndex[i][j].idx[0] ] , _vertices[ tIndex[i][j].idx[1] ] , _vertices[ tIndex[i][j].idx[2] ] );
		if( area[0]>area[1] ) triangles[0] = tIndex[1][0] , triangles[1] = tIndex[1][1];
		else                  triangles[0] = tIndex[0][0] , triangles[1] = tIndex[0][1];
		return triangles;
	}
	_set();
	_addTriangles( 1 , 0 , triangles );
	return triangles;
}
template< typename Index , class Real , unsigned int Dim >
void _MinimalAreaTriangulation< Index , Real , Dim >::_set( void )
{
	FreePointer( _bestTriangulation );
	FreePointer( _midpoint );
	_bestTriangulation = AllocPointer< Real >( _vCount * _vCount );
	_midpoint = AllocPointer< Index >( _vCount * _vCount );
	for( int i=0 ; i<_vCount*_vCount ; i++ ) _bestTriangulation[i] = -1 , _midpoint[i] = -1;
	_subPolygonArea( 1 , 0 );
}

template< typename Index , class Real , unsigned int Dim >
Index _MinimalAreaTriangulation< Index , Real , Dim >::_subPolygonIndex( Index i , Index j ) const { return (Index)( i*_vCount+j ); }

template< typename Index , class Real , unsigned int Dim >
void _MinimalAreaTriangulation< Index , Real , Dim >::_addTriangles( Index i , Index j , std::vector< TriangleIndex< Index > >& triangles ) const
{
	TriangleIndex< Index > tIndex;
	if( j<i ) j += (Index)_vCount;
	if( i==j || i+1==j ) return;
	Index mid = _midpoint[ _subPolygonIndex( i , j%_vCount ) ];
	if( mid!=-1 )
	{
		tIndex.idx[0] = i;
		tIndex.idx[1] = mid;
		tIndex.idx[2] = j%_vCount;
		triangles.push_back( tIndex );
		_addTriangles( i , mid , triangles );
		_addTriangles( mid , j , triangles );
	}
}

// Get the minimial area of the sub-polygon [ v_i , ... , v_j ]
template< typename Index , class Real , unsigned int Dim >
Real _MinimalAreaTriangulation< Index , Real , Dim >::_subPolygonArea( Index i , Index j )
{
	Index idx = _subPolygonIndex( i , j );
	if( _midpoint[idx]!=-1 ) return _bestTriangulation[idx];
	Real a = FLT_MAX , temp;
	if( j<i ) j += (Index)_vCount;
	// If either i==j or i+1=j, the polygon has trivial area
	if( i==j || i+1==j )
	{
		_bestTriangulation[idx] = 0;
		return 0;
	}
	// If we have already computed the minimal area for this edge
	if( _midpoint[idx]!=-1 ) return _bestTriangulation[idx];
	Index mid = -1;

	// For each vertex r \in( i , j ):
	// -- Construct the triangle ( j , r , i )
	// -- Compute the Area(j,r,i) + Area( j , ... , r ) + Area( r , ... , i )
	for( Index r=i+1 ; r<j ; r++ )
	{
		Index idx1 = _subPolygonIndex( i , r%_vCount ); // SubPolygon( r , ... , i )
		Index idx2 = _subPolygonIndex( r%_vCount , j%_vCount ); // SubPolygon( j , ... , r );

		temp = SquareArea( _vertices[i] , _vertices[r%_vCount] , _vertices[j%_vCount] );
		temp = temp<0 ? 0 : (Real)sqrt(temp);
		// If we have already computed Area( r , ... , i ), use that.
		if( _bestTriangulation[idx1]>=0 )
		{
			temp += _bestTriangulation[idx1];
			// If the partial area is already too large, terminate
			if( temp>a ) continue; // Terminate early
								   // Otherwise, compute the total area
			temp += _subPolygonArea( r%_vCount , j%_vCount );
		}
		else
		{
			// Otherwise, compute it now
			temp += _subPolygonArea( r%_vCount , j%_vCount );
			// If the partial area is already too large, terminate
			if( temp>a ) continue;
			// Otherwise, compute the total area
			temp += _subPolygonArea( i , r%_vCount );
		}

		if( temp<a ) a=temp , mid=r%_vCount;
	}
	_bestTriangulation[idx] = a;
	_midpoint[idx] = mid;
	return a;
}

