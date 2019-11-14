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

#include <stdio.h>
#include "MyMiscellany.h"

template< class Real > Real Random( void ){ return Real( rand() )/RAND_MAX; }

template< class Real , int Dim >
Point< Real , Dim > RandomBallPoint( void )
{
	Point< Real , Dim > p;
	while(1)
	{
		for( int d=0 ; d<Dim ; d++ ) p[d] = Real( 1.0-2.0*Random< Real >() );
		double l=SquareLength(p);
		if( SquareLength( p )<=1 ) return p;
	}
}
template< class Real , int Dim >
Point< Real , Dim > RandomSpherePoint( void )
{
	Point< Real , Dim > p = RandomBallPoint< Real , Dim >();
	return p / (Real)Length( p );
}

/////////////////////////
// CoredVectorMeshData //
/////////////////////////
template< class Vertex , typename Index >
CoredVectorMeshData< Vertex , Index >::CoredVectorMeshData( void ) { oocPointIndex = polygonIndex = threadIndex = 0 ; polygons.resize( std::thread::hardware_concurrency() ); }
template< class Vertex , typename Index >
void CoredVectorMeshData< Vertex , Index >::resetIterator ( void ) { oocPointIndex = polygonIndex = threadIndex = 0; }
template< class Vertex , typename Index >
Index CoredVectorMeshData< Vertex , Index >::addOutOfCorePoint( const Vertex& p )
{
	oocPoints.push_back(p);
	return ( Index )oocPoints.size()-1;
}
template< class Vertex , typename Index >
Index CoredVectorMeshData< Vertex , Index >::addOutOfCorePoint_s( unsigned int thread , const Vertex& p )
{
	size_t sz;
	{
		static std::mutex m;
		std::lock_guard< std::mutex > lock(m);
		sz = oocPoints.size();
		oocPoints.push_back(p);
	}
	return (Index)sz;
}
template< class Vertex , typename Index >
void CoredVectorMeshData< Vertex , Index >::addPolygon_s( unsigned int thread , const std::vector< Index >& polygon )
{
	polygons[ thread ].push_back( polygon );
}
template< class Vertex , typename Index >
void CoredVectorMeshData< Vertex , Index >::addPolygon_s( unsigned int thread , const std::vector< CoredVertexIndex< Index > >& vertices )
{
	std::vector< Index > polygon( vertices.size() );
	for( int i=0 ; i<(int)vertices.size() ; i++ ) 
		if( vertices[i].inCore ) polygon[i] =  vertices[i].idx;
		else                     polygon[i] = -vertices[i].idx-1;
	return addPolygon_s( thread , polygon );
}
template< class Vertex , typename Index >
Index CoredVectorMeshData< Vertex , Index >::nextOutOfCorePoint( Vertex& p )
{
	if( oocPointIndex<(Index)oocPoints.size() )
	{
		p=oocPoints[oocPointIndex++];
		return 1;
	}
	else return 0;
}
template< class Vertex , typename Index >
Index CoredVectorMeshData< Vertex , Index >::nextPolygon( std::vector< CoredVertexIndex< Index > >& vertices )
{
	while( true )
	{
		if( threadIndex<(int)polygons.size() )
		{
			if( polygonIndex<(Index)( polygons[threadIndex].size() ) )
			{
				std::vector< Index >& polygon = polygons[threadIndex][ polygonIndex++ ];
				vertices.resize( polygon.size() );
				for( int i=0 ; i<int(polygon.size()) ; i++ )
					if( polygon[i]<0 ) vertices[i].idx = -polygon[i]-1 , vertices[i].inCore = false;
					else               vertices[i].idx =  polygon[i]   , vertices[i].inCore = true;
				return 1;
			}
			else threadIndex++ , polygonIndex = 0;
		}
		else return 0;
	}
}
template< class Vertex , typename Index >
size_t CoredVectorMeshData< Vertex , Index >::outOfCorePointCount( void ){ return oocPoints.size(); }
template< class Vertex , typename Index >
size_t CoredVectorMeshData< Vertex , Index >::polygonCount( void )
{
	size_t count = 0;
	for( size_t i=0 ; i<polygons.size() ; i++ ) count += polygons[i].size();
	return count;
}

///////////////////////
// CoredFileMeshData //
///////////////////////
template< class Vertex , typename Index >
CoredFileMeshData< Vertex , Index >::CoredFileMeshData( const char* fileHeader )
{
	threadIndex = 0;
	oocPoints = 0;
	polygons.resize( std::thread::hardware_concurrency() );
	for( unsigned int i=0 ; i<polygons.size() ; i++ ) polygons[i] = 0;

	char _fileHeader[1024];
	sprintf( _fileHeader , "%s_points_" , fileHeader );
	oocPointFile = new BufferedReadWriteFile( NULL , _fileHeader );
	polygonFiles.resize( std::thread::hardware_concurrency() );
	for( unsigned int i=0 ; i<polygonFiles.size() ; i++ )
	{
		sprintf( _fileHeader , "%s_polygons_t%d_" , fileHeader , i );
		polygonFiles[i] = new BufferedReadWriteFile( NULL , _fileHeader );
	}
}
template< class Vertex , typename Index >
CoredFileMeshData< Vertex , Index >::~CoredFileMeshData( void )
{
	delete oocPointFile;
	for( unsigned int i=0 ; i<polygonFiles.size() ; i++ ) delete polygonFiles[i];
}
template< class Vertex , typename Index >
void CoredFileMeshData< Vertex , Index >::resetIterator ( void )
{
	oocPointFile->reset();
	threadIndex = 0;
	for( unsigned int i=0 ; i<polygonFiles.size() ; i++ ) polygonFiles[i]->reset();
}
template< class Vertex , typename Index >
Index CoredFileMeshData< Vertex , Index >::addOutOfCorePoint( const Vertex& p )
{
	oocPointFile->write( &p , sizeof( Vertex ) );
	oocPoints++;
	return oocPoints-1;
}
template< class Vertex , typename Index >
Index CoredFileMeshData< Vertex , Index >::addOutOfCorePoint_s( unsigned int thread , const Vertex& p )
{
	Index sz;
	{
		static std::mutex m;
		std::lock_guard< std::mutex > lock(m);
		sz = oocPoints;
		oocPointFile->write( &p , sizeof( Vertex ) );
		oocPoints++;
	}
	return sz;
}
template< class Vertex , typename Index >
void CoredFileMeshData< Vertex , Index >::addPolygon_s( unsigned int thread , const std::vector< Index >& vertices )
{
	unsigned int vSize = (unsigned int)vertices.size();
	polygonFiles[thread]->write( &vSize , sizeof(unsigned int) );
	polygonFiles[thread]->write( &vertices[0] , sizeof(Index) * vSize );
	polygons[thread]++;
}
template< class Vertex , typename Index >
void CoredFileMeshData< Vertex , Index >::addPolygon_s( unsigned int thread , const std::vector< CoredVertexIndex< Index > >& vertices )
{
	std::vector< Index > polygon( vertices.size() );
	for( unsigned int i=0 ; i<(unsigned int)vertices.size() ; i++ ) 
		if( vertices[i].inCore ) polygon[i] =  vertices[i].idx;
		else                     polygon[i] = -vertices[i].idx-1;
	return addPolygon_s( thread , polygon );
}
template< class Vertex , typename Index >
Index CoredFileMeshData< Vertex , Index >::nextOutOfCorePoint( Vertex& p )
{
	if( oocPointFile->read( &p , sizeof( Vertex ) ) ) return 1;
	else return 0;
}
template< class Vertex , typename Index >
Index CoredFileMeshData< Vertex , Index >::nextPolygon( std::vector< CoredVertexIndex< Index > >& vertices )
{
	while( true )
	{
		if( threadIndex<(unsigned int)polygonFiles.size() )
		{
			unsigned int pSize;
			if( polygonFiles[threadIndex]->read( &pSize , sizeof(unsigned int) ) )
			{
				std::vector< Index > polygon( pSize );
				if( polygonFiles[threadIndex]->read( &polygon[0] , sizeof(Index)*pSize ) )
				{
					vertices.resize( pSize );
					for( unsigned int i=0 ; i<(unsigned int)polygon.size() ; i++ )
						if( polygon[i]<0 ) vertices[i].idx = -polygon[i]-1 , vertices[i].inCore = false;
						else               vertices[i].idx =  polygon[i]   , vertices[i].inCore = true;
					return 1;
				}
				ERROR_OUT( "Failed to read polygon from file" );
			}
			else threadIndex++;
		}
		else return 0;
	}
}
template< class Vertex , typename Index >
size_t CoredFileMeshData< Vertex , Index >::outOfCorePointCount( void ){ return oocPoints; }
template< class Vertex , typename Index >
size_t CoredFileMeshData< Vertex , Index >::polygonCount( void )
{
	size_t count = 0;
	for( size_t i=0 ; i<polygons.size() ; i++ ) count += polygons[i];
	return count;
}

/////////////
// Simplex //
/////////////
template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
{
	Real values[K+1];
	bool frontSet = false , backSet = false;

	// Evaluate the hyper-plane's function at the vertices and mark if strictly front/back vertices have been found
	for( unsigned int k=0 ; k<=K ; k++ )
	{
		values[k] = Point< Real , Dim >::Dot( p[k] , pNormal ) - pOffset;
		backSet |= ( values[k]<0 ) , frontSet |= ( values[k]>0 );
	}

	// If all the vertices are behind or on, or all the vertices are in front or on, we are done.
	if( !frontSet ){ back.push_back( *this ) ; return; }
	if( !backSet ){ front.push_back( *this ) ; return; }

	// Pick some intersection of the hyper-plane with a simplex edge
	unsigned int v1 , v2;
	Point< Real , Dim > midPoint;
	{
		for( unsigned int i=0 ; i<K ; i++ ) for( unsigned int j=i+1 ; j<=K ; j++ ) if( values[i]*values[j]<0 )
		{
			v1 = i , v2 = j;
			Real t1 = values[i] / ( values[i] - values[j] ) , t2 = (Real)( 1. - t1 );
			midPoint = p[j]*t1 + p[i]*t2;
		}
	}
	// Iterate over each face of the simplex, split it with the hyper-plane and connect the sub-simplices to the mid-point
	for( unsigned int i=0 ; i<=K ; i++ )
	{
		if( i!=v1 && i!=v2 ) continue;
		Simplex< Real , Dim , K-1 > f;		// The face
		Simplex< Real , Dim , K > s;		// The sub-simplex
		for( unsigned int j=0 , idx=0 ; j<=K ; j++ )	if( j!=i ) f[idx++] = p[j];
		std::vector< Simplex< Real , Dim , K-1 > > _back , _front;
		f.split( pNormal , pOffset , _back , _front );
		s[i] = midPoint;

		for( unsigned int j=0 ; j<_back.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _back[j][k];
			back.push_back( s );
		}

		for( unsigned int j=0 ; j<_front.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _front[j][k];
			front.push_back( s );
		}
	}
}