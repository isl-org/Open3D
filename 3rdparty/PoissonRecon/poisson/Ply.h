/*

Header for PLY polygon files.

- Greg Turk, March 1994

A PLY file contains a single polygonal _object_.

An object is composed of lists of _elements_.  Typical elements are
vertices, faces, edges and materials.

Each type of element for a given object has one or more _properties_
associated with the element type.  For instance, a vertex element may
have as properties three floating-point values x,y,z and three unsigned
chars for red, green and blue.

---------------------------------------------------------------

Copyright (c) 1994 The Board of Trustees of The Leland Stanford
Junior University.  All rights reserved.   

Permission to use, copy, modify and distribute this software and its   
documentation for any purpose is hereby granted without fee, provided   
that the above copyright notice and this permission notice appear in   
all copies of this software and that you do not sell the software.   

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,   
EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY   
WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.   

*/

#ifndef PLY_INCLUDED
#define PLY_INCLUDED

#include <limits>
#include <vector>
#include "PlyFile.h"
#include "Geometry.h"

template< class Real > int PLYType( void );
template<> inline int PLYType< int           >( void ){ return PLY_INT   ; }
template<> inline int PLYType<          char >( void ){ return PLY_CHAR  ; }
template<> inline int PLYType< unsigned char >( void ){ return PLY_UCHAR ; }
template<> inline int PLYType<        float  >( void ){ return PLY_FLOAT ; }
template<> inline int PLYType<        double >( void ){ return PLY_DOUBLE; }
template< class Real > inline int PLYType( void )
{
	ERROR_OUT( "Unrecognized type" );
	return -1;
}

template< typename Integer > struct PLYTraits{ static const std::string name; };
template<> const std::string PLYTraits< int >::name="int";
template<> const std::string PLYTraits< unsigned int >::name="unsigned int";
template<> const std::string PLYTraits< long >::name="long";
template<> const std::string PLYTraits< unsigned long >::name="unsigned long";
template<> const std::string PLYTraits< long long >::name="long long";
template<> const std::string PLYTraits< unsigned long long >::name="unsigned long long";

template< typename Index >
struct PlyFace
{
	unsigned int nr_vertices;
	Index *vertices;
	int segment;

	static PlyProperty face_props[];
};
template<>
PlyProperty PlyFace<          int       >::face_props[] = { PlyProperty( "vertex_indices" , PLY_INT       , PLY_INT       , offsetof( PlyFace , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( PlyFace , nr_vertices ) ) };
template<>
PlyProperty PlyFace< unsigned int       >::face_props[] = { PlyProperty( "vertex_indices" , PLY_UINT      , PLY_UINT      , offsetof( PlyFace , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( PlyFace , nr_vertices ) ) };
template<>
PlyProperty PlyFace<          long long >::face_props[] = { PlyProperty( "vertex_indices" , PLY_LONGLONG  , PLY_LONGLONG  , offsetof( PlyFace , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( PlyFace , nr_vertices ) ) };
template<>
PlyProperty PlyFace< unsigned long long >::face_props[] = { PlyProperty( "vertex_indices" , PLY_ULONGLONG , PLY_ULONGLONG , offsetof( PlyFace , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( PlyFace , nr_vertices ) ) };

struct RGBColor
{
	unsigned char c[3];
	unsigned char& operator[]( int idx )       { return c[idx]; }
	unsigned char  operator[]( int idx ) const { return c[idx]; }
	RGBColor( void ){ c[0] = c[1] = c[2] = 0; }
	RGBColor( const RGBColor& rgb ){ memcpy( c , rgb.c , sizeof(unsigned char) * 3 ); }
	RGBColor& operator = ( const RGBColor& rgb ){ memcpy( c , rgb.c , sizeof(unsigned char) * 3 ) ; return *this; }
};

///////////////
// PlyVertex //
///////////////
template< typename _Real , int Dim , typename _RealOnDisk=float >
class PlyVertex
{
public:
	typedef _Real Real;

	PlyVertex& operator += ( const PlyVertex& p ){ point += p.point ; return *this; }
	PlyVertex& operator -= ( const PlyVertex& p ){ point -= p.point ; return *this; }
	PlyVertex& operator *= ( Real s )            { point *= s ; return *this; }
	PlyVertex& operator /= ( Real s )            { point /= s ; return *this; }
	PlyVertex  operator +  ( const PlyVertex& p ) const { return PlyVertex( point + p.point ); }
	PlyVertex  operator -  ( const PlyVertex& p ) const { return PlyVertex( point - p.point ); }
	PlyVertex  operator *  ( Real s )             const { return PlyVertex( point * s ); }
	PlyVertex  operator /  ( Real s )             const { return PlyVertex( point / s ); }

	const static int PlyReadNum = Dim;
	const static int PlyWriteNum = Dim;

	static const PlyProperty* PlyReadProperties( void ){ return _PlyProperties; }
	static const PlyProperty* PlyWriteProperties( void ){ return _PlyProperties; }

	Point< Real , Dim > point;
	PlyVertex( void ) {}
	PlyVertex( Point< Real , Dim > p ) : point(p) { }

	struct Transform
	{
		Transform( void ){}
		Transform( const XForm< Real , Dim+1 >& xForm ) : _pointXForm(xForm) { }
		PlyVertex operator() ( const PlyVertex& p ) const
		{
			PlyVertex _p;
			_p.point = _pointXForm * p.point;
			return _p;
		}
	protected:
		XForm< Real , Dim+1 > _pointXForm;
	};

protected:
	static const PlyProperty _PlyProperties[];
};

template<>
const PlyProperty PlyVertex< float , 2 , float >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_FLOAT , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_FLOAT , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< double , 2 , float >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_FLOAT , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_FLOAT , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< float , 2 , double >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_DOUBLE , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_DOUBLE , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< double , 2 , double >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_DOUBLE , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_DOUBLE , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
};

template<>
const PlyProperty PlyVertex< float , 3 , float >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_FLOAT , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_FLOAT , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "z" , PLY_FLOAT , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< double , 3 , float >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_FLOAT , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_FLOAT , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "z" , PLY_FLOAT , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< float , 3 , double >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_DOUBLE , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_DOUBLE , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "z" , PLY_DOUBLE , PLY_FLOAT , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 ) ,
};
template<>
const PlyProperty PlyVertex< double , 3 , double >::_PlyProperties[] =
{
	PlyProperty( "x" , PLY_DOUBLE , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "y" , PLY_DOUBLE , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 ) ,
	PlyProperty( "z" , PLY_DOUBLE , PLY_DOUBLE , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 ) ,
};

///////////////////////
// PlyVertexWithData //
///////////////////////
template< typename _Real , int Dim , typename Data , typename _RealOnDisk=float >
class PlyVertexWithData
{
public:
	typedef _Real Real;

	PlyVertexWithData& operator += ( const PlyVertexWithData& p ){ point += p.point , data += p.data ; return *this; }
	PlyVertexWithData& operator -= ( const PlyVertexWithData& p ){ point -= p.point , data -= p.data ; return *this; }
	PlyVertexWithData& operator *= ( Real s )                    { point *= s , data *= s ; return *this; }
	PlyVertexWithData& operator /= ( Real s )                    { point /= s , data /= s ; return *this; }
	PlyVertexWithData  operator +  ( const PlyVertexWithData& p ) const { return PlyVertexWithData( point + p.point , data + p.data ); }
	PlyVertexWithData  operator -  ( const PlyVertexWithData& p ) const { return PlyVertexWithData( point - p.point , data - p.data ); }
	PlyVertexWithData  operator *  ( Real s )                     const { return PlyVertexWithData( point * s , data * s ); }
	PlyVertexWithData  operator /  ( Real s )                     const { return PlyVertexWithData( point / s , data / s ); }

	const static int PlyReadNum = Data::PlyReadNum + Dim;
	const static int PlyWriteNum = Data::PlyWriteNum + Dim;

	static const PlyProperty* PlyReadProperties( void ){ _SetReadProperties() ; return _PlyReadProperties; }
	static const PlyProperty* PlyWriteProperties( void ){ _SetWriteProperties() ; return _PlyWriteProperties; }


	Point< Real , Dim > point;
	Data data;
	PlyVertexWithData( void ) {}
	PlyVertexWithData( Point< Real , Dim > p , Data d ) : point(p) , data(d) { }

	struct Transform
	{
		Transform( void ){}
		Transform( const XForm< Real , Dim+1 >& xForm ) : _pointXForm(xForm) , _dataXForm(xForm) { }
		PlyVertexWithData operator() ( const PlyVertexWithData& p ) const
		{
			PlyVertexWithData _p;
			_p.point = _pointXForm * p.point;
			_p.data = _dataXForm( p.data );
			return _p;
		}
	protected:
		XForm< Real , Dim+1 > _pointXForm;
		typename Data::Transform _dataXForm;
	};

protected:
	static void _SetReadProperties( void );
	static void _SetWriteProperties( void );
	static PlyProperty _PlyReadProperties[];
	static PlyProperty _PlyWriteProperties[];
};
template< typename Real , int Dim , typename Data , typename RealOnDisk > PlyProperty PlyVertexWithData< Real , Dim , Data , RealOnDisk >::_PlyReadProperties[ PlyReadNum ];
template< typename Real , int Dim , typename Data , typename RealOnDisk > PlyProperty PlyVertexWithData< Real , Dim , Data , RealOnDisk >::_PlyWriteProperties[ PlyWriteNum ];
template< typename Real , int Dim , typename Data , typename RealOnDisk >
void PlyVertexWithData< Real , Dim , Data , RealOnDisk >::_SetReadProperties( void )
{
	{
		const PlyProperty * ReadProps = PlyVertex< Real , Dim , RealOnDisk >::PlyReadProperties();
		for( int d=0 ; d<PlyVertex< Real , Dim , RealOnDisk >::PlyReadNum ; d++ ) _PlyReadProperties[d] = ReadProps[d];
	}
	{
		const PlyProperty * ReadProps = Data::PlyReadProperties();
		for( int d=0 ; d<Data::PlyReadNum ; d++ )
		{
			_PlyReadProperties[d+PlyVertex< Real , Dim , RealOnDisk >::PlyReadNum ] = ReadProps[d];
			_PlyReadProperties[d+PlyVertex< Real , Dim , RealOnDisk >::PlyReadNum ].offset += (int)offsetof( PlyVertexWithData , data );
		}
	}
}
template< typename Real , int Dim , typename Data , typename RealOnDisk >
void PlyVertexWithData< Real , Dim , Data , RealOnDisk >::_SetWriteProperties( void )
{
	{
		const PlyProperty * WriteProps = PlyVertex< Real , Dim , RealOnDisk >::PlyWriteProperties();
		for( int d=0 ; d<PlyVertex< Real , Dim , RealOnDisk >::PlyWriteNum ; d++ ) _PlyWriteProperties[d] = WriteProps[d];
	}
	{
		const PlyProperty * WriteProps = Data::PlyWriteProperties();
		for( int d=0 ; d<Data::PlyWriteNum ; d++ )
		{
			_PlyWriteProperties[d+PlyVertex< Real , Dim , RealOnDisk >::PlyWriteNum ] = WriteProps[d];
			_PlyWriteProperties[d+PlyVertex< Real , Dim , RealOnDisk >::PlyWriteNum ].offset += (int)offsetof( PlyVertexWithData , data );
		}
	}
}

template< class Vertex , typename Index , class Real , int Dim , typename OutputIndex=int >
int PlyWritePolygons( const char* fileName , CoredMeshData< Vertex , Index >* mesh , int file_type , const Point< float , Dim >& translate , float scale , const std::vector< std::string >& comments , XForm< Real , Dim+1 > xForm=XForm< Real , Dim+1 >::Identity() );

template< class Vertex , typename Index , class Real , int Dim , typename OutputIndex=int  >
int PlyWritePolygons( const char* fileName , CoredMeshData< Vertex , Index >* mesh , int file_type ,                                                       const std::vector< std::string >& comments , XForm< Real , Dim+1 > xForm=XForm< Real , Dim+1 >::Identity() );

inline bool PlyReadHeader( char* fileName , const PlyProperty* properties , int propertyNum , bool* readFlags , int& file_type )
{
	std::vector< std::string > elist;
	float version;

	PlyFile *ply = PlyFile::Read( fileName , elist , file_type , version );
	if( !ply ) return false;

	for( int i=0 ; i<elist.size() ; i++ ) if( elist[i]=="vertex" ) for( int j=0 ; j<propertyNum ; j++ ) if( readFlags ) readFlags[j] = ply->get_property( elist[i].c_str() , &properties[j] )!=0;

	delete ply;
	return true;
}
inline bool PlyReadHeader( char* fileName , const PlyProperty* properties , int propertyNum , bool* readFlags )
{
	int file_type;
	return PlyReadHeader( fileName , properties , propertyNum , readFlags , file_type );
}


template< class Vertex , typename Index >
int PlyReadPolygons( const char* fileName,
	std::vector< Vertex >& vertices , std::vector<std::vector< Index > >& polygons ,
	const PlyProperty* properties , int propertyNum ,
	int& file_type ,
	std::vector< std::string > &comments , bool* readFlags=NULL );

template< class Vertex , typename Index >
int PlyWritePolygons( const char* fileName ,
	const std::vector< Vertex > &vertices , const std::vector< std::vector< Index > > &polygons ,
	const PlyProperty* properties , int propertyNum ,
	int file_type ,
	const std::vector< std::string > &comments );

template< class Vertex , typename Index >
int PlyWritePolygons( const char* fileName ,
	const std::vector< Vertex > &vertices , const std::vector< std::vector< Index > > &polygons ,
	const PlyProperty *properties , int propertyNum ,
	int file_type ,
	const std::vector< std::string > &comments )
{
	size_t nr_vertices = vertices.size();
	size_t nr_faces = polygons.size();
	float version;
	std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "face" ) };
	PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
	if (!ply){return 0;}

	//
	// describe vertex and face properties
	//
	ply->element_count( "vertex", nr_vertices );
	for( int i=0 ; i<propertyNum ; i++ ) ply->describe_property( "vertex" , &properties[i] );
	ply->element_count( "face" , nr_faces );
	ply->describe_property( "face" , PlyFace< Index >::face_props );

	// Write in the comments
	for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
	ply->header_complete();

	// write vertices
	ply->put_element_setup( elem_names[0] );
	for( size_t i=0 ; i<vertices.size() ; i++ ) ply->put_element( (void *)&vertices[i] );

	// write faces
	PlyFace< Index > ply_face;
	int maxFaceVerts=3;
	ply_face.nr_vertices = 3;
	ply_face.vertices = new Index[3];

	ply->put_element_setup( elem_names[1] );
	for( size_t i=0 ; i<nr_faces ; i++ )
	{
		if( (int)polygons[i].size()>maxFaceVerts )
		{
			delete[] ply_face.vertices;
			maxFaceVerts = (int)polygons[i].size();
			ply_face.vertices=new Index[ maxFaceVerts ];
		}
		ply_face.nr_vertices = (int)polygons[i].size();
		for( size_t j=0 ; j<ply_face.nr_vertices ; j++ ) ply_face.vertices[j] = polygons[i][j];
		ply->put_element( (void *)&ply_face );
	}

	delete[] ply_face.vertices;
	delete ply;

	return 1;
}

template< class Vertex , typename Index >
int PlyReadPolygons
(
	const char *fileName ,
	std::vector< Vertex > &vertices ,
	std::vector< std::vector< Index > > &polygons ,
	const PlyProperty *properties ,
	int propertyNum ,
	int &file_type ,
	std::vector< std::string > &comments ,
	bool *readFlags
)
{
	std::vector< std::string > elist = { std::string( "vertex" ) , std::string( "face" ) };
	float version;

	PlyFile *ply = PlyFile::Read( fileName , elist , file_type , version );
	if(!ply) return 0;

	comments.reserve( comments.size() + ply->comments.size() );
	for( int i=0 ; i<ply->comments.size() ; i++ ) comments.push_back( ply->comments[i] );

	for( int i=0 ; i<elist.size() ; i++ )
	{
		std::string &elem_name = elist[i];
		size_t num_elems;
		std::vector< PlyProperty * > plist = ply->get_element_description( elem_name , num_elems );
		if( !plist.size() )
		{
			delete ply;
			return 0;
		}		
		if( elem_name=="vertex" )
		{
			for( int i=0 ; i<propertyNum ; i++)
			{
				int hasProperty = ply->get_property( elem_name , &properties[i] );
				if( readFlags ) readFlags[i] = (hasProperty!=0);
			}
			vertices.resize( num_elems );
			for( size_t j=0 ; j<num_elems ; j++ ) ply->get_element( (void *)&vertices[j] );
		}
		else if( elem_name=="face" )
		{
			ply->get_property( elem_name , PlyFace< Index >::face_props );
			polygons.resize( num_elems );
			for( size_t j=0 ; j<num_elems ; j++ )
			{
				PlyFace< Index > ply_face;
				ply->get_element( (void *)&ply_face );
				polygons[j].resize( ply_face.nr_vertices );
				for( unsigned int k=0 ; k<ply_face.nr_vertices ; k++ ) polygons[j][k] = ply_face.vertices[k];
				free( ply_face.vertices );
			}  // for, read faces
		}  // if face
		else ply->get_other_element( elem_name , num_elems );

		for( int j=0 ; j<plist.size() ; j++ ) delete plist[j];
	}  // for each type of element

	delete ply;
	return 1;
}

template< class Vertex , typename Index , class Real , int Dim , typename OutputIndex >
int PlyWritePolygons( const char* fileName , CoredMeshData< Vertex , Index >* mesh , int file_type , const Point< float , Dim >& translate , float scale , const std::vector< std::string > &comments , XForm< Real , Dim+1 > xForm )
{
	if( mesh->outOfCorePointCount()+mesh->inCorePoints.size()>(size_t)std::numeric_limits<OutputIndex>::max() )
	{
		if( std::is_same< Index , OutputIndex >::value ) ERROR_OUT( "more vertices than can be represented using " , PLYTraits< Index >::name );
		WARN( "more vertices than can be represented using " , PLYTraits< OutputIndex >::name , " using " , PLYTraits< Index >::name , " instead" );
		return PlyWritePolygons< Vertex , Index , Real , Dim , Index >( fileName , mesh , file_type , translate , scale , comments , xForm );
	}
	size_t nr_vertices = mesh->outOfCorePointCount()+mesh->inCorePoints.size();
	size_t nr_faces = mesh->polygonCount();
	float version;
	std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "face" ) };
	PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
	if( !ply ) return 0;

	mesh->resetIterator();

	//
	// describe vertex and face properties
	//
	ply->element_count( "vertex" , nr_vertices );
	for( int i=0 ; i<Vertex::Components ; i++ ) ply->describe_property( "vertex" , &Vertex::Properties[i] );
	ply->element_count( "face" , nr_faces );
	ply->describe_property( "face" , PlyFace< OutputIndex >::face_props );

	// Write in the comments
	for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
	ply->header_complete();

	// write vertices
	ply->put_element_setup( "vertex" );
	for( size_t i=0 ; i<mesh->inCorePoints.size() ; i++ )
	{
		Vertex vertex = xForm * ( mesh->inCorePoints[i] * scale + translate );
		ply->put_element( (void *)&vertex );
	}
	for( size_t i=0; i<mesh->outOfCorePointCount() ; i++ )
	{
		Vertex vertex;
		mesh->nextOutOfCorePoint( vertex );
		vertex = xForm * ( vertex * scale + translate );
		ply->put_element( (void *)&vertex );
	}  // for, write vertices

	   // write faces
	std::vector< CoredVertexIndex< Index > > polygon;
	ply->put_element_setup( "face" );
	for( size_t i=0 ; i<nr_faces ; i++ )
	{
		//
		// create and fill a struct that the ply code can handle
		//
		PlyFace< OutputIndex > ply_face;
		mesh->nextPolygon( polygon );
		ply_face.nr_vertices = int( polygon.size() );
		ply_face.vertices = new OutputIndex[ polygon.size() ];
		for( int j=0 ; j<int(polygon.size()) ; j++ )
			if( polygon[j].inCore ) ply_face.vertices[j] = (OutputIndex)polygon[j].idx;
			else                    ply_face.vertices[j] = (OutputIndex)( polygon[j].idx + mesh->inCorePoints.size() );
			ply->put_element( (void *)&ply_face );
			delete[] ply_face.vertices;
	}  // for, write faces

	delete ply;

	return 1;
}

template< class Vertex , typename Index , class Real , int Dim , typename OutputIndex >
int PlyWritePolygons( const char* fileName , CoredMeshData< Vertex , Index >* mesh , int file_type , const std::vector< std::string > &comments , XForm< Real , Dim+1 > xForm )
{
	if( mesh->outOfCorePointCount()+mesh->inCorePoints.size()>(size_t)std::numeric_limits<OutputIndex>::max() )
	{
		if( std::is_same< Index , OutputIndex >::value ) ERROR_OUT( "more vertices than can be represented using " , PLYTraits< Index >::name );
		WARN( "more vertices than can be represented using " , PLYTraits< OutputIndex >::name , " using " , PLYTraits< Index >::name , " instead" );
		return PlyWritePolygons< Vertex , Index , Real , Dim , Index >( fileName , mesh , file_type , comments , xForm );
	}
	size_t nr_vertices = mesh->outOfCorePointCount()+mesh->inCorePoints.size();
	size_t nr_faces = mesh->polygonCount();
	float version;
	std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "face" ) };
	PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
	if( !ply ) return 0;

	mesh->resetIterator();

	//
	// describe vertex and face properties
	//
	ply->element_count( "vertex" , nr_vertices );
	typename Vertex::Transform _xForm( xForm );
	const PlyProperty* PlyWriteProperties = Vertex::PlyWriteProperties();
	for( int i=0 ; i<Vertex::PlyWriteNum ; i++ ) ply->describe_property( "vertex" , &PlyWriteProperties[i] );
	ply->element_count( "face" , nr_faces );
	ply->describe_property( "face" , PlyFace< OutputIndex >::face_props );

	// Write in the comments
	for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
	ply->header_complete();
	// write vertices
	ply->put_element_setup( "vertex" );
	for( size_t i=0 ; i<mesh->inCorePoints.size() ; i++ )
	{
		Vertex vertex = _xForm( mesh->inCorePoints[i] );
		ply->put_element( (void *)&vertex );
	}
	for( size_t i=0; i<mesh->outOfCorePointCount() ; i++ )
	{
		Vertex vertex;
		mesh->nextOutOfCorePoint( vertex );
		vertex = _xForm( vertex );
		ply->put_element( (void *)&vertex );
	}  // for, write vertices

	   // write faces
	std::vector< CoredVertexIndex< Index > > polygon;
	ply->put_element_setup( "face" );
	for( size_t i=0 ; i<nr_faces ; i++ )
	{
		//
		// create and fill a struct that the ply code can handle
		//
		PlyFace< OutputIndex > ply_face;
		mesh->nextPolygon( polygon );
		ply_face.nr_vertices = int( polygon.size() );
		ply_face.vertices = new OutputIndex[ polygon.size() ];
		for( int j=0 ; j<int(polygon.size()) ; j++ )
			if( polygon[j].inCore ) ply_face.vertices[j] = (OutputIndex)polygon[j].idx;
			else                    ply_face.vertices[j] = (OutputIndex)( polygon[j].idx + mesh->inCorePoints.size() );
			ply->put_element( (void *)&ply_face );
			delete[] ply_face.vertices;
	}  // for, write faces

	delete ply;

	return 1;
}
inline int PlyDefaultFileType( void ){ return PLY_ASCII; }

#endif // PLY_INCLUDED
