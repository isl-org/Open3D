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
prior writften permission. 

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

#ifndef POINT_STREAM_INCLUDED
#define POINT_STREAM_INCLUDED

#include <functional>
#include "Ply.h"
#include "Geometry.h"


template< class Real , unsigned int Dim >
class InputPointStream
{
public:
	virtual ~InputPointStream( void ){}
	virtual void reset( void ) = 0;
	virtual bool nextPoint( Point< Real , Dim >& p ) = 0;
	virtual size_t nextPoints( Point< Real , Dim >* p , size_t count )
	{
		size_t c=0;
		for( size_t i=0 ; i<count ; i++ , c++ ) if( !nextPoint( p[i] ) ) break;
		return c;
	}
	void boundingBox( Point< Real , Dim >& min , Point< Real , Dim >& max )
	{
		bool first = true;
		Point< Real , Dim > p;
		while( nextPoint( p ) )
		{
			for( unsigned int i=0 ; i<Dim ; i++ )
			{
				if( first || p[i]<min[i] ) min[i] = p[i];
				if( first || p[i]>max[i] ) max[i] = p[i];
			}
			first = false;
		}
		reset();
	}
};

template< class Real , unsigned int Dim >
class OutputPointStream
{
public:
	virtual ~OutputPointStream( void ){}
	virtual void nextPoint( const Point< Real , Dim >& p ) = 0;
	virtual void nextPoints( const Point< Real , Dim >* p , size_t count ){ for( size_t i=0 ; i<count ; i++ ) nextPoint( p[i] ); }
};

template< class Real , unsigned int Dim , class Data >
class InputPointStreamWithData : public InputPointStream< Real , Dim >
{
public:
	virtual ~InputPointStreamWithData( void ){}
	virtual void reset( void ) = 0;
	virtual bool nextPoint( Point< Real , Dim >& p , Data& d ) = 0;

	virtual bool nextPoint( Point< Real , Dim >& p ){ Data d ; return nextPoint( p , d ); }
	virtual size_t nextPoints( Point< Real , Dim >* p , Data* d , size_t count )
	{
		size_t c=0;
		for( size_t i=0 ; i<count ; i++ , c++ ) if( !nextPoint( p[i] , d[i] ) ) break;
		return c;
	}
	virtual size_t nextPoints( Point< Real , Dim >* p , size_t count ){ return InputPointStream< Real , Dim >::nextPoints( p , count ); }
};

template< class Real , unsigned int Dim , class Data >
class OutputPointStreamWithData : public OutputPointStream< Real , Dim >
{
public:
	virtual ~OutputPointStreamWithData( void ){}
	virtual void nextPoint( const Point< Real , Dim >& p , const Data& d ) = 0;

	virtual void nextPoint( const Point< Real , Dim >& p ){ Data d ; return nextPoint( p , d ); }
	virtual void nextPoints( const Point< Real , Dim >* p , const Data* d , size_t count ){ for( size_t i=0 ; i<count ; i++ ) nextPoint( p[i] , d[i] ); }
	virtual void nextPoints( const Point< Real , Dim >* p , size_t count ){ OutputPointStream< Real , Dim >::nextPoints( p , count ); }
};

template< class Real , unsigned int Dim >
class TransformedInputPointStream : public InputPointStream< Real , Dim >
{
	std::function< void ( Point< Real , Dim >& ) > _xForm;
	InputPointStream< Real , Dim >& _stream;
public:
	TransformedInputPointStream( std::function< void ( Point< Real , Dim >& ) > xForm , InputPointStream< Real , Dim >& stream ) : _xForm(xForm) , _stream(stream) {;}
	virtual void reset( void ){ _stream.reset(); }
	virtual bool nextPoint( Point< Real , Dim >& p )
	{
		bool ret = _stream.nextPoint( p );
		_xForm( p );
		return ret;
	}
};

template< class Real , unsigned int Dim >
class TransformedOutputPointStream : public OutputPointStream< Real , Dim >
{
	std::function< void ( Point< Real , Dim >& ) > _xForm;
	OutputPointStream< Real , Dim >& _stream;
public:
	TransformedOutputPointStream( std::function< void ( Point< Real , Dim >& ) > xForm , OutputPointStream< Real , Dim >& stream ) : _xForm(xForm) , _stream(stream) {;}
	virtual void reset( void ){ _stream.reset(); }
	virtual bool nextPoint( const Point< Real , Dim >& p )
	{
		Point< Real , Dim > _p = p;
		_xForm( _p );
		return _stream.nextPoint( _p );
	}
};

template< class Real , unsigned int Dim , class Data >
class TransformedInputPointStreamWithData : public InputPointStreamWithData< Real , Dim , Data >
{
	std::function< void ( Point< Real , Dim >& , Data& ) > _xForm;
	InputPointStreamWithData< Real , Dim , Data >& _stream;
public:
	TransformedInputPointStreamWithData( std::function< void ( Point< Real , Dim >& , Data& ) > xForm , InputPointStreamWithData< Real , Dim , Data >& stream ) : _xForm(xForm) , _stream(stream) {;}
	virtual void reset( void ){ _stream.reset(); }
	virtual bool nextPoint( Point< Real , Dim >& p , Data& d )
	{
		bool ret = _stream.nextPoint( p , d );
		_xForm( p , d );
		return ret;
	}
};

template< class Real , unsigned int Dim , class Data >
class TransformedOutputPointStreamWithData : public OutputPointStreamWithData< Real , Dim , Data >
{
	std::function< void ( Point< Real , Dim >& , Data& ) > _xForm;
	OutputPointStreamWithData< Real , Dim , Data >& _stream;
public:
	TransformedOutputPointStreamWithData( std::function< void ( Point< Real , Dim >& , Data& ) > xForm , OutputPointStreamWithData< Real , Dim , Data >& stream ) : _xForm(xForm) , _stream(stream) {;}
	virtual void nextPoint( const Point< Real , Dim >& p , const Data& d )
	{
		Point< Real , Dim > _p = p;
		Data _d = d;
		_xForm( _p , _d );
		_stream.nextPoint( _p , _d );
	}
};

template< class Real , unsigned int Dim >
class MemoryInputPointStream : public InputPointStream< Real , Dim >
{
	const Point< Real , Dim >* _points;
	size_t _pointCount;
	size_t _current;
public:
	MemoryInputPointStream( size_t pointCount , const Point< Real , Dim >* points );
	~MemoryInputPointStream( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim , class Data >
class MemoryInputPointStreamWithData : public InputPointStreamWithData< Real , Dim , Data >
{
	const std::pair< Point< Real , Dim > , Data >* _points;
	size_t _pointCount;
	size_t _current;
public:
	MemoryInputPointStreamWithData( size_t pointCount , const std::pair< Point< Real , Dim > , Data >* points );
	~MemoryInputPointStreamWithData( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p , Data& d );
};

template< class Real , unsigned int Dim >
class ASCIIInputPointStream : public InputPointStream< Real , Dim >
{
	FILE* _fp;
public:
	ASCIIInputPointStream( const char* fileName );
	~ASCIIInputPointStream( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim >
class ASCIIOutputPointStream : public OutputPointStream< Real , Dim >
{
	FILE* _fp;
public:
	ASCIIOutputPointStream( const char* fileName );
	~ASCIIOutputPointStream( void );
	void nextPoint( const Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim , class Data >
class ASCIIInputPointStreamWithData : public InputPointStreamWithData< Real , Dim , Data >
{
	FILE* _fp;
	void (*_ReadData)( FILE* , Data& );
public:
	ASCIIInputPointStreamWithData( const char* fileName , void (*ReadData)( FILE* , Data& ) );
	~ASCIIInputPointStreamWithData( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p , Data& d );
};

template< class Real , unsigned int Dim , class Data >
class ASCIIOutputPointStreamWithData : public OutputPointStreamWithData< Real , Dim , Data >
{
	FILE* _fp;
	void (*_WriteData)( FILE* , const Data& );
public:
	ASCIIOutputPointStreamWithData( const char* fileName , void (*WriteData)( FILE* , const Data& ) );
	~ASCIIOutputPointStreamWithData( void );
	void nextPoint( const Point< Real , Dim >& p , const Data& d );
};

template< class Real , unsigned int Dim >
class BinaryInputPointStream : public InputPointStream< Real , Dim >
{
	FILE* _fp;
public:
	BinaryInputPointStream( const char* filename );
	~BinaryInputPointStream( void ){ fclose( _fp ) , _fp=NULL; }
	void reset( void ){ fseek( _fp , SEEK_SET , 0 ); }
	bool nextPoint( Point< Real , Dim >& p );
};
template< class Real , unsigned int Dim >
class BinaryOutputPointStream : public OutputPointStream< Real , Dim >
{
	FILE* _fp;
public:
	BinaryOutputPointStream( const char* filename );
	~BinaryOutputPointStream( void ){ fclose( _fp ) , _fp=NULL; }
	void reset( void ){ fseek( _fp , SEEK_SET , 0 ); }
	void nextPoint( const Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim , class Data >
class BinaryInputPointStreamWithData : public InputPointStreamWithData< Real , Dim , Data >
{
	FILE* _fp;
	void (*_ReadData)( FILE* , Data& );
public:
	BinaryInputPointStreamWithData( const char* filename , void (*ReadData)( FILE* , Data& ) );
	~BinaryInputPointStreamWithData( void ){ fclose( _fp ) , _fp=NULL; }
	void reset( void ){ fseek( _fp , SEEK_SET , 0 ); }
	bool nextPoint( Point< Real , Dim >& p , Data& d );
};
template< class Real , unsigned int Dim , class Data >
class BinaryOutputPointStreamWithData : public OutputPointStreamWithData< Real , Dim , Data >
{
	FILE* _fp;
	void (*_WriteData)( FILE* , const Data& );
public:
	BinaryOutputPointStreamWithData( const char* filename , void (*WriteData)( FILE* , const Data& ) );
	~BinaryOutputPointStreamWithData( void ){ fclose( _fp ) , _fp=NULL; }
	void reset( void ){ fseek( _fp , SEEK_SET , 0 ); }
	void nextPoint( const Point< Real , Dim >& p , const Data& d );
};

template< class Real , unsigned int Dim >
class PLYInputPointStream : public InputPointStream< Real , Dim >
{
	char* _fileName;
	PlyFile* _ply;
	std::vector< std::string > _elist;

	size_t _pCount , _pIdx;
	void _free( void );
public:
	PLYInputPointStream( const char* fileName );
	~PLYInputPointStream( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim , class Data >
class PLYInputPointStreamWithData : public InputPointStreamWithData< Real , Dim , Data >
{
	struct _PlyVertexWithData : public PlyVertex< Real , Dim > { Data data; };
	char* _fileName;
	PlyFile* _ply;
	std::vector< std::string > _elist;
	PlyProperty* _dataProperties;
	int _dataPropertiesCount;
	bool (*_validationFunction)( const bool* );

	size_t _pCount , _pIdx;
	void _free( void );
public:
	PLYInputPointStreamWithData( const char* fileName , const PlyProperty* dataProperties , int dataPropertiesCount , bool (*validationFunction)( const bool* )=NULL );
	~PLYInputPointStreamWithData( void );
	void reset( void );
	bool nextPoint( Point< Real , Dim >& p , Data& d );
};

template< class Real , unsigned int Dim >
class PLYOutputPointStream : public OutputPointStream< Real , Dim >
{
	PlyFile* _ply;
	size_t _pCount , _pIdx;
public:
	PLYOutputPointStream( const char* fileName , size_t count , int fileType );
	~PLYOutputPointStream( void );
	void nextPoint( const Point< Real , Dim >& p );
};

template< class Real , unsigned int Dim , class Data >
class PLYOutputPointStreamWithData : public OutputPointStreamWithData< Real , Dim , Data >
{
	struct _PlyVertexWithData : public PlyVertex< Real , Dim > { Data data; };
	PlyFile* _ply;
	size_t _pCount , _pIdx;
public:
	PLYOutputPointStreamWithData( const char* fileName , size_t count , int fileType , const PlyProperty* dataProperties , int dataPropertiesCount );
	~PLYOutputPointStreamWithData( void );
	void nextPoint( const Point< Real , Dim >& p , const Data& d );
};

#include "PointStream.inl"
#endif // POINT_STREAM_INCLUDED
