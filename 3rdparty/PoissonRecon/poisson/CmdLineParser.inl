/* -*- C++ -*-
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

#include <cassert>
#include <string.h>

#if defined( WIN32 ) || defined( _WIN64 )
inline int strcasecmp( const char* c1 , const char* c2 ){ return _stricmp( c1 , c2 ); }
#endif // WIN32 || _WIN64

template< > void cmdLineCleanUp< int    >( int*    t ){ }
template< > void cmdLineCleanUp< float  >( float*  t ){ }
template< > void cmdLineCleanUp< double >( double* t ){ }
template< > void cmdLineCleanUp< char*  >( char** t ){ if( *t ) free( *t ) ; *t = NULL; }
template< > int    cmdLineInitialize< int    >( void ){ return 0; }
template< > float  cmdLineInitialize< float  >( void ){ return 0.f; }
template< > double cmdLineInitialize< double >( void ){ return 0.; }
template< > char*  cmdLineInitialize< char*  >( void ){ return NULL; }
template< > void cmdLineWriteValue< int    >( int    t , char* str ){ sprintf( str , "%d" , t ); }
template< > void cmdLineWriteValue< float  >( float  t , char* str ){ sprintf( str , "%f" , t ); }
template< > void cmdLineWriteValue< double >( double t , char* str ){ sprintf( str , "%f" , t ); }
template< > void cmdLineWriteValue< char*  >( char*  t , char* str ){ if( t ) sprintf( str , "%s" , t ) ; else str[0]=0; }
template< > int    cmdLineCopy( int    t ){ return t;  }
template< > float  cmdLineCopy( float  t ){ return t;  }
template< > double cmdLineCopy( double t ){ return t;  }
#if defined( WIN32 ) || defined( _WIN64 )
template< > char*  cmdLineCopy( char* t ){ return _strdup( t ); }
#else // !WIN32 && !_WIN64
template< > char*  cmdLineCopy( char* t ){ return strdup( t ); }
#endif // WIN32 || _WIN64
template< > int    cmdLineStringToType( const char* str ){ return atoi( str ); }
template< > float  cmdLineStringToType( const char* str ){ return float( atof( str ) ); }
template< > double cmdLineStringToType( const char* str ){ return double( atof( str ) ); }
#if defined( WIN32 ) || defined( _WIN64 )
template< > char*  cmdLineStringToType( const char* str ){ return _strdup( str ); }
#else // !WIN32 && !_WIN64
template< > char*  cmdLineStringToType( const char* str ){ return  strdup( str ); }
#endif // WIN32 || _WIN64


/////////////////////
// cmdLineReadable //
/////////////////////
#if defined( WIN32 ) || defined( _WIN64 )
inline cmdLineReadable::cmdLineReadable( const char *name ) : set(false) { this->name = _strdup( name ); }
#else // !WIN32 && !_WIN64
inline cmdLineReadable::cmdLineReadable( const char *name ) : set(false) { this->name =  strdup( name ); }
#endif // WIN32 || _WIN64

inline cmdLineReadable::~cmdLineReadable( void ){ if( name ) free( name ) ; name = NULL; }
inline int cmdLineReadable::read( char** , int ){ set = true ; return 0; }
inline void cmdLineReadable::writeValue( char* str ) const { str[0] = 0; }

//////////////////////
// cmdLineParameter //
//////////////////////
template< class Type > cmdLineParameter< Type >::~cmdLineParameter( void ) { cmdLineCleanUp( &value ); }
template< class Type > cmdLineParameter< Type >::cmdLineParameter( const char *name ) : cmdLineReadable( name ){ value = cmdLineInitialize< Type >(); }
template< class Type > cmdLineParameter< Type >::cmdLineParameter( const char *name , Type v ) : cmdLineReadable( name ){ value = cmdLineCopy< Type >( v ); }
template< class Type >
int cmdLineParameter< Type >::read( char** argv , int argc )
{
	if( argc>0 )
	{
		cmdLineCleanUp< Type >( &value ) , value = cmdLineStringToType< Type >( argv[0] );
		set = true;
		return 1;
	}
	else return 0;
}
template< class Type >
void cmdLineParameter< Type >::writeValue( char* str ) const { cmdLineWriteValue< Type >( value , str ); }


///////////////////////////
// cmdLineParameterArray //
///////////////////////////
template< class Type , int Dim >
cmdLineParameterArray< Type , Dim >::cmdLineParameterArray( const char *name , const Type* v ) : cmdLineReadable( name )
{
	if( v ) for( int i=0 ; i<Dim ; i++ ) values[i] = cmdLineCopy< Type >( v[i] );
	else    for( int i=0 ; i<Dim ; i++ ) values[i] = cmdLineInitialize< Type >();
}
template< class Type , int Dim >
cmdLineParameterArray< Type , Dim >::~cmdLineParameterArray( void ){ for( int i=0 ; i<Dim ; i++ ) cmdLineCleanUp< Type >( values+i ); }
template< class Type , int Dim >
int cmdLineParameterArray< Type , Dim >::read( char** argv , int argc )
{
	if( argc>=Dim )
	{
		for( int i=0 ; i<Dim ; i++ ) cmdLineCleanUp< Type >( values+i ) , values[i] = cmdLineStringToType< Type >( argv[i] );
		set = true;
		return Dim;
	}
	else return 0;
}
template< class Type , int Dim >
void cmdLineParameterArray< Type , Dim >::writeValue( char* str ) const
{
	char* temp=str;
	for( int i=0 ; i<Dim ; i++ )
	{
		cmdLineWriteValue< Type >( values[i] , temp );
		temp = str+strlen( str );
	}
}
///////////////////////
// cmdLineParameters //
///////////////////////
template< class Type >
cmdLineParameters< Type >::cmdLineParameters( const char* name ) : cmdLineReadable( name ) , values(NULL) , count(0) { }
template< class Type >
cmdLineParameters< Type >::~cmdLineParameters( void )
{
	if( values ) delete[] values;
	values = NULL;
	count = 0;
}
template< class Type >
int cmdLineParameters< Type >::read( char** argv , int argc )
{
	if( values ) delete[] values;
	values = NULL;

	if( argc>0 )
	{
		count = atoi(argv[0]);
		if( count <= 0 || argc <= count ) return 1;
		values = new Type[count];
		if( !values ) return 0;
		for( int i=0 ; i<count ; i++ ) values[i] = cmdLineStringToType< Type >( argv[i+1] );
		set = true;
		return count+1;
	}
	else return 0;
}
template< class Type >
void cmdLineParameters< Type >::writeValue( char* str ) const
{
	char* temp=str;
	for( int i=0 ; i<count ; i++ )
	{
		cmdLineWriteValue< Type >( values[i] , temp );
		temp = str+strlen( str );
	}
}


inline char* FileExtension( char* fileName )
{
	char* temp = fileName;
	for( int i=0 ; i<strlen(fileName) ; i++ ) if( fileName[i]=='.' ) temp = &fileName[i+1];
	return temp;
}

inline char* GetFileExtension( const char* fileName )
{
	char* fileNameCopy;
	char* ext=NULL;
	char* temp;

	fileNameCopy=new char[strlen(fileName)+1];
	assert(fileNameCopy);
	strcpy(fileNameCopy,fileName);
	temp=strtok(fileNameCopy,".");
	while(temp!=NULL)
	{
		if(ext!=NULL){delete[] ext;}
		ext=new char[strlen(temp)+1];
		assert(ext);
		strcpy(ext,temp);
		temp=strtok(NULL,".");
	}
	delete[] fileNameCopy;
	return ext;
}
inline char* GetLocalFileName( const char* fileName )
{
	char* fileNameCopy;
	char* name=NULL;
	char* temp;

	fileNameCopy=new char[strlen(fileName)+1];
	assert(fileNameCopy);
	strcpy(fileNameCopy,fileName);
	temp=strtok(fileNameCopy,"\\");
	while(temp!=NULL){
		if(name!=NULL){delete[] name;}
		name=new char[strlen(temp)+1];
		assert(name);
		strcpy(name,temp);
		temp=strtok(NULL,"\\");
	}
	delete[] fileNameCopy;
	return name;
}
inline char* LocalFileName( char* fileName )
{
	char* temp = fileName;
	for( int i=0 ; i<(int)strlen(fileName) ; i++ ) if( fileName[i] =='\\' ) temp = &fileName[i+1];
	return temp;
}
inline char* DirectoryName( char* fileName )
{
	for( int i=int( strlen(fileName) )-1 ; i>=0 ; i-- )
		if( fileName[i] =='\\' )
		{
			fileName[i] = 0;
			return fileName;
		}
	fileName[0] = 0;
	return fileName;
}

inline void cmdLineParse( int argc , char **argv , cmdLineReadable** params )
{
	while( argc>0 )
	{
		if( argv[0][0]=='-' && argv[0][1]=='-' )
		{
			cmdLineReadable* readable=NULL;
			for( int i=0 ; params[i]!=NULL && readable==NULL ; i++ ) if( !strcasecmp( params[i]->name , argv[0]+2 ) ) readable = params[i];
			if( readable )
			{
				int j = readable->read( argv+1 , argc-1 );
				argv += j , argc -= j;
			}
			else
			{
				WARN( "Invalid option: " , argv[0] );
				for( int i=0 ; params[i]!=NULL ; i++ ) fprintf( stderr , "\t--%s\n" , params[i]->name );
			}
		}
		else WARN( "Parameter name should be of the form --<name>: " , argv[0] );
		++argv , --argc;
	}
}

inline char** ReadWords(const char* fileName,int& cnt)
{
	char** names;
	char temp[500];
	FILE* fp;

	fp=fopen(fileName,"r");
	if(!fp){return NULL;}
	cnt=0;
	while(fscanf(fp," %s ",temp)==1){cnt++;}
	fclose(fp);

	names=new char*[cnt];
	if(!names){return NULL;}

	fp=fopen(fileName,"r");
	if(!fp){
		delete[] names;
		cnt=0;
		return NULL;
	}
	cnt=0;
	while(fscanf(fp," %s ",temp)==1){
		names[cnt]=new char[strlen(temp)+1];
		if(!names){
			for(int j=0;j<cnt;j++){delete[] names[j];}
			delete[] names;
			cnt=0;
			fclose(fp);
			return NULL;
		}
		strcpy(names[cnt],temp);
		cnt++;
	}
	fclose(fp);
	return names;
}