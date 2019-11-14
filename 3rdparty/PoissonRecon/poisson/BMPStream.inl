/*
Copyright (c) 2010, Michael Kazhdan
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
#ifndef BMP_STREAM_INCLUDED
#define BMP_STREAM_INCLUDED

#include <stdio.h>
#include "Array.h"

/* constants for the biCompression field */
#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L

/* Some magic numbers */

#define BMP_BF_TYPE 0x4D42
/* word BM */

#define BMP_BF_OFF_BITS 54
/* 14 for file header + 40 for info header (not sizeof(), but packed size) */

#define BMP_BI_SIZE 40
/* packed size of info header */

#ifndef _WIN32
typedef struct tagBITMAPFILEHEADER
{
    unsigned short int bfType;
    unsigned int bfSize;
    unsigned short int bfReserved1;
    unsigned short int bfReserved2;
    unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short int biPlanes;
    unsigned short int biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
} BITMAPINFOHEADER;
#endif // !_WIN32


struct BMPInfo
{
	FILE* fp;
	Pointer( unsigned char ) data;
	int width , lineLength;
};
inline void BMPGetImageInfo( char* fileName , int& width , int& height , int& channels , int& bytesPerChannel )
{
    BITMAPFILEHEADER bmfh;
    BITMAPINFOHEADER bmih;

	FILE* fp = fopen( fileName , "rb" );
	if( !fp ) ERROR_OUT( "Failed to open: %s" , fileName );

	fread( &bmfh , sizeof( BITMAPFILEHEADER ) , 1 , fp );
	fread( &bmih , sizeof( BITMAPINFOHEADER ) , 1 , fp );

	if( bmfh.bfType!=BMP_BF_TYPE || bmfh.bfOffBits!=BMP_BF_OFF_BITS ){ fclose(fp) ; ERROR_OUT( "Bad bitmap file header" ); };
	if( bmih.biSize!=BMP_BI_SIZE || bmih.biWidth<=0 || bmih.biHeight<=0 || bmih.biPlanes!=1 || bmih.biBitCount!=24 || bmih.biCompression!=BI_RGB ) { fclose(fp) ; ERROR_OUT( "Bad bitmap file info" ); }
	width           = bmih.biWidth;
	height          = bmih.biHeight;
	channels        = 3;
	bytesPerChannel = 1;
	int lineLength = width * channels;
	if( lineLength % 4 ) lineLength = (lineLength / 4 + 1) * 4;
	if( bmih.biSizeImage!=lineLength*height ){ fclose(fp) ; ERROR_OUT( "Bad bitmap image size" ) , fclose( fp ); };
	fclose( fp );
}

inline void* BMPInitRead( char* fileName , int& width , int& height )
{
    BITMAPFILEHEADER bmfh;
    BITMAPINFOHEADER bmih;

	BMPInfo* info = (BMPInfo*)malloc( sizeof( BMPInfo ) );
	info->fp = fopen( fileName , "rb" );
	if( !info->fp ) ERROR_OUT( "Failed to open: %s" , fileName );

	fread( &bmfh , sizeof( BITMAPFILEHEADER ) , 1 , info->fp );
	fread( &bmih , sizeof( BITMAPINFOHEADER ) , 1 , info->fp );

	if( bmfh.bfType!=BMP_BF_TYPE || bmfh.bfOffBits!=BMP_BF_OFF_BITS ) ERROR_OUT( "Bad bitmap file header" );
	if( bmih.biSize!=BMP_BI_SIZE || bmih.biWidth<=0 || bmih.biHeight<=0 || bmih.biPlanes!=1 || bmih.biBitCount!=24 || bmih.biCompression!=BI_RGB ) ERROR_OUT( "Bad bitmap file info" );

	info->width = width = bmih.biWidth;
	height = bmih.biHeight;
	info->lineLength = width * 3;
	if( info->lineLength % 4 ) info->lineLength = (info->lineLength / 4 + 1) * 4;
	if( bmih.biSizeImage!=info->lineLength*height ) ERROR_OUT( "Bad bitmap image size" );
	info->data = AllocPointer< unsigned char >( info->lineLength );
	if( !info->data ) ERROR_OUT( "Could not allocate memory for bitmap data" );

	fseek( info->fp , (long) bmfh.bfOffBits , SEEK_SET );
	fseek( info->fp , (long) info->lineLength * height , SEEK_CUR );
	return info;
}
template< int Channels , bool HDR >
inline void* BMPInitWrite( char* fileName , int width , int height , int quality )
{
	if( HDR ) WARN( "No HDR support for JPEG" );
	BITMAPFILEHEADER bmfh;
	BITMAPINFOHEADER bmih;

	BMPInfo* info = (BMPInfo*)malloc( sizeof( BMPInfo ) );
	info->fp = fopen( fileName , "wb" );
	if( !info->fp ) ERROR_OUT( "Failed to open: %s" , fileName );
	info->width = width;

	info->lineLength = width * 3;	/* RGB */
	if( info->lineLength % 4 ) info->lineLength = (info->lineLength / 4 + 1) * 4;
	info->data = AllocPointer< unsigned char >( info->lineLength );
	if( !info->data ) ERROR_OUT( "Could not allocate memory for bitmap data" );
	/* Write file header */

	bmfh.bfType = BMP_BF_TYPE;
	bmfh.bfSize = BMP_BF_OFF_BITS + info->lineLength * height;
	bmfh.bfReserved1 = 0;
	bmfh.bfReserved2 = 0;
	bmfh.bfOffBits = BMP_BF_OFF_BITS;

	fwrite( &bmfh , sizeof(BITMAPFILEHEADER) , 1 , info->fp );

	bmih.biSize = BMP_BI_SIZE;
	bmih.biWidth = width;
	bmih.biHeight = -height;
	bmih.biPlanes = 1;
	bmih.biBitCount = 24;			/* RGB */
	bmih.biCompression = BI_RGB;	/* RGB */
	bmih.biSizeImage = info->lineLength * (unsigned int) bmih.biHeight;	/* RGB */
	bmih.biXPelsPerMeter = 2925;
	bmih.biYPelsPerMeter = 2925;
	bmih.biClrUsed = 0;
	bmih.biClrImportant = 0;

	fwrite( &bmih , sizeof(BITMAPINFOHEADER) , 1 , info->fp );

	return info;
}
template< int Channels , class ChannelType >
inline void BMPWriteRow( Pointer( ChannelType ) pixels , void* v , int j )
{
	BMPInfo* info = (BMPInfo*)v;
	ConvertRow< ChannelType , unsigned char >( pixels , info->data , info->width , Channels , 3 );
	for( int i=0 ; i<info->width ; i++ ) { unsigned char temp = info->data[i*3] ; info->data[i*3] = info->data[i*3+2] ; info->data[i*3+2] = temp; }
	fwrite( info->data , sizeof(unsigned char) , info->width*3 , info->fp );
	int nbytes = info->width*3;
	while( nbytes % 4 ) putc( 0 , info->fp ) , nbytes++;
}
template< int Channels , class ChannelType >
void BMPReadRow( Pointer( ChannelType ) pixels , void* v , int j )
{
	BMPInfo* info = (BMPInfo*)v;

	fseek( info->fp , -info->lineLength , SEEK_CUR );
    fread( info->data , 1 , info->lineLength , info->fp );
	fseek( info->fp , -info->lineLength , SEEK_CUR );
	if( ferror(info->fp) ) ERROR_OUT( "Error reading bitmap row" );
	for( int i=0 ; i<info->width ; i++ ) { unsigned char temp = info->data[i*3] ; info->data[i*3] = info->data[i*3+2] ; info->data[i*3+2] = temp; }
	ConvertRow< unsigned char , ChannelType >( ( ConstPointer( unsigned char ) )info->data , pixels , info->width , 3 , Channels );
}
inline void BMPFinalize( void* v )
{
	BMPInfo* info = (BMPInfo*)v;
	fclose( info->fp );
	FreePointer( info->data );
	free( info );
}

inline void BMPFinalizeWrite( void* v ){ BMPFinalize( v ); }
inline void BMPFinalizeRead ( void* v ){ BMPFinalize( v ); }
#endif // BMP_STREAM_INCLUDED