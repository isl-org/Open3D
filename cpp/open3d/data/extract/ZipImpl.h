// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

/* unzip.h -- IO for uncompress .zip files using zlib
   Version 1.1, February 14h, 2010
   part of the MiniZip project - ( http://www.winimage.com/zLibDll/minizip.html
  )

         Copyright (C) 1998-2010 Gilles Vollant (minizip) (
  http://www.winimage.com/zLibDll/minizip.html )

         Modifications of Unzip for Zip64
         Copyright (C) 2007-2008 Even Rouault

         Modifications for Zip64 support on both zip and unzip
         Copyright (C) 2009-2010 Mathias Svensson ( http://result42.com )

         For more info read MiniZip_info.txt

         ---------------------------------------------------------------------------------

        Condition of use and distribution are the same than zlib :

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  ---------------------------------------------------------------------------------

        Changes

        See header of unzip64.c

*/

#pragma once

#ifndef _ZLIBIOAPI64_H
#define _ZLIBIOAPI64_H

#pragma once

#ifdef WIN32
#include <windows.h>
// #define USEWIN32IOAPI

//#ifdef __cplusplus
// extern "C" {
//#endif

// void fill_win32_filefunc OF((zlib_filefunc_def * pzlib_filefunc_def));
// void fill_win32_filefunc64 OF((zlib_filefunc64_def * pzlib_filefunc_def));
// void fill_win32_filefunc64A OF((zlib_filefunc64_def * pzlib_filefunc_def));
// void fill_win32_filefunc64W OF((zlib_filefunc64_def * pzlib_filefunc_def));

//#ifdef __cplusplus
//}
//#endif
#endif

#if (!defined(_WIN32)) && (!defined(WIN32)) && (!defined(__APPLE__))

// Linux needs this to support file operation on files larger then 4+GB
// But might need better if/def to select just the platforms that needs them.

#ifndef __USE_FILE_OFFSET64
#define __USE_FILE_OFFSET64
#endif
#ifndef __USE_LARGEFILE64
#define __USE_LARGEFILE64
#endif
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif
#ifndef _FILE_OFFSET_BIT
#define _FILE_OFFSET_BIT 64
#endif

#endif

#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>

#if defined(USE_FILE32API)
#define fopen64 fopen
#define ftello64 ftell
#define fseeko64 fseek
#else
#ifdef __FreeBSD__
#define fopen64 fopen
#define ftello64 ftello
#define fseeko64 fseeko
#endif
#ifdef _MSC_VER
#define fopen64 fopen
#if (_MSC_VER >= 1400) && (!(defined(NO_MSCVER_FILE64_FUNC)))
#define ftello64 _ftelli64
#define fseeko64 _fseeki64
#else  // old MSC
#define ftello64 ftell
#define fseeko64 fseek
#endif
#endif
#endif

/*
#ifndef ZPOS64_T
  #ifdef _WIN32
                #define ZPOS64_T fpos_t
  #else
    #include <stdint.h>
    #define ZPOS64_T uint64_t
  #endif
#endif
*/

#ifdef HAVE_MINIZIP64_CONF_H
#include "mz64conf.h"
#endif

/* a type choosen by DEFINE */
#ifdef HAVE_64BIT_INT_CUSTOM
typedef 64BIT_INT_CUSTOM_TYPE ZPOS64_T;
#else
#ifdef HAS_STDINT_H
#include "stdint.h"
typedef uint64_t ZPOS64_T;
#else

/* Maximum unsigned 32-bit value used as placeholder for zip64 */
#define MAXU32 0xffffffff

#if defined(_MSC_VER) || defined(__BORLANDC__)
typedef unsigned __int64 ZPOS64_T;
#else
typedef unsigned long long int ZPOS64_T;
#endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ZLIB_FILEFUNC_SEEK_CUR (1)
#define ZLIB_FILEFUNC_SEEK_END (2)
#define ZLIB_FILEFUNC_SEEK_SET (0)

#define ZLIB_FILEFUNC_MODE_READ (1)
#define ZLIB_FILEFUNC_MODE_WRITE (2)
#define ZLIB_FILEFUNC_MODE_READWRITEFILTER (3)

#define ZLIB_FILEFUNC_MODE_EXISTING (4)
#define ZLIB_FILEFUNC_MODE_CREATE (8)

#ifndef ZCALLBACK
#if (defined(WIN32) || defined(_WIN32) || defined(WINDOWS) || \
     defined(_WINDOWS)) &&                                    \
        defined(CALLBACK) && defined(USEWINDOWS_CALLBACK)
#define ZCALLBACK CALLBACK
#else
#define ZCALLBACK
#endif
#endif

typedef voidpf(ZCALLBACK *open_file_func)
        OF((voidpf opaque, const char *filename, int mode));
typedef uLong(ZCALLBACK *read_file_func)
        OF((voidpf opaque, voidpf stream, void *buf, uLong size));
typedef uLong(ZCALLBACK *write_file_func)
        OF((voidpf opaque, voidpf stream, const void *buf, uLong size));
typedef int(ZCALLBACK *close_file_func) OF((voidpf opaque, voidpf stream));
typedef int(ZCALLBACK *testerror_file_func) OF((voidpf opaque, voidpf stream));

typedef long(ZCALLBACK *tell_file_func) OF((voidpf opaque, voidpf stream));
typedef long(ZCALLBACK *seek_file_func)
        OF((voidpf opaque, voidpf stream, uLong offset, int origin));

/* here is the "old" 32 bits structure structure */
typedef struct zlib_filefunc_def_s {
    open_file_func zopen_file;
    read_file_func zread_file;
    write_file_func zwrite_file;
    tell_file_func ztell_file;
    seek_file_func zseek_file;
    close_file_func zclose_file;
    testerror_file_func zerror_file;
    voidpf opaque;
} zlib_filefunc_def;

typedef ZPOS64_T(ZCALLBACK *tell64_file_func)
        OF((voidpf opaque, voidpf stream));
typedef long(ZCALLBACK *seek64_file_func)
        OF((voidpf opaque, voidpf stream, ZPOS64_T offset, int origin));
typedef voidpf(ZCALLBACK *open64_file_func)
        OF((voidpf opaque, const void *filename, int mode));

typedef struct zlib_filefunc64_def_s {
    open64_file_func zopen64_file;
    read_file_func zread_file;
    write_file_func zwrite_file;
    tell64_file_func ztell64_file;
    seek64_file_func zseek64_file;
    close_file_func zclose_file;
    testerror_file_func zerror_file;
    voidpf opaque;
} zlib_filefunc64_def;

void fill_fopen64_filefunc OF((zlib_filefunc64_def * pzlib_filefunc_def));
void fill_fopen_filefunc OF((zlib_filefunc_def * pzlib_filefunc_def));

/* now internal definition, only for zip.c and unzip.h */
typedef struct zlib_filefunc64_32_def_s {
    zlib_filefunc64_def zfile_func64;
    open_file_func zopen32_file;
    tell_file_func ztell32_file;
    seek_file_func zseek32_file;
} zlib_filefunc64_32_def;

#define ZREAD64(filefunc, filestream, buf, size)                             \
    ((*((filefunc).zfile_func64.zread_file))((filefunc).zfile_func64.opaque, \
                                             filestream, buf, size))
#define ZWRITE64(filefunc, filestream, buf, size)                             \
    ((*((filefunc).zfile_func64.zwrite_file))((filefunc).zfile_func64.opaque, \
                                              filestream, buf, size))
//#define ZTELL64(filefunc,filestream)            ((*((filefunc).ztell64_file))
//((filefunc).opaque,filestream)) #define ZSEEK64(filefunc,filestream,pos,mode)
//((*((filefunc).zseek64_file)) ((filefunc).opaque,filestream,pos,mode))
#define ZCLOSE64(filefunc, filestream)                                        \
    ((*((filefunc).zfile_func64.zclose_file))((filefunc).zfile_func64.opaque, \
                                              filestream))
#define ZERROR64(filefunc, filestream)                                        \
    ((*((filefunc).zfile_func64.zerror_file))((filefunc).zfile_func64.opaque, \
                                              filestream))

voidpf call_zopen64 OF((const zlib_filefunc64_32_def *pfilefunc,
                        const void *filename,
                        int mode));
long call_zseek64 OF((const zlib_filefunc64_32_def *pfilefunc,
                      voidpf filestream,
                      ZPOS64_T offset,
                      int origin));
ZPOS64_T call_ztell64 OF((const zlib_filefunc64_32_def *pfilefunc,
                          voidpf filestream));

void fill_zlib_filefunc64_32_def_from_filefunc32(
        zlib_filefunc64_32_def *p_filefunc64_32,
        const zlib_filefunc_def *p_filefunc32);

#define ZOPEN64(filefunc, filename, mode) \
    (call_zopen64((&(filefunc)), (filename), (mode)))
#define ZTELL64(filefunc, filestream) \
    (call_ztell64((&(filefunc)), (filestream)))
#define ZSEEK64(filefunc, filestream, pos, mode) \
    (call_zseek64((&(filefunc)), (filestream), (pos), (mode)))

#ifdef __cplusplus
}
#endif

#endif

#ifndef _unz64_H
#define _unz64_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _ZLIB_H
#include <zlib.h>
#endif

// #ifndef _ZLIBIOAPI_H
// #include "open3d/data/extract/minizip/ioapi.h"
// #endif

// #ifdef HAVE_BZIP2
// #include "bzlib.h"
// #endif

#define Z_BZIP2ED 12

#if defined(STRICTUNZIP) || defined(STRICTZIPUNZIP)
/* like the STRICT of WIN32, we define a pointer that cannot be converted
    from (void*) without cast */
typedef struct TagunzFile__ {
    int unused;
} unzFile__;
typedef unzFile__ *unzFile;
#else
typedef voidp unzFile;
#endif

#define UNZ_OK (0)
#define UNZ_END_OF_LIST_OF_FILE (-100)
#define UNZ_ERRNO (Z_ERRNO)
#define UNZ_EOF (0)
#define UNZ_PARAMERROR (-102)
#define UNZ_BADZIPFILE (-103)
#define UNZ_INTERNALERROR (-104)
#define UNZ_CRCERROR (-105)

/* tm_unz contain date/time info */
typedef struct tm_unz_s {
    uInt tm_sec;  /* seconds after the minute - [0,59] */
    uInt tm_min;  /* minutes after the hour - [0,59] */
    uInt tm_hour; /* hours since midnight - [0,23] */
    uInt tm_mday; /* day of the month - [1,31] */
    uInt tm_mon;  /* months since January - [0,11] */
    uInt tm_year; /* years - [1980..2044] */
} tm_unz;

/* unz_global_info structure contain global data about the ZIPfile
   These data comes from the end of central dir */
typedef struct unz_global_info64_s {
    ZPOS64_T number_entry; /* total number of entries in
                             the central dir on this disk */
    uLong size_comment;    /* size of the global comment of the zipfile */
} unz_global_info64;

typedef struct unz_global_info_s {
    uLong number_entry; /* total number of entries in
                             the central dir on this disk */
    uLong size_comment; /* size of the global comment of the zipfile */
} unz_global_info;

/* unz_file_info contain information about a file in the zipfile */
typedef struct unz_file_info64_s {
    uLong version;              /* version made by                 2 bytes */
    uLong version_needed;       /* version needed to extract       2 bytes */
    uLong flag;                 /* general purpose bit flag        2 bytes */
    uLong compression_method;   /* compression method              2 bytes */
    uLong dosDate;              /* last mod file date in Dos fmt   4 bytes */
    uLong crc;                  /* crc-32                          4 bytes */
    ZPOS64_T compressed_size;   /* compressed size                 8 bytes */
    ZPOS64_T uncompressed_size; /* uncompressed size               8 bytes */
    uLong size_filename;        /* filename length                 2 bytes */
    uLong size_file_extra;      /* extra field length              2 bytes */
    uLong size_file_comment;    /* file comment length             2 bytes */

    uLong disk_num_start; /* disk number start               2 bytes */
    uLong internal_fa;    /* internal file attributes        2 bytes */
    uLong external_fa;    /* external file attributes        4 bytes */

    tm_unz tmu_date;
} unz_file_info64;

typedef struct unz_file_info_s {
    uLong version;            /* version made by                 2 bytes */
    uLong version_needed;     /* version needed to extract       2 bytes */
    uLong flag;               /* general purpose bit flag        2 bytes */
    uLong compression_method; /* compression method              2 bytes */
    uLong dosDate;            /* last mod file date in Dos fmt   4 bytes */
    uLong crc;                /* crc-32                          4 bytes */
    uLong compressed_size;    /* compressed size                 4 bytes */
    uLong uncompressed_size;  /* uncompressed size               4 bytes */
    uLong size_filename;      /* filename length                 2 bytes */
    uLong size_file_extra;    /* extra field length              2 bytes */
    uLong size_file_comment;  /* file comment length             2 bytes */

    uLong disk_num_start; /* disk number start               2 bytes */
    uLong internal_fa;    /* internal file attributes        2 bytes */
    uLong external_fa;    /* external file attributes        4 bytes */

    tm_unz tmu_date;
} unz_file_info;

extern int ZEXPORT unzStringFileNameCompare OF((const char *fileName1,
                                                const char *fileName2,
                                                int iCaseSensitivity));
/*
   Compare two filename (fileName1,fileName2).
   If iCaseSenisivity = 1, comparision is case sensitivity (like strcmp)
   If iCaseSenisivity = 2, comparision is not case sensitivity (like strcmpi
                                or strcasecmp)
   If iCaseSenisivity = 0, case sensitivity is defaut of your operating system
    (like 1 on Unix, 2 on Windows)
*/

extern unzFile ZEXPORT unzOpen OF((const char *path));
extern unzFile ZEXPORT unzOpen64 OF((const void *path));
/*
  Open a Zip file. path contain the full pathname (by example,
     on a Windows XP computer "c:\\zlib\\zlib113.zip" or on an Unix computer
     "zlib/zlib113.zip".
     If the zipfile cannot be opened (file don't exist or in not valid), the
       return value is NULL.
     Else, the return value is a unzFile Handle, usable with other function
       of this unzip package.
     the "64" function take a const void* pointer, because the path is just the
       value passed to the open64_file_func callback.
     Under Windows, if UNICODE is defined, using fill_fopen64_filefunc, the path
       is a pointer to a wide unicode string (LPCTSTR is LPCWSTR), so const
  char* does not describe the reality
*/

extern unzFile ZEXPORT unzOpen2 OF((const char *path,
                                    zlib_filefunc_def *pzlib_filefunc_def));
/*
   Open a Zip file, like unzOpen, but provide a set of file low level API
      for read/write the zip file (see ioapi.h)
*/

extern unzFile ZEXPORT unzOpen2_64
        OF((const void *path, zlib_filefunc64_def *pzlib_filefunc_def));
/*
   Open a Zip file, like unz64Open, but provide a set of file low level API
      for read/write the zip file (see ioapi.h)
*/

extern int ZEXPORT unzClose OF((unzFile file));
/*
  Close a ZipFile opened with unzOpen.
  If there is files inside the .Zip opened with unzOpenCurrentFile (see later),
    these files MUST be closed with unzCloseCurrentFile before call unzClose.
  return UNZ_OK if there is no problem. */

extern int ZEXPORT unzGetGlobalInfo OF((unzFile file,
                                        unz_global_info *pglobal_info));

extern int ZEXPORT unzGetGlobalInfo64 OF((unzFile file,
                                          unz_global_info64 *pglobal_info));
/*
  Write info about the ZipFile in the *pglobal_info structure.
  No preparation of the structure is needed
  return UNZ_OK if there is no problem. */

extern int ZEXPORT unzGetGlobalComment OF((unzFile file,
                                           char *szComment,
                                           uLong uSizeBuf));
/*
  Get the global comment string of the ZipFile, in the szComment buffer.
  uSizeBuf is the size of the szComment buffer.
  return the number of byte copied or an error code <0
*/

/***************************************************************************/
/* Unzip package allow you browse the directory of the zipfile */

extern int ZEXPORT unzGoToFirstFile OF((unzFile file));
/*
  Set the current file of the zipfile to the first file.
  return UNZ_OK if there is no problem
*/

extern int ZEXPORT unzGoToNextFile OF((unzFile file));
/*
  Set the current file of the zipfile to the next file.
  return UNZ_OK if there is no problem
  return UNZ_END_OF_LIST_OF_FILE if the actual file was the latest.
*/

extern int ZEXPORT unzLocateFile OF((unzFile file,
                                     const char *szFileName,
                                     int iCaseSensitivity));
/*
  Try locate the file szFileName in the zipfile.
  For the iCaseSensitivity signification, see unzStringFileNameCompare

  return value :
  UNZ_OK if the file is found. It becomes the current file.
  UNZ_END_OF_LIST_OF_FILE if the file is not found
*/

/* ****************************************** */
/* Ryan supplied functions */
/* unz_file_info contain information about a file in the zipfile */
typedef struct unz_file_pos_s {
    uLong pos_in_zip_directory; /* offset in zip file directory */
    uLong num_of_file;          /* # of file */
} unz_file_pos;

extern int ZEXPORT unzGetFilePos(unzFile file, unz_file_pos *file_pos);

extern int ZEXPORT unzGoToFilePos(unzFile file, unz_file_pos *file_pos);

typedef struct unz64_file_pos_s {
    ZPOS64_T pos_in_zip_directory; /* offset in zip file directory */
    ZPOS64_T num_of_file;          /* # of file */
} unz64_file_pos;

extern int ZEXPORT unzGetFilePos64(unzFile file, unz64_file_pos *file_pos);

extern int ZEXPORT unzGoToFilePos64(unzFile file,
                                    const unz64_file_pos *file_pos);

/* ****************************************** */

extern int ZEXPORT unzGetCurrentFileInfo64 OF((unzFile file,
                                               unz_file_info64 *pfile_info,
                                               char *szFileName,
                                               uLong fileNameBufferSize,
                                               void *extraField,
                                               uLong extraFieldBufferSize,
                                               char *szComment,
                                               uLong commentBufferSize));

extern int ZEXPORT unzGetCurrentFileInfo OF((unzFile file,
                                             unz_file_info *pfile_info,
                                             char *szFileName,
                                             uLong fileNameBufferSize,
                                             void *extraField,
                                             uLong extraFieldBufferSize,
                                             char *szComment,
                                             uLong commentBufferSize));
/*
  Get Info about the current file
  if pfile_info!=NULL, the *pfile_info structure will contain somes info about
        the current file
  if szFileName!=NULL, the filemane string will be copied in szFileName
            (fileNameBufferSize is the size of the buffer)
  if extraField!=NULL, the extra field information will be copied in extraField
            (extraFieldBufferSize is the size of the buffer).
            This is the Central-header version of the extra field
  if szComment!=NULL, the comment string of the file will be copied in szComment
            (commentBufferSize is the size of the buffer)
*/

/** Addition for GDAL : START */

extern ZPOS64_T ZEXPORT unzGetCurrentFileZStreamPos64 OF((unzFile file));

/** Addition for GDAL : END */

/***************************************************************************/
/* for reading the content of the current zipfile, you can open it, read data
   from it, and close it (you can close it before reading all the file)
   */

extern int ZEXPORT unzOpenCurrentFile OF((unzFile file));
/*
  Open for reading data the current file in the zipfile.
  If there is no error, the return value is UNZ_OK.
*/

extern int ZEXPORT unzOpenCurrentFilePassword OF((unzFile file,
                                                  const char *password));
/*
  Open for reading data the current file in the zipfile.
  password is a crypting password
  If there is no error, the return value is UNZ_OK.
*/

extern int ZEXPORT unzOpenCurrentFile2
        OF((unzFile file, int *method, int *level, int raw));
/*
  Same than unzOpenCurrentFile, but open for read raw the file (not uncompress)
    if raw==1
  *method will receive method of compression, *level will receive level of
     compression
  note : you can set level parameter as NULL (if you did not want known level,
         but you CANNOT set method parameter as NULL
*/

extern int ZEXPORT unzOpenCurrentFile3 OF(
        (unzFile file, int *method, int *level, int raw, const char *password));
/*
  Same than unzOpenCurrentFile, but open for read raw the file (not uncompress)
    if raw==1
  *method will receive method of compression, *level will receive level of
     compression
  note : you can set level parameter as NULL (if you did not want known level,
         but you CANNOT set method parameter as NULL
*/

extern int ZEXPORT unzCloseCurrentFile OF((unzFile file));
/*
  Close the file in zip opened with unzOpenCurrentFile
  Return UNZ_CRCERROR if all the file was read but the CRC is not good
*/

extern int ZEXPORT unzReadCurrentFile OF((unzFile file,
                                          voidp buf,
                                          unsigned len));
/*
  Read bytes from the current file (opened by unzOpenCurrentFile)
  buf contain buffer where data must be copied
  len the size of buf.

  return the number of byte copied if somes bytes are copied
  return 0 if the end of file was reached
  return <0 with error code if there is an error
    (UNZ_ERRNO for IO error, or zLib error for uncompress error)
*/

extern z_off_t ZEXPORT unztell OF((unzFile file));

extern ZPOS64_T ZEXPORT unztell64 OF((unzFile file));
/*
  Give the current position in uncompressed data
*/

extern int ZEXPORT unzeof OF((unzFile file));
/*
  return 1 if the end of file was reached, 0 elsewhere
*/

extern int ZEXPORT unzGetLocalExtrafield OF((unzFile file,
                                             voidp buf,
                                             unsigned len));
/*
  Read extra field from the current file (opened by unzOpenCurrentFile)
  This is the local-header version of the extra field (sometimes, there is
    more info in the local-header version than in the central-header)

  if buf==NULL, it return the size of the local extra field

  if buf!=NULL, len is the size of the buffer, the extra header is copied in
    buf.
  the return value is the number of bytes copied in buf, or (if <0)
    the error code
*/

/***************************************************************************/

/* Get the current file offset */
extern ZPOS64_T ZEXPORT unzGetOffset64(unzFile file);
extern uLong ZEXPORT unzGetOffset(unzFile file);

/* Set the current file offset */
extern int ZEXPORT unzSetOffset64(unzFile file, ZPOS64_T pos);
extern int ZEXPORT unzSetOffset(unzFile file, uLong pos);

#ifdef __cplusplus
}
#endif

#endif /* _unz64_H */
