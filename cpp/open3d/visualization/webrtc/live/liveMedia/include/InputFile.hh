/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// "liveMedia"
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// Common routines for opening/closing named input files
// C++ header

#ifndef _INPUT_FILE_HH
#define _INPUT_FILE_HH

#include <UsageEnvironment.hh>
#include <stdio.h>

#if (defined(__WIN32__) || defined(_WIN32) || defined(_WIN32_WCE))
#ifndef _WIN32_WCE
// Include header files that might be needed by Windows (in code that uses this header file):
#include <io.h>
#include <fcntl.h>
#endif

#define READ_FROM_FILES_SYNCHRONOUSLY 1
    // Because Windows is a silly toy operating system that doesn't (reliably) treat
    // open files as being readable sockets (which can be handled within the default
    // "BasicTaskScheduler" event loop, using "select()"), we implement file reading
    // in Windows using synchronous, rather than asynchronous, I/O.  This can severely
    // limit the scalability of servers using this code that run on Windows.
    // If this is a problem for you, then either use a better operating system,
    // or else write your own Windows-specific event loop ("TaskScheduler" subclass)
    // that can handle readable data in Windows open files as an event.
#endif

#ifndef _WIN32_WCE
#include <sys/stat.h>
#endif

FILE* OpenInputFile(UsageEnvironment& env, char const* fileName);

void CloseInputFile(FILE* fid);

#undef GetFileSize // because some platforms already define this as a macro
u_int64_t GetFileSize(char const* fileName, FILE* fid);
    // 0 means zero-length, unbounded, or unknown

int64_t SeekFile64(FILE *fid, int64_t offset, int whence);
    // A platform-independent routine for seeking within (possibly) large files

int64_t TellFile64(FILE *fid);
    // A platform-independent routine for reporting the position within
    // (possibly) large files

Boolean FileIsSeekable(FILE *fid);
    // Tests whether "fid" is seekable, by trying to seek within it.

#endif
