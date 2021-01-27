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
/* "groupsock" interface
 * Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
 * Common include files, typically used for networking
 */

#ifndef _NET_COMMON_H
#define _NET_COMMON_H

#if defined(__WIN32__) || defined(_WIN32) || defined(_WIN32_WCE)
/* Windows */
#if defined(WINNT) || defined(_WINNT) || defined(__BORLANDC__) || \
        defined(__MINGW32__) || defined(_WIN32_WCE) || defined(_MSC_VER)
#define _MSWSOCK_
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#include <errno.h>
#include <string.h>
#include <windows.h>

#define closeSocket closesocket
#ifdef EWOULDBLOCK
#undef EWOULDBLOCK
#endif
#ifdef EINPROGRESS
#undef EINPROGRESS
#endif
#ifdef EAGAIN
#undef EAGAIN
#endif
#ifdef EINTR
#undef EINTR
#endif
#define EWOULDBLOCK WSAEWOULDBLOCK
#define EINPROGRESS WSAEWOULDBLOCK
#define EAGAIN WSAEWOULDBLOCK
#define EINTR WSAEINTR

#if defined(_WIN32_WCE)
#define NO_STRSTREAM 1
#endif

/* Definitions of size-specific types: */
typedef __int64 int64_t;
typedef unsigned __int64 u_int64_t;

typedef int int32_t;
typedef unsigned u_int32_t;

typedef short int16_t;
typedef unsigned short u_int16_t;

typedef unsigned char u_int8_t;

// For "uintptr_t" and "intptr_t", we assume that if they're not already
// defined, then this must be an old, 32-bit version of Windows:
#if !defined(_MSC_STDINT_H_) && !defined(_UINTPTR_T_DEFINED) && \
        !defined(_UINTPTR_T_DECLARED) && !defined(_UINTPTR_T)
typedef unsigned uintptr_t;
#endif
#if !defined(_MSC_STDINT_H_) && !defined(_INTPTR_T_DEFINED) && \
        !defined(_INTPTR_T_DECLARED) && !defined(_INTPTR_T)
typedef int intptr_t;
#endif

#elif defined(VXWORKS)
/* VxWorks */
#include <hostLib.h>
#include <ioLib.h>
#include <resolvLib.h>
#include <sockLib.h>
#include <sys/times.h>
#include <time.h>
#include <timers.h>

typedef unsigned int u_int32_t;
typedef unsigned short u_int16_t;
typedef unsigned char u_int8_t;

#else
/* Unix */
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(_QNX4)
#include <sys/select.h>
#include <unix.h>
#endif

#define closeSocket close

#ifdef SOLARIS
#define u_int64_t uint64_t
#define u_int32_t uint32_t
#define u_int16_t uint16_t
#define u_int8_t uint8_t
#endif
#endif

#ifndef SOCKLEN_T
#define SOCKLEN_T int
#endif

#endif
