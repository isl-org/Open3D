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
// Because MD5 may not be implemented (at least, with the same interface) on all systems,
// we have our own implementation.
// C++ header

#ifndef _OUR_MD5_HH
#define _OUR_MD5_HH

extern char* our_MD5Data(unsigned char const* data, unsigned dataSize, char* outputDigest);
    // "outputDigest" must be either NULL (in which case this function returns a heap-allocated
    // buffer, which should be later delete[]d by the caller), or else it must point to
    // a (>=)33-byte buffer (which this function will also return).

extern unsigned char* our_MD5DataRaw(unsigned char const* data, unsigned dataSize,
				     unsigned char* outputDigest);
    // Like "ourMD5Data()", except that it returns the digest in 'raw' binary form, rather than
    // as an ASCII hex string.
    // "outputDigest" must be either NULL (in which case this function returns a heap-allocated
    // buffer, which should be later delete[]d by the caller), or else it must point to
    // a (>=)16-byte buffer (which this function will also return).

#endif
