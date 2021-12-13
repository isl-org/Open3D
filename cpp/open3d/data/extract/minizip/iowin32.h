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

/* iowin32.h -- IO base function header for compress/uncompress .zip
     Version 1.1, February 14h, 2010
     part of the MiniZip project - (
   http://www.winimage.com/zLibDll/minizip.html )

         Copyright (C) 1998-2010 Gilles Vollant (minizip) (
   http://www.winimage.com/zLibDll/minizip.html )

         Modifications for Zip64 support
         Copyright (C) 2009-2010 Mathias Svensson ( http://result42.com )

         For more info read MiniZip_info.txt

*/

#pragma once

#include <windows.h>

//#ifdef __cplusplus
// extern "C" {
//#endif

void fill_win32_filefunc OF((zlib_filefunc_def * pzlib_filefunc_def));
void fill_win32_filefunc64 OF((zlib_filefunc64_def * pzlib_filefunc_def));
void fill_win32_filefunc64A OF((zlib_filefunc64_def * pzlib_filefunc_def));
void fill_win32_filefunc64W OF((zlib_filefunc64_def * pzlib_filefunc_def));

//#ifdef __cplusplus
//}
//#endif
