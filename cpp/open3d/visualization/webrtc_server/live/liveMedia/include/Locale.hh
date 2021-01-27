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
// Support for temporarily setting the locale (e.g., to "C" or "POSIX") for (e.g.) parsing or printing
// floating-point numbers in protocol headers, or calling toupper()/tolower() on human-input strings.
// C++ header

#ifndef _LOCALE_HH
#define _LOCALE_HH

// If you're on a system that (for whatever reason) doesn't have either the "setlocale()" or the "newlocale()" function, then
// add "-DLOCALE_NOT_USED" to your "config.*" file.

// If you're on a system that (for whatever reason) has "setlocale()" but not "newlocale()", then
// add "-DNEWLOCALE_NOT_USED" to your "config.*" file.
// (Note that -DLOCALE_NOT_USED implies -DNEWLOCALE_NOT_USED; you do not need both.)
// Also, for Windows systems, we define "NEWLOCALE_NOT_USED" by default, because at least some Windows systems
// (or their development environments) don't have "newlocale()".  If, however, your Windows system *does* have "newlocale()",
// then you can override this by defining "NEWLOCALE_USED" before #including this file.

// Finally, some old development environments need a header file "xlocale.h" to use "newlocale()".
// Should you need this header file, add "-DNEED_XLOCALE_H" to your "config.*" file.

#ifdef NEWLOCALE_USED
#undef LOCALE_NOT_USED
#undef NEWLOCALE_NOT_USED
#else
#if defined(__WIN32__) || defined(_WIN32)
#define NEWLOCALE_NOT_USED 1
#endif
#endif

#ifndef LOCALE_NOT_USED
#include <locale.h>
#ifndef NEWLOCALE_NOT_USED
#ifdef NEED_XLOCALE_H
#include <xlocale.h>
#endif
#endif
#endif


enum LocaleCategory { All, Numeric }; // define and implement more categories later, as needed

class Locale {
public:
  Locale(char const* newLocale, LocaleCategory category = All);
  virtual ~Locale();

private:
#ifndef LOCALE_NOT_USED
#ifndef NEWLOCALE_NOT_USED
  locale_t fLocale, fPrevLocale;
#else
  int fCategoryNum;
  char* fPrevLocale;
#endif
#endif
};

#endif
