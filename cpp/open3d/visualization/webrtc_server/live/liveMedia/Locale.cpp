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
// Support for temporarily setting the locale (e.g., to "C" or "POSIX") for
// (e.g.) parsing or printing floating-point numbers in protocol headers, or
// calling toupper()/tolower() on human-input strings. Implementation

#include "Locale.hh"

#include <strDup.hh>

Locale::Locale(char const* newLocale, LocaleCategory category) {
#ifndef LOCALE_NOT_USED
#ifndef NEWLOCALE_NOT_USED
    int categoryMask = 0;
    switch (category) {
        case All: {
            categoryMask = LC_ALL_MASK;
            break;
        }
        case Numeric: {
            categoryMask = LC_NUMERIC_MASK;
            break;
        }
    }
    fLocale = newlocale(categoryMask, newLocale, NULL);
    fPrevLocale = uselocale(fLocale);
#else
    switch (category) {
        case All: {
            fCategoryNum = LC_ALL;
            break;
        }
        case Numeric: {
            fCategoryNum = LC_NUMERIC;
            break;
        }
    }
    fPrevLocale = strDup(setlocale(fCategoryNum, NULL));
    setlocale(fCategoryNum, newLocale);
#endif
#endif
}

Locale::~Locale() {
#ifndef LOCALE_NOT_USED
#ifndef NEWLOCALE_NOT_USED
    if (fLocale != (locale_t)0) {
        uselocale(fPrevLocale);
        freelocale(fLocale);
    }
#else
    if (fPrevLocale != NULL) {
        setlocale(fCategoryNum, fPrevLocale);
        delete[] fPrevLocale;
    }
#endif
#endif
}
