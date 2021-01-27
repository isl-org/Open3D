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
#ifndef _BOOLEAN_HH
#define _BOOLEAN_HH

#if defined(__BORLANDC__) || (!defined(USE_LIVE555_BOOLEAN) && defined(_MSC_VER) &&  _MSC_VER >= 1400)
// Use the "bool" type defined by the Borland compiler, and MSVC++ 8.0, Visual Studio 2005 and higher
typedef bool Boolean;
#define False false
#define True true
#else
typedef unsigned char Boolean;
#ifndef __MSHTML_LIBRARY_DEFINED__
#ifndef False
const Boolean False = 0;
#endif
#ifndef True
const Boolean True = 1;
#endif

#endif
#endif

#endif
