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
// Bit Vector data structure
// C++ header

#ifndef _BIT_VECTOR_HH
#define _BIT_VECTOR_HH

#ifndef _BOOLEAN_HH
#include "Boolean.hh"
#endif

class BitVector {
public:
  BitVector(unsigned char* baseBytePtr,
	    unsigned baseBitOffset,
	    unsigned totNumBits);

  void setup(unsigned char* baseBytePtr,
	     unsigned baseBitOffset,
	     unsigned totNumBits);

  void putBits(unsigned from, unsigned numBits); // "numBits" <= 32
  void put1Bit(unsigned bit);

  unsigned getBits(unsigned numBits); // "numBits" <= 32
  unsigned get1Bit();
  Boolean get1BitBoolean() { return get1Bit() != 0; }

  void skipBits(unsigned numBits);

  unsigned curBitIndex() const { return fCurBitIndex; }
  unsigned totNumBits() const { return fTotNumBits; }
  unsigned numBitsRemaining() const { return fTotNumBits - fCurBitIndex; }

  unsigned get_expGolomb();
      // Returns the value of the next bits, assuming that they were encoded using an exponential-Golomb code of order 0
  int get_expGolombSigned(); // signed version of the above

private:
  unsigned char* fBaseBytePtr;
  unsigned fBaseBitOffset;
  unsigned fTotNumBits;
  unsigned fCurBitIndex;
};

// A general bit copy operation:
void shiftBits(unsigned char* toBasePtr, unsigned toBitOffset,
	       unsigned char const* fromBasePtr, unsigned fromBitOffset,
	       unsigned numBits);

#endif
