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
// Because MD5 may not be implemented (at least, with the same interface) on all
// systems, we have our own implementation. Implementation

#include "ourMD5.hh"

#include <NetCommon.h>  // for u_int32_t, u_int64_t
#include <string.h>

#define DIGEST_SIZE_IN_BYTES 16
#define DIGEST_SIZE_IN_HEX_DIGITS (2 * DIGEST_SIZE_IN_BYTES)
#define DIGEST_SIZE_AS_STRING (DIGEST_SIZE_IN_HEX_DIGITS + 1)

// The state of a MD5 computation in progress:

class MD5Context {
public:
    MD5Context();
    ~MD5Context();

    void addData(unsigned char const* inputData, unsigned inputDataSize);
    void end(char* outputDigest /*must point to an array of size DIGEST_SIZE_AS_STRING*/);
    void finalize(unsigned char* outputDigestInBytes);
    // Like "end()", except that the argument is a byte array, of size
    // DIGEST_SIZE_IN_BYTES. This function is used to implement "end()".

private:
    void zeroize();  // to remove potentially sensitive information
    void transform64Bytes(
            unsigned char const block[64]);  // does the actual MD5 transform

private:
    u_int32_t fState[4];  // ABCD
    u_int64_t fBitCount;  // number of bits, modulo 2^64
    unsigned char fWorkingBuffer[64];
};

char* our_MD5Data(unsigned char const* data,
                  unsigned dataSize,
                  char* outputDigest) {
    MD5Context ctx;

    ctx.addData(data, dataSize);

    if (outputDigest == NULL) outputDigest = new char[DIGEST_SIZE_AS_STRING];
    ctx.end(outputDigest);

    return outputDigest;
}

unsigned char* our_MD5DataRaw(unsigned char const* data,
                              unsigned dataSize,
                              unsigned char* outputDigest) {
    MD5Context ctx;

    ctx.addData(data, dataSize);

    if (outputDigest == NULL)
        outputDigest = new unsigned char[DIGEST_SIZE_IN_BYTES];
    ctx.finalize(outputDigest);

    return outputDigest;
}

////////// MD5Context implementation //////////

MD5Context::MD5Context() : fBitCount(0) {
    // Initialize with magic constants:
    fState[0] = 0x67452301;
    fState[1] = 0xefcdab89;
    fState[2] = 0x98badcfe;
    fState[3] = 0x10325476;
}

MD5Context::~MD5Context() { zeroize(); }

void MD5Context::addData(unsigned char const* inputData,
                         unsigned inputDataSize) {
    // Begin by noting how much of our 64-byte working buffer remains unfilled:
    u_int64_t const byteCount = fBitCount >> 3;
    unsigned bufferBytesInUse = (unsigned)(byteCount & 0x3F);
    unsigned bufferBytesRemaining = 64 - bufferBytesInUse;

    // Then update our bit count:
    fBitCount += inputDataSize << 3;

    unsigned i = 0;
    if (inputDataSize >= bufferBytesRemaining) {
        // We have enough input data to do (64-byte) MD5 transforms.
        // Do this now, starting with a transform on our working buffer, then
        // with (as many as possible) transforms on rest of the input data.

        memcpy((unsigned char*)&fWorkingBuffer[bufferBytesInUse],
               (unsigned char*)inputData, bufferBytesRemaining);
        transform64Bytes(fWorkingBuffer);
        bufferBytesInUse = 0;

        for (i = bufferBytesRemaining; i + 63 < inputDataSize; i += 64) {
            transform64Bytes(&inputData[i]);
        }
    }

    // Copy any remaining (and currently un-transformed) input data into our
    // working buffer:
    if (i < inputDataSize) {
        memcpy((unsigned char*)&fWorkingBuffer[bufferBytesInUse],
               (unsigned char*)&inputData[i], inputDataSize - i);
    }
}

void MD5Context::end(char* outputDigest) {
    unsigned char digestInBytes[DIGEST_SIZE_IN_BYTES];
    finalize(digestInBytes);

    // Convert the digest from bytes (binary) to hex digits:
    static char const hex[] = "0123456789abcdef";
    unsigned i;
    for (i = 0; i < DIGEST_SIZE_IN_BYTES; ++i) {
        outputDigest[2 * i] = hex[digestInBytes[i] >> 4];
        outputDigest[2 * i + 1] = hex[digestInBytes[i] & 0x0F];
    }
    outputDigest[2 * i] = '\0';
}

// Routines that unpack 32 and 64-bit values into arrays of bytes (in
// little-endian order). (These are used to implement "finalize()".)

static void unpack32(unsigned char out[4], u_int32_t in) {
    for (unsigned i = 0; i < 4; ++i) {
        out[i] = (unsigned char)((in >> (8 * i)) & 0xFF);
    }
}

static void unpack64(unsigned char out[8], u_int64_t in) {
    for (unsigned i = 0; i < 8; ++i) {
        out[i] = (unsigned char)((in >> (8 * i)) & 0xFF);
    }
}

static unsigned char const PADDING[64] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

void MD5Context::finalize(unsigned char* outputDigestInBytes) {
    // Unpack our bit count:
    unsigned char bitCountInBytes[8];
    unpack64(bitCountInBytes, fBitCount);

    // Before 'finalizing', make sure that we transform any remaining bytes in
    // our working buffer:
    u_int64_t const byteCount = fBitCount >> 3;
    unsigned bufferBytesInUse = (unsigned)(byteCount & 0x3F);
    unsigned numPaddingBytes = (bufferBytesInUse < 56)
                                       ? (56 - bufferBytesInUse)
                                       : (64 + 56 - bufferBytesInUse);
    addData(PADDING, numPaddingBytes);

    addData(bitCountInBytes, 8);

    // Unpack our 'state' into the output digest:
    unpack32(&outputDigestInBytes[0], fState[0]);
    unpack32(&outputDigestInBytes[4], fState[1]);
    unpack32(&outputDigestInBytes[8], fState[2]);
    unpack32(&outputDigestInBytes[12], fState[3]);

    zeroize();
}

void MD5Context::zeroize() {
    fState[0] = fState[1] = fState[2] = fState[3] = 0;
    fBitCount = 0;
    for (unsigned i = 0; i < 64; ++i) fWorkingBuffer[i] = 0;
}

////////// Implementation of the MD5 transform
///("MD5Context::transform64Bytes()") //////////

// Constants for the transform:
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

// Basic MD5 functions:
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

// Rotate "x" left "n" bits:
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Other transforms:
#define FF(a, b, c, d, x, s, ac)                         \
    {                                                    \
        (a) += F((b), (c), (d)) + (x) + (u_int32_t)(ac); \
        (a) = ROTATE_LEFT((a), (s));                     \
        (a) += (b);                                      \
    }
#define GG(a, b, c, d, x, s, ac)                         \
    {                                                    \
        (a) += G((b), (c), (d)) + (x) + (u_int32_t)(ac); \
        (a) = ROTATE_LEFT((a), (s));                     \
        (a) += (b);                                      \
    }
#define HH(a, b, c, d, x, s, ac)                         \
    {                                                    \
        (a) += H((b), (c), (d)) + (x) + (u_int32_t)(ac); \
        (a) = ROTATE_LEFT((a), (s));                     \
        (a) += (b);                                      \
    }
#define II(a, b, c, d, x, s, ac)                         \
    {                                                    \
        (a) += I((b), (c), (d)) + (x) + (u_int32_t)(ac); \
        (a) = ROTATE_LEFT((a), (s));                     \
        (a) += (b);                                      \
    }

void MD5Context::transform64Bytes(unsigned char const block[64]) {
    u_int32_t a = fState[0], b = fState[1], c = fState[2], d = fState[3];

    // Begin by packing "block" into an array ("x") of 16 32-bit values (in
    // little-endian order):
    u_int32_t x[16];
    for (unsigned i = 0, j = 0; i < 16; ++i, j += 4) {
        x[i] = ((u_int32_t)block[j]) | (((u_int32_t)block[j + 1]) << 8) |
               (((u_int32_t)block[j + 2]) << 16) |
               (((u_int32_t)block[j + 3]) << 24);
    }

    // Now, perform the transform on the array "x":

    // Round 1
    FF(a, b, c, d, x[0], S11, 0xd76aa478);   // 1
    FF(d, a, b, c, x[1], S12, 0xe8c7b756);   // 2
    FF(c, d, a, b, x[2], S13, 0x242070db);   // 3
    FF(b, c, d, a, x[3], S14, 0xc1bdceee);   // 4
    FF(a, b, c, d, x[4], S11, 0xf57c0faf);   // 5
    FF(d, a, b, c, x[5], S12, 0x4787c62a);   // 6
    FF(c, d, a, b, x[6], S13, 0xa8304613);   // 7
    FF(b, c, d, a, x[7], S14, 0xfd469501);   // 8
    FF(a, b, c, d, x[8], S11, 0x698098d8);   // 9
    FF(d, a, b, c, x[9], S12, 0x8b44f7af);   // 10
    FF(c, d, a, b, x[10], S13, 0xffff5bb1);  // 11
    FF(b, c, d, a, x[11], S14, 0x895cd7be);  // 12
    FF(a, b, c, d, x[12], S11, 0x6b901122);  // 13
    FF(d, a, b, c, x[13], S12, 0xfd987193);  // 14
    FF(c, d, a, b, x[14], S13, 0xa679438e);  // 15
    FF(b, c, d, a, x[15], S14, 0x49b40821);  // 16

    // Round 2
    GG(a, b, c, d, x[1], S21, 0xf61e2562);   // 17
    GG(d, a, b, c, x[6], S22, 0xc040b340);   // 18
    GG(c, d, a, b, x[11], S23, 0x265e5a51);  // 19
    GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);   // 20
    GG(a, b, c, d, x[5], S21, 0xd62f105d);   // 21
    GG(d, a, b, c, x[10], S22, 0x2441453);   // 22
    GG(c, d, a, b, x[15], S23, 0xd8a1e681);  // 23
    GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);   // 24
    GG(a, b, c, d, x[9], S21, 0x21e1cde6);   // 25
    GG(d, a, b, c, x[14], S22, 0xc33707d6);  // 26
    GG(c, d, a, b, x[3], S23, 0xf4d50d87);   // 27
    GG(b, c, d, a, x[8], S24, 0x455a14ed);   // 28
    GG(a, b, c, d, x[13], S21, 0xa9e3e905);  // 29
    GG(d, a, b, c, x[2], S22, 0xfcefa3f8);   // 30
    GG(c, d, a, b, x[7], S23, 0x676f02d9);   // 31
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);  // 32

    // Round 3
    HH(a, b, c, d, x[5], S31, 0xfffa3942);   // 33
    HH(d, a, b, c, x[8], S32, 0x8771f681);   // 34
    HH(c, d, a, b, x[11], S33, 0x6d9d6122);  // 35
    HH(b, c, d, a, x[14], S34, 0xfde5380c);  // 36
    HH(a, b, c, d, x[1], S31, 0xa4beea44);   // 37
    HH(d, a, b, c, x[4], S32, 0x4bdecfa9);   // 38
    HH(c, d, a, b, x[7], S33, 0xf6bb4b60);   // 39
    HH(b, c, d, a, x[10], S34, 0xbebfbc70);  // 40
    HH(a, b, c, d, x[13], S31, 0x289b7ec6);  // 41
    HH(d, a, b, c, x[0], S32, 0xeaa127fa);   // 42
    HH(c, d, a, b, x[3], S33, 0xd4ef3085);   // 43
    HH(b, c, d, a, x[6], S34, 0x4881d05);    // 44
    HH(a, b, c, d, x[9], S31, 0xd9d4d039);   // 45
    HH(d, a, b, c, x[12], S32, 0xe6db99e5);  // 46
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8);  // 47
    HH(b, c, d, a, x[2], S34, 0xc4ac5665);   // 48

    // Round 4
    II(a, b, c, d, x[0], S41, 0xf4292244);   // 49
    II(d, a, b, c, x[7], S42, 0x432aff97);   // 50
    II(c, d, a, b, x[14], S43, 0xab9423a7);  // 51
    II(b, c, d, a, x[5], S44, 0xfc93a039);   // 52
    II(a, b, c, d, x[12], S41, 0x655b59c3);  // 53
    II(d, a, b, c, x[3], S42, 0x8f0ccc92);   // 54
    II(c, d, a, b, x[10], S43, 0xffeff47d);  // 55
    II(b, c, d, a, x[1], S44, 0x85845dd1);   // 56
    II(a, b, c, d, x[8], S41, 0x6fa87e4f);   // 57
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0);  // 58
    II(c, d, a, b, x[6], S43, 0xa3014314);   // 59
    II(b, c, d, a, x[13], S44, 0x4e0811a1);  // 60
    II(a, b, c, d, x[4], S41, 0xf7537e82);   // 61
    II(d, a, b, c, x[11], S42, 0xbd3af235);  // 62
    II(c, d, a, b, x[2], S43, 0x2ad7d2bb);   // 63
    II(b, c, d, a, x[9], S44, 0xeb86d391);   // 64

    fState[0] += a;
    fState[1] += b;
    fState[2] += c;
    fState[3] += d;

    // Zeroize sensitive information.
    for (unsigned k = 0; k < 16; ++k) x[k] = 0;
}
