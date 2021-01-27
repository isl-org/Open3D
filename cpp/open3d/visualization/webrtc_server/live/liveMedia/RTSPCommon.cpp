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
// Common routines used by both RTSP clients and servers
// Implementation

#include "RTSPCommon.hh"

#include <ctype.h>  // for "isxdigit()
#include <stdio.h>
#include <string.h>
#include <time.h>  // for "strftime()" and "gmtime()"

#include "Locale.hh"

static void decodeURL(char* url) {
    // Replace (in place) any %<hex><hex> sequences with the appropriate 8-bit
    // character.
    char* cursor = url;
    while (*cursor) {
        if ((cursor[0] == '%') && cursor[1] && isxdigit(cursor[1]) &&
            cursor[2] && isxdigit(cursor[2])) {
            // We saw a % followed by 2 hex digits, so we copy the literal hex
            // value into the URL, then advance the cursor past it:
            char hex[3];
            hex[0] = cursor[1];
            hex[1] = cursor[2];
            hex[2] = '\0';
            *url++ = (char)strtol(hex, NULL, 16);
            cursor += 3;
        } else {
            // Common case: This is a normal character or a bogus % expression,
            // so just copy it
            *url++ = *cursor++;
        }
    }

    *url = '\0';
}

Boolean parseRTSPRequestString(char const* reqStr,
                               unsigned reqStrSize,
                               char* resultCmdName,
                               unsigned resultCmdNameMaxSize,
                               char* resultURLPreSuffix,
                               unsigned resultURLPreSuffixMaxSize,
                               char* resultURLSuffix,
                               unsigned resultURLSuffixMaxSize,
                               char* resultCSeq,
                               unsigned resultCSeqMaxSize,
                               char* resultSessionIdStr,
                               unsigned resultSessionIdStrMaxSize,
                               unsigned& contentLength) {
    // This parser is currently rather dumb; it should be made smarter #####

    // "Be liberal in what you accept": Skip over any whitespace at the start of
    // the request:
    unsigned i;
    for (i = 0; i < reqStrSize; ++i) {
        char c = reqStr[i];
        if (!(c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\0'))
            break;
    }
    if (i == reqStrSize)
        return False;  // The request consisted of nothing but whitespace!

    // Then read everything up to the next space (or tab) as the command name:
    Boolean parseSucceeded = False;
    unsigned i1 = 0;
    for (; i1 < resultCmdNameMaxSize - 1 && i < reqStrSize; ++i, ++i1) {
        char c = reqStr[i];
        if (c == ' ' || c == '\t') {
            parseSucceeded = True;
            break;
        }

        resultCmdName[i1] = c;
    }
    resultCmdName[i1] = '\0';
    if (!parseSucceeded) return False;

    // Skip over the prefix of any "rtsp://" or "rtsp:/" URL that follows:
    unsigned j = i + 1;
    while (j < reqStrSize && (reqStr[j] == ' ' || reqStr[j] == '\t'))
        ++j;  // skip over any additional white space
    for (; (int)j < (int)(reqStrSize - 8); ++j) {
        if ((reqStr[j] == 'r' || reqStr[j] == 'R') &&
            (reqStr[j + 1] == 't' || reqStr[j + 1] == 'T') &&
            (reqStr[j + 2] == 's' || reqStr[j + 2] == 'S') &&
            (reqStr[j + 3] == 'p' || reqStr[j + 3] == 'P') &&
            reqStr[j + 4] == ':' && reqStr[j + 5] == '/') {
            j += 6;
            if (reqStr[j] == '/') {
                // This is a "rtsp://" URL; skip over the host:port part that
                // follows:
                ++j;
                while (j < reqStrSize && reqStr[j] != '/' && reqStr[j] != ' ')
                    ++j;
            } else {
                // This is a "rtsp:/" URL; back up to the "/":
                --j;
            }
            i = j;
            break;
        }
    }

    // Look for the URL suffix (before the following "RTSP/"):
    parseSucceeded = False;
    for (unsigned k = i + 1; (int)k < (int)(reqStrSize - 5); ++k) {
        if (reqStr[k] == 'R' && reqStr[k + 1] == 'T' && reqStr[k + 2] == 'S' &&
            reqStr[k + 3] == 'P' && reqStr[k + 4] == '/') {
            while (--k >= i && reqStr[k] == ' ') {
            }  // go back over all spaces before "RTSP/"
            unsigned k1 = k;
            while (k1 > i && reqStr[k1] != '/') --k1;

            // ASSERT: At this point
            //   i: first space or slash after "host" or "host:port"
            //   k: last non-space before "RTSP/"
            //   k1: last slash in the range [i,k]

            // The URL suffix comes from [k1+1,k]
            // Copy "resultURLSuffix":
            unsigned n = 0, k2 = k1 + 1;
            if (k2 <= k) {
                if (k - k1 + 1 > resultURLSuffixMaxSize)
                    return False;  // there's no room
                while (k2 <= k) resultURLSuffix[n++] = reqStr[k2++];
            }
            resultURLSuffix[n] = '\0';

            // The URL 'pre-suffix' comes from [i+1,k1-1]
            // Copy "resultURLPreSuffix":
            n = 0;
            k2 = i + 1;
            if (k2 + 1 <= k1) {
                if (k1 - i > resultURLPreSuffixMaxSize)
                    return False;  // there's no room
                while (k2 <= k1 - 1) resultURLPreSuffix[n++] = reqStr[k2++];
            }
            resultURLPreSuffix[n] = '\0';
            decodeURL(resultURLPreSuffix);

            i = k + 7;  // to go past " RTSP/"
            parseSucceeded = True;
            break;
        }
    }
    if (!parseSucceeded) return False;

    // Look for "CSeq:" (mandatory, case insensitive), skip whitespace,
    // then read everything up to the next \r or \n as 'CSeq':
    parseSucceeded = False;
    for (j = i; (int)j < (int)(reqStrSize - 5); ++j) {
        if (_strncasecmp("CSeq:", &reqStr[j], 5) == 0) {
            j += 5;
            while (j < reqStrSize && (reqStr[j] == ' ' || reqStr[j] == '\t'))
                ++j;
            unsigned n;
            for (n = 0; n < resultCSeqMaxSize - 1 && j < reqStrSize; ++n, ++j) {
                char c = reqStr[j];
                if (c == '\r' || c == '\n') {
                    parseSucceeded = True;
                    break;
                }

                resultCSeq[n] = c;
            }
            resultCSeq[n] = '\0';
            break;
        }
    }
    if (!parseSucceeded) return False;

    // Look for "Session:" (optional, case insensitive), skip whitespace,
    // then read everything up to the next \r or \n as 'Session':
    resultSessionIdStr[0] = '\0';  // default value (empty string)
    for (j = i; (int)j < (int)(reqStrSize - 8); ++j) {
        if (_strncasecmp("Session:", &reqStr[j], 8) == 0) {
            j += 8;
            while (j < reqStrSize && (reqStr[j] == ' ' || reqStr[j] == '\t'))
                ++j;
            unsigned n;
            for (n = 0; n < resultSessionIdStrMaxSize - 1 && j < reqStrSize;
                 ++n, ++j) {
                char c = reqStr[j];
                if (c == '\r' || c == '\n') {
                    break;
                }

                resultSessionIdStr[n] = c;
            }
            resultSessionIdStr[n] = '\0';
            break;
        }
    }

    // Also: Look for "Content-Length:" (optional, case insensitive)
    contentLength = 0;  // default value
    for (j = i; (int)j < (int)(reqStrSize - 15); ++j) {
        if (_strncasecmp("Content-Length:", &(reqStr[j]), 15) == 0) {
            j += 15;
            while (j < reqStrSize && (reqStr[j] == ' ' || reqStr[j] == '\t'))
                ++j;
            unsigned num;
            if (sscanf(&reqStr[j], "%u", &num) == 1) {
                contentLength = num;
            }
        }
    }
    return True;
}

Boolean parseRangeParam(char const* paramStr,
                        double& rangeStart,
                        double& rangeEnd,
                        char*& absStartTime,
                        char*& absEndTime,
                        Boolean& startTimeIsNow) {
    delete[] absStartTime;
    delete[] absEndTime;
    absStartTime = absEndTime =
            NULL;  // by default, unless "paramStr" is a "clock=..." string
    startTimeIsNow = False;  // by default
    double start, end;
    int numCharsMatched1 = 0, numCharsMatched2 = 0, numCharsMatched3 = 0,
        numCharsMatched4 = 0;
    Locale l("C", Numeric);
    if (sscanf(paramStr, "npt = %lf - %lf", &start, &end) == 2) {
        rangeStart = start;
        rangeEnd = end;
    } else if (sscanf(paramStr, "npt = %n%lf -", &numCharsMatched1, &start) ==
               1) {
        if (paramStr[numCharsMatched1] == '-') {
            // special case for "npt = -<endtime>", which matches here:
            rangeStart = 0.0;
            startTimeIsNow = True;
            rangeEnd = -start;
        } else {
            rangeStart = start;
            rangeEnd = 0.0;
        }
    } else if (sscanf(paramStr, "npt = now - %lf", &end) == 1) {
        rangeStart = 0.0;
        startTimeIsNow = True;
        rangeEnd = end;
    } else if (sscanf(paramStr, "npt = now -%n", &numCharsMatched2) == 0 &&
               numCharsMatched2 > 0) {
        rangeStart = 0.0;
        startTimeIsNow = True;
        rangeEnd = 0.0;
    } else if (sscanf(paramStr, "clock = %n", &numCharsMatched3) == 0 &&
               numCharsMatched3 > 0) {
        rangeStart = rangeEnd = 0.0;

        char const* utcTimes = &paramStr[numCharsMatched3];
        size_t len = strlen(utcTimes) + 1;
        char* as = new char[len];
        char* ae = new char[len];
        int sscanfResult = sscanf(utcTimes, "%[^-]-%[^\r\n]", as, ae);
        if (sscanfResult == 2) {
            absStartTime = as;
            absEndTime = ae;
        } else if (sscanfResult == 1) {
            absStartTime = as;
            delete[] ae;
        } else {
            delete[] as;
            delete[] ae;
            return False;
        }
    } else if (sscanf(paramStr, "smtpe = %n", &numCharsMatched4) == 0 &&
               numCharsMatched4 > 0) {
        // We accept "smtpe=" parameters, but currently do not interpret them.
    } else {
        return False;  // The header is malformed
    }

    return True;
}

Boolean parseRangeHeader(char const* buf,
                         double& rangeStart,
                         double& rangeEnd,
                         char*& absStartTime,
                         char*& absEndTime,
                         Boolean& startTimeIsNow) {
    // First, find "Range:"
    while (1) {
        if (*buf == '\0') return False;  // not found
        if (_strncasecmp(buf, "Range: ", 7) == 0) break;
        ++buf;
    }

    char const* fields = buf + 7;
    while (*fields == ' ') ++fields;
    return parseRangeParam(fields, rangeStart, rangeEnd, absStartTime,
                           absEndTime, startTimeIsNow);
}

Boolean parseScaleHeader(char const* buf, float& scale) {
    // Initialize the result parameter to a default value:
    scale = 1.0;

    // First, find "Scale:"
    while (1) {
        if (*buf == '\0') return False;  // not found
        if (_strncasecmp(buf, "Scale:", 6) == 0) break;
        ++buf;
    }

    char const* fields = buf + 6;
    while (*fields == ' ') ++fields;
    float sc;
    if (sscanf(fields, "%f", &sc) == 1) {
        scale = sc;
    } else {
        return False;  // The header is malformed
    }

    return True;
}

// Used to implement "RTSPOptionIsSupported()":
static Boolean isSeparator(char c) {
    return c == ' ' || c == ',' || c == ';' || c == ':';
}

Boolean RTSPOptionIsSupported(char const* commandName,
                              char const* optionsResponseString) {
    do {
        if (commandName == NULL || optionsResponseString == NULL) break;

        unsigned const commandNameLen = strlen(commandName);
        if (commandNameLen == 0) break;

        // "optionsResponseString" is assumed to be a list of command names,
        // separated by " " and/or ",", ";", or ":" Scan through these, looking
        // for "commandName".
        while (1) {
            // Skip over separators:
            while (*optionsResponseString != '\0' &&
                   isSeparator(*optionsResponseString))
                ++optionsResponseString;
            if (*optionsResponseString == '\0') break;

            // At this point, "optionsResponseString" begins with a command name
            // (with perhaps a separator afterwads).
            if (strncmp(commandName, optionsResponseString, commandNameLen) ==
                0) {
                // We have at least a partial match here.
                optionsResponseString += commandNameLen;
                if (*optionsResponseString == '\0' ||
                    isSeparator(*optionsResponseString))
                    return True;
            }

            // No match.  Skip over the rest of the command name:
            while (*optionsResponseString != '\0' &&
                   !isSeparator(*optionsResponseString))
                ++optionsResponseString;
        }
    } while (0);

    return False;
}

char const* dateHeader() {
    static char buf[200];
#if !defined(_WIN32_WCE)
    time_t tt = time(NULL);
    strftime(buf, sizeof buf, "Date: %a, %b %d %Y %H:%M:%S GMT\r\n",
             gmtime(&tt));
#else
    // WinCE apparently doesn't have "time()", "strftime()", or "gmtime()",
    // so generate the "Date:" header a different, WinCE-specific way.
    // (Thanks to Pierre l'Hussiez for this code)
    // RSF: But where is the "Date: " string?  This code doesn't look quite
    // right...
    SYSTEMTIME SystemTime;
    GetSystemTime(&SystemTime);
    WCHAR dateFormat[] = L"ddd, MMM dd yyyy";
    WCHAR timeFormat[] = L"HH:mm:ss GMT\r\n";
    WCHAR inBuf[200];
    DWORD locale = LOCALE_NEUTRAL;

    int ret = GetDateFormat(locale, 0, &SystemTime, (LPTSTR)dateFormat,
                            (LPTSTR)inBuf, sizeof inBuf);
    inBuf[ret - 1] = ' ';
    ret = GetTimeFormat(locale, 0, &SystemTime, (LPTSTR)timeFormat,
                        (LPTSTR)inBuf + ret, (sizeof inBuf) - ret);
    wcstombs(buf, inBuf, wcslen(inBuf));
#endif
    return buf;
}
