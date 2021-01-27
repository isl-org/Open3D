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
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// Basic Usage Environment: for a simple, non-scripted, console application
// Implementation

#include "BasicUsageEnvironment0.hh"

#include <stdio.h>
#if defined(__WIN32__) || defined(_WIN32) || defined(_WIN32_WCE)
#define snprintf _snprintf
#endif

////////// BasicUsageEnvironment //////////

BasicUsageEnvironment0::BasicUsageEnvironment0(TaskScheduler& taskScheduler)
    : UsageEnvironment(taskScheduler), fBufferMaxSize(RESULT_MSG_BUFFER_MAX) {
    reset();
}

BasicUsageEnvironment0::~BasicUsageEnvironment0() {}

void BasicUsageEnvironment0::reset() {
    fCurBufferSize = 0;
    fResultMsgBuffer[fCurBufferSize] = '\0';
}

// Implementation of virtual functions:

char const* BasicUsageEnvironment0::getResultMsg() const {
    return fResultMsgBuffer;
}

void BasicUsageEnvironment0::setResultMsg(MsgString msg) {
    reset();
    appendToResultMsg(msg);
}

void BasicUsageEnvironment0::setResultMsg(MsgString msg1, MsgString msg2) {
    setResultMsg(msg1);
    appendToResultMsg(msg2);
}

void BasicUsageEnvironment0::setResultMsg(MsgString msg1,
                                          MsgString msg2,
                                          MsgString msg3) {
    setResultMsg(msg1, msg2);
    appendToResultMsg(msg3);
}

void BasicUsageEnvironment0::setResultErrMsg(MsgString msg, int err) {
    setResultMsg(msg);

    if (err == 0) err = getErrno();
#if defined(__WIN32__) || defined(_WIN32) || defined(_WIN32_WCE)
#ifndef _UNICODE
    char errMsg[RESULT_MSG_BUFFER_MAX] = "\0";
    if (0 != FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, 0, errMsg,
                            sizeof(errMsg) / sizeof(errMsg[0]), NULL)) {
        // Remove all trailing '\r', '\n' and '.'
        for (char* p = errMsg + strlen(errMsg);
             p != errMsg &&
             (*p == '\r' || *p == '\n' || *p == '.' || *p == '\0');
             --p) {
            *p = '\0';
        }
    } else
        snprintf(errMsg, sizeof(errMsg) / sizeof(errMsg[0]), "error %d", err);
    appendToResultMsg(errMsg);
#endif
#else
    appendToResultMsg(strerror(err));
#endif
}

void BasicUsageEnvironment0::appendToResultMsg(MsgString msg) {
    char* curPtr = &fResultMsgBuffer[fCurBufferSize];
    unsigned spaceAvailable = fBufferMaxSize - fCurBufferSize;
    unsigned msgLength = strlen(msg);

    // Copy only enough of "msg" as will fit:
    if (msgLength > spaceAvailable - 1) {
        msgLength = spaceAvailable - 1;
    }

    memmove(curPtr, (char*)msg, msgLength);
    fCurBufferSize += msgLength;
    fResultMsgBuffer[fCurBufferSize] = '\0';
}

void BasicUsageEnvironment0::reportBackgroundError() {
    fputs(getResultMsg(), stderr);
}
