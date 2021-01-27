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
// A filter that breaks up a H.264 Video Elementary Stream into NAL units.
// Implementation

#include "H264VideoStreamFramer.hh"

H264VideoStreamFramer* H264VideoStreamFramer ::createNew(
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters) {
    return new H264VideoStreamFramer(env, inputSource, True,
                                     includeStartCodeInOutput,
                                     insertAccessUnitDelimiters);
}

H264VideoStreamFramer ::H264VideoStreamFramer(
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean createParser,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters)
    : H264or5VideoStreamFramer(264,
                               env,
                               inputSource,
                               createParser,
                               includeStartCodeInOutput,
                               insertAccessUnitDelimiters) {}

H264VideoStreamFramer::~H264VideoStreamFramer() {}

Boolean H264VideoStreamFramer::isH264VideoStreamFramer() const { return True; }
