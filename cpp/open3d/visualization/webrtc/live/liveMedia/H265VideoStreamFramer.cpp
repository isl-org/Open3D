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
// A filter that breaks up a H.265 Video Elementary Stream into NAL units.
// Implementation

#include "H265VideoStreamFramer.hh"

H265VideoStreamFramer* H265VideoStreamFramer ::createNew(
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters) {
    return new H265VideoStreamFramer(env, inputSource, True,
                                     includeStartCodeInOutput,
                                     insertAccessUnitDelimiters);
}

H265VideoStreamFramer ::H265VideoStreamFramer(
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean createParser,
        Boolean includeStartCodeInOutput,
        Boolean insertAccessUnitDelimiters)
    : H264or5VideoStreamFramer(265,
                               env,
                               inputSource,
                               createParser,
                               includeStartCodeInOutput,
                               insertAccessUnitDelimiters) {}

H265VideoStreamFramer::~H265VideoStreamFramer() {}

Boolean H265VideoStreamFramer::isH265VideoStreamFramer() const { return True; }
