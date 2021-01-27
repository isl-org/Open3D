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
// Copyright (c) 1996-2020, Live Networks, Inc.  All rights reserved
// "Group Endpoint Id"
// Implementation

#include "GroupEId.hh"

GroupEId::GroupEId(struct in_addr const& groupAddr,
                   portNumBits portNum,
                   u_int8_t ttl) {
    struct in_addr sourceFilterAddr;
    sourceFilterAddr.s_addr = ~0;  // indicates no source filter

    init(groupAddr, sourceFilterAddr, portNum, ttl);
}

GroupEId::GroupEId(struct in_addr const& groupAddr,
                   struct in_addr const& sourceFilterAddr,
                   portNumBits portNum) {
    init(groupAddr, sourceFilterAddr, portNum, 255);
}

Boolean GroupEId::isSSM() const {
    return fSourceFilterAddress.s_addr != netAddressBits(~0);
}

void GroupEId::init(struct in_addr const& groupAddr,
                    struct in_addr const& sourceFilterAddr,
                    portNumBits portNum,
                    u_int8_t ttl) {
    fGroupAddress = groupAddr;
    fSourceFilterAddress = sourceFilterAddr;
    fPortNum = portNum;
    fTTL = ttl;
}
