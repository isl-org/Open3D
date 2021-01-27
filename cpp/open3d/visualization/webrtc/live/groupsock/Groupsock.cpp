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
// 'Group sockets'
// Implementation

#include "Groupsock.hh"

#include "GroupsockHelper.hh"
//##### Eventually fix the following #include; we shouldn't know about tunnels
#include "TunnelEncaps.hh"

#ifndef NO_SSTREAM
#include <sstream>
#endif
#include <stdio.h>

///////// OutputSocket //////////

OutputSocket::OutputSocket(UsageEnvironment& env)
    : Socket(env, 0 /* let kernel choose port */),
      fSourcePort(0),
      fLastSentTTL(256 /*hack: a deliberately invalid value*/) {}

OutputSocket::OutputSocket(UsageEnvironment& env, Port port)
    : Socket(env, port),
      fSourcePort(0),
      fLastSentTTL(256 /*hack: a deliberately invalid value*/) {}

OutputSocket::~OutputSocket() {}

Boolean OutputSocket::write(netAddressBits address,
                            portNumBits portNum,
                            u_int8_t ttl,
                            unsigned char* buffer,
                            unsigned bufferSize) {
    struct in_addr destAddr;
    destAddr.s_addr = address;
    if ((unsigned)ttl == fLastSentTTL) {
        // Optimization: Don't do a 'set TTL' system call again
        if (!writeSocket(env(), socketNum(), destAddr, portNum, buffer,
                         bufferSize))
            return False;
    } else {
        if (!writeSocket(env(), socketNum(), destAddr, portNum, ttl, buffer,
                         bufferSize))
            return False;
        fLastSentTTL = (unsigned)ttl;
    }

    if (sourcePortNum() == 0) {
        // Now that we've sent a packet, we can find out what the
        // kernel chose as our ephemeral source port number:
        if (!getSourcePort(env(), socketNum(), fSourcePort)) {
            if (DebugLevel >= 1)
                env() << *this
                      << ": failed to get source port: " << env().getResultMsg()
                      << "\n";
            return False;
        }
    }

    return True;
}

// By default, we don't do reads:
Boolean OutputSocket ::handleRead(unsigned char* /*buffer*/,
                                  unsigned /*bufferMaxSize*/,
                                  unsigned& /*bytesRead*/,
                                  struct sockaddr_in& /*fromAddressAndPort*/) {
    return True;
}

///////// destRecord //////////

destRecord ::destRecord(struct in_addr const& addr,
                        Port const& port,
                        u_int8_t ttl,
                        unsigned sessionId,
                        destRecord* next)
    : fNext(next), fGroupEId(addr, port.num(), ttl), fSessionId(sessionId) {}

destRecord::~destRecord() { delete fNext; }

///////// Groupsock //////////

NetInterfaceTrafficStats Groupsock::statsIncoming;
NetInterfaceTrafficStats Groupsock::statsOutgoing;
NetInterfaceTrafficStats Groupsock::statsRelayedIncoming;
NetInterfaceTrafficStats Groupsock::statsRelayedOutgoing;

// Constructor for a source-independent multicast group
Groupsock::Groupsock(UsageEnvironment& env,
                     struct in_addr const& groupAddr,
                     Port port,
                     u_int8_t ttl)
    : OutputSocket(env, port),
      deleteIfNoMembers(False),
      isSlave(False),
      fDests(new destRecord(groupAddr, port, ttl, 0, NULL)),
      fIncomingGroupEId(groupAddr, port.num(), ttl) {
    if (!socketJoinGroup(env, socketNum(), groupAddr.s_addr)) {
        if (DebugLevel >= 1) {
            env << *this << ": failed to join group: " << env.getResultMsg()
                << "\n";
        }
    }

    // Make sure we can get our source address:
    if (ourIPAddress(env) == 0) {
        if (DebugLevel >= 0) {  // this is a fatal error
            env << "Unable to determine our source address: "
                << env.getResultMsg() << "\n";
        }
    }

    if (DebugLevel >= 2) env << *this << ": created\n";
}

// Constructor for a source-specific multicast group
Groupsock::Groupsock(UsageEnvironment& env,
                     struct in_addr const& groupAddr,
                     struct in_addr const& sourceFilterAddr,
                     Port port)
    : OutputSocket(env, port),
      deleteIfNoMembers(False),
      isSlave(False),
      fDests(new destRecord(groupAddr, port, 255, 0, NULL)),
      fIncomingGroupEId(groupAddr, sourceFilterAddr, port.num()) {
    // First try a SSM join.  If that fails, try a regular join:
    if (!socketJoinGroupSSM(env, socketNum(), groupAddr.s_addr,
                            sourceFilterAddr.s_addr)) {
        if (DebugLevel >= 3) {
            env << *this << ": SSM join failed: " << env.getResultMsg();
            env << " - trying regular join instead\n";
        }
        if (!socketJoinGroup(env, socketNum(), groupAddr.s_addr)) {
            if (DebugLevel >= 1) {
                env << *this << ": failed to join group: " << env.getResultMsg()
                    << "\n";
            }
        }
    }

    if (DebugLevel >= 2) env << *this << ": created\n";
}

Groupsock::~Groupsock() {
    if (isSSM()) {
        if (!socketLeaveGroupSSM(env(), socketNum(), groupAddress().s_addr,
                                 sourceFilterAddress().s_addr)) {
            socketLeaveGroup(env(), socketNum(), groupAddress().s_addr);
        }
    } else {
        socketLeaveGroup(env(), socketNum(), groupAddress().s_addr);
    }

    delete fDests;

    if (DebugLevel >= 2) env() << *this << ": deleting\n";
}

destRecord* Groupsock ::createNewDestRecord(struct in_addr const& addr,
                                            Port const& port,
                                            u_int8_t ttl,
                                            unsigned sessionId,
                                            destRecord* next) {
    // Default implementation:
    return new destRecord(addr, port, ttl, sessionId, next);
}

void Groupsock::changeDestinationParameters(struct in_addr const& newDestAddr,
                                            Port newDestPort,
                                            int newDestTTL,
                                            unsigned sessionId) {
    destRecord* dest;
    for (dest = fDests; dest != NULL && dest->fSessionId != sessionId;
         dest = dest->fNext) {
    }

    if (dest == NULL) {  // There's no existing 'destRecord' for this
                         // "sessionId"; add a new one:
        fDests = createNewDestRecord(newDestAddr, newDestPort, newDestTTL,
                                     sessionId, fDests);
        return;
    }

    // "dest" is an existing 'destRecord' for this "sessionId"; change its
    // values to the new ones:
    struct in_addr destAddr = dest->fGroupEId.groupAddress();
    if (newDestAddr.s_addr != 0) {
        if (newDestAddr.s_addr != destAddr.s_addr &&
            IsMulticastAddress(newDestAddr.s_addr)) {
            // If the new destination is a multicast address, then we assume
            // that we want to join it also.  (If this is not in fact the case,
            // then call "multicastSendOnly()" afterwards.)
            socketLeaveGroup(env(), socketNum(), destAddr.s_addr);
            socketJoinGroup(env(), socketNum(), newDestAddr.s_addr);
        }
        destAddr.s_addr = newDestAddr.s_addr;
    }

    portNumBits destPortNum = dest->fGroupEId.portNum();
    if (newDestPort.num() != 0) {
        if (newDestPort.num() != destPortNum &&
            IsMulticastAddress(destAddr.s_addr)) {
            // Also bind to the new port number:
            changePort(newDestPort);
            // And rejoin the multicast group:
            socketJoinGroup(env(), socketNum(), destAddr.s_addr);
        }
        destPortNum = newDestPort.num();
    }

    u_int8_t destTTL = ttl();
    if (newDestTTL != ~0) destTTL = (u_int8_t)newDestTTL;

    dest->fGroupEId = GroupEId(destAddr, destPortNum, destTTL);

    // Finally, remove any other 'destRecord's that might also have this
    // "sessionId":
    removeDestinationFrom(dest->fNext, sessionId);
}

unsigned Groupsock ::lookupSessionIdFromDestination(
        struct sockaddr_in const& destAddrAndPort) const {
    destRecord* dest = lookupDestRecordFromDestination(destAddrAndPort);
    if (dest == NULL) return 0;

    return dest->fSessionId;
}

void Groupsock::addDestination(struct in_addr const& addr,
                               Port const& port,
                               unsigned sessionId) {
    // Default implementation:
    // If there's no existing 'destRecord' with the same "addr", "port", and
    // "sessionId", add a new one:
    for (destRecord* dest = fDests; dest != NULL; dest = dest->fNext) {
        if (sessionId == dest->fSessionId &&
            addr.s_addr == dest->fGroupEId.groupAddress().s_addr &&
            port.num() == dest->fGroupEId.portNum()) {
            return;
        }
    }

    fDests = createNewDestRecord(addr, port, 255, sessionId, fDests);
}

void Groupsock::removeDestination(unsigned sessionId) {
    // Default implementation:
    removeDestinationFrom(fDests, sessionId);
}

void Groupsock::removeAllDestinations() {
    delete fDests;
    fDests = NULL;
}

void Groupsock::multicastSendOnly() {
    // We disable this code for now, because - on some systems - leaving the
    // multicast group seems to cause sent packets to not be received by other
    // applications (at least, on the same host).
#if 0
  socketLeaveGroup(env(), socketNum(), fIncomingGroupEId.groupAddress().s_addr);
  for (destRecord* dests = fDests; dests != NULL; dests = dests->fNext) {
    socketLeaveGroup(env(), socketNum(), dests->fGroupEId.groupAddress().s_addr);
  }
#endif
}

Boolean Groupsock::output(UsageEnvironment& env,
                          unsigned char* buffer,
                          unsigned bufferSize,
                          DirectedNetInterface* interfaceNotToFwdBackTo) {
    do {
        // First, do the datagram send, to each destination:
        Boolean writeSuccess = True;
        for (destRecord* dests = fDests; dests != NULL; dests = dests->fNext) {
            if (!write(dests->fGroupEId.groupAddress().s_addr,
                       dests->fGroupEId.portNum(), dests->fGroupEId.ttl(),
                       buffer, bufferSize)) {
                writeSuccess = False;
                break;
            }
        }
        if (!writeSuccess) break;
        statsOutgoing.countPacket(bufferSize);
        statsGroupOutgoing.countPacket(bufferSize);

        // Then, forward to our members:
        int numMembers = 0;
        if (!members().IsEmpty()) {
            numMembers = outputToAllMembersExcept(interfaceNotToFwdBackTo,
                                                  ttl(), buffer, bufferSize,
                                                  ourIPAddress(env));
            if (numMembers < 0) break;
        }

        if (DebugLevel >= 3) {
            env << *this << ": wrote " << bufferSize << " bytes, ttl "
                << (unsigned)ttl();
            if (numMembers > 0) {
                env << "; relayed to " << numMembers << " members";
            }
            env << "\n";
        }
        return True;
    } while (0);

    if (DebugLevel >= 0) {  // this is a fatal error
        UsageEnvironment::MsgString msg = strDup(env.getResultMsg());
        env.setResultMsg("Groupsock write failed: ", msg);
        delete[](char*) msg;
    }
    return False;
}

Boolean Groupsock::handleRead(unsigned char* buffer,
                              unsigned bufferMaxSize,
                              unsigned& bytesRead,
                              struct sockaddr_in& fromAddressAndPort) {
    // Read data from the socket, and relay it across any attached tunnels
    //##### later make this code more general - independent of tunnels

    bytesRead = 0;

    int maxBytesToRead = bufferMaxSize - TunnelEncapsulationTrailerMaxSize;
    int numBytes = readSocket(env(), socketNum(), buffer, maxBytesToRead,
                              fromAddressAndPort);
    if (numBytes < 0) {
        if (DebugLevel >= 0) {  // this is a fatal error
            UsageEnvironment::MsgString msg = strDup(env().getResultMsg());
            env().setResultMsg("Groupsock read failed: ", msg);
            delete[](char*) msg;
        }
        return False;
    }

    // If we're a SSM group, make sure the source address matches:
    if (isSSM() &&
        fromAddressAndPort.sin_addr.s_addr != sourceFilterAddress().s_addr) {
        return True;
    }

    // We'll handle this data.
    // Also write it (with the encapsulation trailer) to each member,
    // unless the packet was originally sent by us to begin with.
    bytesRead = numBytes;

    int numMembers = 0;
    if (!wasLoopedBackFromUs(env(), fromAddressAndPort)) {
        statsIncoming.countPacket(numBytes);
        statsGroupIncoming.countPacket(numBytes);
        numMembers =
                outputToAllMembersExcept(NULL, ttl(), buffer, bytesRead,
                                         fromAddressAndPort.sin_addr.s_addr);
        if (numMembers > 0) {
            statsRelayedIncoming.countPacket(numBytes);
            statsGroupRelayedIncoming.countPacket(numBytes);
        }
    }
    if (DebugLevel >= 3) {
        env() << *this << ": read " << bytesRead << " bytes from "
              << AddressString(fromAddressAndPort).val() << ", port "
              << ntohs(fromAddressAndPort.sin_port);
        if (numMembers > 0) {
            env() << "; relayed to " << numMembers << " members";
        }
        env() << "\n";
    }

    return True;
}

Boolean Groupsock::wasLoopedBackFromUs(UsageEnvironment& env,
                                       struct sockaddr_in& fromAddressAndPort) {
    if (fromAddressAndPort.sin_addr.s_addr == ourIPAddress(env) ||
        fromAddressAndPort.sin_addr.s_addr == 0x7F000001 /*127.0.0.1*/) {
        if (fromAddressAndPort.sin_port == sourcePortNum()) {
#ifdef DEBUG_LOOPBACK_CHECKING
            if (DebugLevel >= 3) {
                env() << *this << ": got looped-back packet\n";
            }
#endif
            return True;
        }
    }

    return False;
}

destRecord* Groupsock ::lookupDestRecordFromDestination(
        struct sockaddr_in const& destAddrAndPort) const {
    for (destRecord* dest = fDests; dest != NULL; dest = dest->fNext) {
        if (destAddrAndPort.sin_addr.s_addr ==
                    dest->fGroupEId.groupAddress().s_addr &&
            destAddrAndPort.sin_port == dest->fGroupEId.portNum()) {
            return dest;
        }
    }
    return NULL;
}

void Groupsock::removeDestinationFrom(destRecord*& dests, unsigned sessionId) {
    destRecord** destsPtr = &dests;
    while (*destsPtr != NULL) {
        if (sessionId == (*destsPtr)->fSessionId) {
            // Remove the record pointed to by *destsPtr :
            destRecord* next = (*destsPtr)->fNext;
            (*destsPtr)->fNext = NULL;
            delete (*destsPtr);
            *destsPtr = next;
        } else {
            destsPtr = &((*destsPtr)->fNext);
        }
    }
}

int Groupsock::outputToAllMembersExcept(DirectedNetInterface* exceptInterface,
                                        u_int8_t ttlToFwd,
                                        unsigned char* data,
                                        unsigned size,
                                        netAddressBits sourceAddr) {
    // Don't forward TTL-0 packets
    if (ttlToFwd == 0) return 0;

    DirectedNetInterfaceSet::Iterator iter(members());
    unsigned numMembers = 0;
    DirectedNetInterface* interf;
    while ((interf = iter.next()) != NULL) {
        // Check whether we've asked to exclude this interface:
        if (interf == exceptInterface) continue;

        // Check that the packet's source address makes it OK to
        // be relayed across this interface:
        UsageEnvironment& saveEnv = env();
        // because the following call may delete "this"
        if (!interf->SourceAddrOKForRelaying(saveEnv, sourceAddr)) {
            if (strcmp(saveEnv.getResultMsg(), "") != 0) {
                // Treat this as a fatal error
                return -1;
            } else {
                continue;
            }
        }

        if (numMembers == 0) {
            // We know that we're going to forward to at least one
            // member, so fill in the tunnel encapsulation trailer.
            // (Note: Allow for it not being 4-byte-aligned.)
            TunnelEncapsulationTrailer* trailerInPacket =
                    (TunnelEncapsulationTrailer*)&data[size];
            TunnelEncapsulationTrailer* trailer;

            Boolean misaligned = ((uintptr_t)trailerInPacket & 3) != 0;
            unsigned trailerOffset;
            u_int8_t tunnelCmd;
            if (isSSM()) {
                // add an 'auxilliary address' before the trailer
                trailerOffset = TunnelEncapsulationTrailerAuxSize;
                tunnelCmd = TunnelDataAuxCmd;
            } else {
                trailerOffset = 0;
                tunnelCmd = TunnelDataCmd;
            }
            unsigned trailerSize =
                    TunnelEncapsulationTrailerSize + trailerOffset;
            unsigned tmpTr[TunnelEncapsulationTrailerMaxSize];
            if (misaligned) {
                trailer = (TunnelEncapsulationTrailer*)&tmpTr;
            } else {
                trailer = trailerInPacket;
            }
            trailer += trailerOffset;

            if (fDests != NULL) {
                trailer->address() = fDests->fGroupEId.groupAddress().s_addr;
                Port destPort(ntohs(fDests->fGroupEId.portNum()));
                trailer->port() = destPort;  // structure copy
            }
            trailer->ttl() = ttlToFwd;
            trailer->command() = tunnelCmd;

            if (isSSM()) {
                trailer->auxAddress() = sourceFilterAddress().s_addr;
            }

            if (misaligned) {
                memmove(trailerInPacket, trailer - trailerOffset, trailerSize);
            }

            size += trailerSize;
        }

        interf->write(data, size);
        ++numMembers;
    }

    return numMembers;
}

UsageEnvironment& operator<<(UsageEnvironment& s, const Groupsock& g) {
    UsageEnvironment& s1 = s << timestampString() << " Groupsock("
                             << g.socketNum() << ": "
                             << AddressString(g.groupAddress()).val() << ", "
                             << g.port() << ", ";
    if (g.isSSM()) {
        return s1 << "SSM source: "
                  << AddressString(g.sourceFilterAddress()).val() << ")";
    } else {
        return s1 << (unsigned)(g.ttl()) << ")";
    }
}

////////// GroupsockLookupTable //////////

// A hash table used to index Groupsocks by socket number.

static HashTable*& getSocketTable(UsageEnvironment& env) {
    _groupsockPriv* priv = groupsockPriv(env);
    if (priv->socketTable == NULL) {  // We need to create it
        priv->socketTable = HashTable::create(ONE_WORD_HASH_KEYS);
    }
    return priv->socketTable;
}

static Boolean unsetGroupsockBySocket(Groupsock const* groupsock) {
    do {
        if (groupsock == NULL) break;

        int sock = groupsock->socketNum();
        // Make sure "sock" is in bounds:
        if (sock < 0) break;

        HashTable*& sockets = getSocketTable(groupsock->env());

        Groupsock* gs = (Groupsock*)sockets->Lookup((char*)(long)sock);
        if (gs == NULL || gs != groupsock) break;
        sockets->Remove((char*)(long)sock);

        if (sockets->IsEmpty()) {
            // We can also delete the table (to reclaim space):
            delete sockets;
            sockets = NULL;
            reclaimGroupsockPriv(gs->env());
        }

        return True;
    } while (0);

    return False;
}

static Boolean setGroupsockBySocket(UsageEnvironment& env,
                                    int sock,
                                    Groupsock* groupsock) {
    do {
        // Make sure the "sock" parameter is in bounds:
        if (sock < 0) {
            char buf[100];
            sprintf(buf, "trying to use bad socket (%d)", sock);
            env.setResultMsg(buf);
            break;
        }

        HashTable* sockets = getSocketTable(env);

        // Make sure we're not replacing an existing Groupsock (although that
        // shouldn't happen)
        Boolean alreadyExists = (sockets->Lookup((char*)(long)sock) != 0);
        if (alreadyExists) {
            char buf[100];
            sprintf(buf, "Attempting to replace an existing socket (%d)", sock);
            env.setResultMsg(buf);
            break;
        }

        sockets->Add((char*)(long)sock, groupsock);
        return True;
    } while (0);

    return False;
}

static Groupsock* getGroupsockBySocket(UsageEnvironment& env, int sock) {
    do {
        // Make sure the "sock" parameter is in bounds:
        if (sock < 0) break;

        HashTable* sockets = getSocketTable(env);
        return (Groupsock*)sockets->Lookup((char*)(long)sock);
    } while (0);

    return NULL;
}

Groupsock* GroupsockLookupTable::Fetch(UsageEnvironment& env,
                                       netAddressBits groupAddress,
                                       Port port,
                                       u_int8_t ttl,
                                       Boolean& isNew) {
    isNew = False;
    Groupsock* groupsock;
    do {
        groupsock = (Groupsock*)fTable.Lookup(groupAddress, (~0), port);
        if (groupsock == NULL) {  // we need to create one:
            groupsock = AddNew(env, groupAddress, (~0), port, ttl);
            if (groupsock == NULL) break;
            isNew = True;
        }
    } while (0);

    return groupsock;
}

Groupsock* GroupsockLookupTable::Fetch(UsageEnvironment& env,
                                       netAddressBits groupAddress,
                                       netAddressBits sourceFilterAddr,
                                       Port port,
                                       Boolean& isNew) {
    isNew = False;
    Groupsock* groupsock;
    do {
        groupsock =
                (Groupsock*)fTable.Lookup(groupAddress, sourceFilterAddr, port);
        if (groupsock == NULL) {  // we need to create one:
            groupsock = AddNew(env, groupAddress, sourceFilterAddr, port, 0);
            if (groupsock == NULL) break;
            isNew = True;
        }
    } while (0);

    return groupsock;
}

Groupsock* GroupsockLookupTable::Lookup(netAddressBits groupAddress,
                                        Port port) {
    return (Groupsock*)fTable.Lookup(groupAddress, (~0), port);
}

Groupsock* GroupsockLookupTable::Lookup(netAddressBits groupAddress,
                                        netAddressBits sourceFilterAddr,
                                        Port port) {
    return (Groupsock*)fTable.Lookup(groupAddress, sourceFilterAddr, port);
}

Groupsock* GroupsockLookupTable::Lookup(UsageEnvironment& env, int sock) {
    return getGroupsockBySocket(env, sock);
}

Boolean GroupsockLookupTable::Remove(Groupsock const* groupsock) {
    unsetGroupsockBySocket(groupsock);
    return fTable.Remove(groupsock->groupAddress().s_addr,
                         groupsock->sourceFilterAddress().s_addr,
                         groupsock->port());
}

Groupsock* GroupsockLookupTable::AddNew(UsageEnvironment& env,
                                        netAddressBits groupAddress,
                                        netAddressBits sourceFilterAddress,
                                        Port port,
                                        u_int8_t ttl) {
    Groupsock* groupsock;
    do {
        struct in_addr groupAddr;
        groupAddr.s_addr = groupAddress;
        if (sourceFilterAddress == netAddressBits(~0)) {
            // regular, ISM groupsock
            groupsock = new Groupsock(env, groupAddr, port, ttl);
        } else {
            // SSM groupsock
            struct in_addr sourceFilterAddr;
            sourceFilterAddr.s_addr = sourceFilterAddress;
            groupsock = new Groupsock(env, groupAddr, sourceFilterAddr, port);
        }

        if (groupsock == NULL || groupsock->socketNum() < 0) break;

        if (!setGroupsockBySocket(env, groupsock->socketNum(), groupsock))
            break;

        fTable.Add(groupAddress, sourceFilterAddress, port, (void*)groupsock);
    } while (0);

    return groupsock;
}

GroupsockLookupTable::Iterator::Iterator(GroupsockLookupTable& groupsocks)
    : fIter(AddressPortLookupTable::Iterator(groupsocks.fTable)) {}

Groupsock* GroupsockLookupTable::Iterator::next() {
    return (Groupsock*)fIter.next();
};
