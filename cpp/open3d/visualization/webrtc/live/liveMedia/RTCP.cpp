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
// RTCP
// Implementation

#include "RTCP.hh"

#include "GroupsockHelper.hh"
#include "rtcp_from_spec.h"
#if defined(__WIN32__) || defined(_WIN32) || defined(_QNX4)
#define snprintf _snprintf
#endif

////////// RTCPMemberDatabase //////////

class RTCPMemberDatabase {
public:
    RTCPMemberDatabase(RTCPInstance& ourRTCPInstance)
        : fOurRTCPInstance(ourRTCPInstance),
          fNumMembers(1 /*ourself*/),
          fTable(HashTable::create(ONE_WORD_HASH_KEYS)) {}

    virtual ~RTCPMemberDatabase() { delete fTable; }

    Boolean isMember(u_int32_t ssrc) const {
        return fTable->Lookup((char*)(long)ssrc) != NULL;
    }

    Boolean noteMembership(u_int32_t ssrc, unsigned curTimeCount) {
        Boolean isNew = !isMember(ssrc);

        if (isNew) {
            ++fNumMembers;
        }

        // Record the current time, so we can age stale members
        fTable->Add((char*)(long)ssrc, (void*)(long)curTimeCount);

        return isNew;
    }

    Boolean remove(u_int32_t ssrc) {
        Boolean wasPresent = fTable->Remove((char*)(long)ssrc);
        if (wasPresent) {
            --fNumMembers;
        }
        return wasPresent;
    }

    unsigned numMembers() const { return fNumMembers; }

    void reapOldMembers(unsigned threshold);

private:
    RTCPInstance& fOurRTCPInstance;
    unsigned fNumMembers;
    HashTable* fTable;
};

void RTCPMemberDatabase::reapOldMembers(unsigned threshold) {
    Boolean foundOldMember;
    u_int32_t oldSSRC = 0;

    do {
        foundOldMember = False;

        HashTable::Iterator* iter = HashTable::Iterator::create(*fTable);
        uintptr_t timeCount;
        char const* key;
        while ((timeCount = (uintptr_t)(iter->next(key))) != 0) {
#ifdef DEBUG
            fprintf(stderr, "reap: checking SSRC 0x%lx: %ld (threshold %d)\n",
                    (unsigned long)key, timeCount, threshold);
#endif
            if (timeCount < (uintptr_t)threshold) {  // this SSRC is old
                uintptr_t ssrc = (uintptr_t)key;
                oldSSRC = (u_int32_t)ssrc;
                foundOldMember = True;
            }
        }
        delete iter;

        if (foundOldMember) {
#ifdef DEBUG
            fprintf(stderr, "reap: removing SSRC 0x%x\n", oldSSRC);
#endif
            fOurRTCPInstance.removeSSRC(oldSSRC, True);
        }
    } while (foundOldMember);
}

////////// RTCPInstance //////////

static double dTimeNow() {
    struct timeval timeNow;
    gettimeofday(&timeNow, NULL);
    return (double)(timeNow.tv_sec + timeNow.tv_usec / 1000000.0);
}

static unsigned const maxRTCPPacketSize = 1456;
// bytes (1500, minus some allowance for IP, UDP, UMTP headers)
static unsigned const preferredRTCPPacketSize = 1000;  // bytes

RTCPInstance::RTCPInstance(UsageEnvironment& env,
                           Groupsock* RTCPgs,
                           unsigned totSessionBW,
                           unsigned char const* cname,
                           RTPSink* sink,
                           RTPSource* source,
                           Boolean isSSMSource)
    : Medium(env),
      fRTCPInterface(this, RTCPgs),
      fTotSessionBW(totSessionBW),
      fSink(sink),
      fSource(source),
      fIsSSMSource(isSSMSource),
      fCNAME(RTCP_SDES_CNAME, cname),
      fOutgoingReportCount(1),
      fAveRTCPSize(0),
      fIsInitial(1),
      fPrevNumMembers(0),
      fLastSentSize(0),
      fLastReceivedSize(0),
      fLastReceivedSSRC(0),
      fTypeOfEvent(EVENT_UNKNOWN),
      fTypeOfPacket(PACKET_UNKNOWN_TYPE),
      fHaveJustSentPacket(False),
      fLastPacketSentSize(0),
      fByeHandlerTask(NULL),
      fByeWithReasonHandlerTask(NULL),
      fByeHandlerClientData(NULL),
      fSRHandlerTask(NULL),
      fSRHandlerClientData(NULL),
      fRRHandlerTask(NULL),
      fRRHandlerClientData(NULL),
      fSpecificRRHandlerTable(NULL),
      fAppHandlerTask(NULL),
      fAppHandlerClientData(NULL) {
#ifdef DEBUG
    fprintf(stderr, "RTCPInstance[%p]::RTCPInstance()\n", this);
#endif
    if (fTotSessionBW == 0) {  // not allowed!
        env << "RTCPInstance::RTCPInstance error: totSessionBW parameter "
               "should not be zero!\n";
        fTotSessionBW = 1;
    }

    if (isSSMSource) RTCPgs->multicastSendOnly();  // don't receive multicast

    double timeNow = dTimeNow();
    fPrevReportTime = fNextReportTime = timeNow;

    fKnownMembers = new RTCPMemberDatabase(*this);
    fInBuf = new unsigned char[maxRTCPPacketSize];
    if (fKnownMembers == NULL || fInBuf == NULL) return;
    fNumBytesAlreadyRead = 0;

    fOutBuf = new OutPacketBuffer(preferredRTCPPacketSize, maxRTCPPacketSize,
                                  maxRTCPPacketSize);
    if (fOutBuf == NULL) return;

    if (fSource != NULL && fSource->RTPgs() == RTCPgs) {
        // We're receiving RTCP reports that are multiplexed with RTP, so ask
        // the RTP source to give them to us:
        fSource->registerForMultiplexedRTCPPackets(this);
    } else {
        // Arrange to handle incoming reports from the network:
        TaskScheduler::BackgroundHandlerProc* handler =
                (TaskScheduler::BackgroundHandlerProc*)&incomingReportHandler;
        fRTCPInterface.startNetworkReading(handler);
    }

    // Send our first report.
    fTypeOfEvent = EVENT_REPORT;
    onExpire(this);
}

struct RRHandlerRecord {
    TaskFunc* rrHandlerTask;
    void* rrHandlerClientData;
};

RTCPInstance::~RTCPInstance() {
#ifdef DEBUG
    fprintf(stderr, "RTCPInstance[%p]::~RTCPInstance()\n", this);
#endif
    // Begin by sending a BYE.  We have to do this immediately, without
    // 'reconsideration', because "this" is going away.
    fTypeOfEvent = EVENT_BYE;  // not used, but...
    sendBYE();

    if (fSource != NULL && fSource->RTPgs() == fRTCPInterface.gs()) {
        // We were receiving RTCP reports that were multiplexed with RTP, so
        // tell the RTP source to stop giving them to us:
        fSource->deregisterForMultiplexedRTCPPackets();
        fRTCPInterface.forgetOurGroupsock();
        // so that the "fRTCPInterface" destructor doesn't turn off background
        // read handling
    }

    if (fSpecificRRHandlerTable != NULL) {
        AddressPortLookupTable::Iterator iter(*fSpecificRRHandlerTable);
        RRHandlerRecord* rrHandler;
        while ((rrHandler = (RRHandlerRecord*)iter.next()) != NULL) {
            delete rrHandler;
        }
        delete fSpecificRRHandlerTable;
    }

    delete fKnownMembers;
    delete fOutBuf;
    delete[] fInBuf;
}

void RTCPInstance::noteArrivingRR(struct sockaddr_in const& fromAddressAndPort,
                                  int tcpSocketNum,
                                  unsigned char tcpStreamChannelId) {
    // If a 'RR handler' was set, call it now:

    // Specific RR handler:
    if (fSpecificRRHandlerTable != NULL) {
        netAddressBits fromAddr;
        portNumBits fromPortNum;
        if (tcpSocketNum < 0) {
            // Normal case: We read the RTCP packet over UDP
            fromAddr = fromAddressAndPort.sin_addr.s_addr;
            fromPortNum = ntohs(fromAddressAndPort.sin_port);
        } else {
            // Special case: We read the RTCP packet over TCP (interleaved)
            // Hack: Use the TCP socket and channel id to look up the handler
            fromAddr = tcpSocketNum;
            fromPortNum = tcpStreamChannelId;
        }
        Port fromPort(fromPortNum);
        RRHandlerRecord* rrHandler =
                (RRHandlerRecord*)(fSpecificRRHandlerTable->Lookup(
                        fromAddr, (~0), fromPort));
        if (rrHandler != NULL) {
            if (rrHandler->rrHandlerTask != NULL) {
                (*(rrHandler->rrHandlerTask))(rrHandler->rrHandlerClientData);
            }
        }
    }

    // General RR handler:
    if (fRRHandlerTask != NULL) (*fRRHandlerTask)(fRRHandlerClientData);
}

RTCPInstance* RTCPInstance::createNew(UsageEnvironment& env,
                                      Groupsock* RTCPgs,
                                      unsigned totSessionBW,
                                      unsigned char const* cname,
                                      RTPSink* sink,
                                      RTPSource* source,
                                      Boolean isSSMSource) {
    return new RTCPInstance(env, RTCPgs, totSessionBW, cname, sink, source,
                            isSSMSource);
}

Boolean RTCPInstance::lookupByName(UsageEnvironment& env,
                                   char const* instanceName,
                                   RTCPInstance*& resultInstance) {
    resultInstance = NULL;  // unless we succeed

    Medium* medium;
    if (!Medium::lookupByName(env, instanceName, medium)) return False;

    if (!medium->isRTCPInstance()) {
        env.setResultMsg(instanceName, " is not a RTCP instance");
        return False;
    }

    resultInstance = (RTCPInstance*)medium;
    return True;
}

Boolean RTCPInstance::isRTCPInstance() const { return True; }

unsigned RTCPInstance::numMembers() const {
    if (fKnownMembers == NULL) return 0;

    return fKnownMembers->numMembers();
}

void RTCPInstance::setByeHandler(TaskFunc* handlerTask,
                                 void* clientData,
                                 Boolean handleActiveParticipantsOnly) {
    fByeHandlerTask = handlerTask;
    fByeWithReasonHandlerTask = NULL;
    fByeHandlerClientData = clientData;
    fByeHandleActiveParticipantsOnly = handleActiveParticipantsOnly;
}

void RTCPInstance::setByeWithReasonHandler(
        ByeWithReasonHandlerFunc* handlerTask,
        void* clientData,
        Boolean handleActiveParticipantsOnly) {
    fByeHandlerTask = NULL;
    fByeWithReasonHandlerTask = handlerTask;
    fByeHandlerClientData = clientData;
    fByeHandleActiveParticipantsOnly = handleActiveParticipantsOnly;
}

void RTCPInstance::setSRHandler(TaskFunc* handlerTask, void* clientData) {
    fSRHandlerTask = handlerTask;
    fSRHandlerClientData = clientData;
}

void RTCPInstance::setRRHandler(TaskFunc* handlerTask, void* clientData) {
    fRRHandlerTask = handlerTask;
    fRRHandlerClientData = clientData;
}

void RTCPInstance ::setSpecificRRHandler(netAddressBits fromAddress,
                                         Port fromPort,
                                         TaskFunc* handlerTask,
                                         void* clientData) {
    if (handlerTask == NULL && clientData == NULL) {
        unsetSpecificRRHandler(fromAddress, fromPort);
        return;
    }

    RRHandlerRecord* rrHandler = new RRHandlerRecord;
    rrHandler->rrHandlerTask = handlerTask;
    rrHandler->rrHandlerClientData = clientData;
    if (fSpecificRRHandlerTable == NULL) {
        fSpecificRRHandlerTable = new AddressPortLookupTable;
    }
    RRHandlerRecord* existingRecord =
            (RRHandlerRecord*)fSpecificRRHandlerTable->Add(fromAddress, (~0),
                                                           fromPort, rrHandler);
    delete existingRecord;  // if any
}

void RTCPInstance ::unsetSpecificRRHandler(netAddressBits fromAddress,
                                           Port fromPort) {
    if (fSpecificRRHandlerTable == NULL) return;

    RRHandlerRecord* rrHandler =
            (RRHandlerRecord*)(fSpecificRRHandlerTable->Lookup(fromAddress,
                                                               (~0), fromPort));
    if (rrHandler != NULL) {
        fSpecificRRHandlerTable->Remove(fromAddress, (~0), fromPort);
        delete rrHandler;
    }
}

void RTCPInstance::setAppHandler(RTCPAppHandlerFunc* handlerTask,
                                 void* clientData) {
    fAppHandlerTask = handlerTask;
    fAppHandlerClientData = clientData;
}

void RTCPInstance::sendAppPacket(u_int8_t subtype,
                                 char const* name,
                                 u_int8_t* appDependentData,
                                 unsigned appDependentDataSize) {
    // Set up the first 4 bytes: V,PT,subtype,PT,length:
    u_int32_t rtcpHdr = 0x80000000;  // version 2, no padding
    rtcpHdr |= (subtype & 0x1F) << 24;
    rtcpHdr |= (RTCP_PT_APP << 16);
    unsigned length = 2 + (appDependentDataSize + 3) / 4;
    rtcpHdr |= (length & 0xFFFF);
    fOutBuf->enqueueWord(rtcpHdr);

    // Set up the next 4 bytes: SSRC:
    fOutBuf->enqueueWord(fSource != NULL ? fSource->SSRC()
                                         : fSink != NULL ? fSink->SSRC() : 0);

    // Set up the next 4 bytes: name:
    char nameBytes[4];
    nameBytes[0] = nameBytes[1] = nameBytes[2] = nameBytes[3] =
            '\0';  // by default
    if (name != NULL) {
        snprintf(nameBytes, 4, "%s", name);
    }
    fOutBuf->enqueue((u_int8_t*)nameBytes, 4);

    // Set up the remaining bytes (if any): application-dependent data (+
    // padding):
    if (appDependentData != NULL && appDependentDataSize > 0) {
        fOutBuf->enqueue(appDependentData, appDependentDataSize);

        unsigned modulo = appDependentDataSize % 4;
        unsigned paddingSize = modulo == 0 ? 0 : 4 - modulo;
        u_int8_t const paddingByte = 0x00;
        for (unsigned i = 0; i < paddingSize; ++i)
            fOutBuf->enqueue(&paddingByte, 1);
    }

    // Finally, send the packet:
    sendBuiltPacket();
}

void RTCPInstance::setStreamSocket(int sockNum, unsigned char streamChannelId) {
    // Turn off background read handling:
    fRTCPInterface.stopNetworkReading();

    // Switch to RTCP-over-TCP:
    fRTCPInterface.setStreamSocket(sockNum, streamChannelId);

    // Turn background reading back on:
    TaskScheduler::BackgroundHandlerProc* handler =
            (TaskScheduler::BackgroundHandlerProc*)&incomingReportHandler;
    fRTCPInterface.startNetworkReading(handler);
}

void RTCPInstance::addStreamSocket(int sockNum, unsigned char streamChannelId) {
    // First, turn off background read handling for the default (UDP) socket:
    envir().taskScheduler().turnOffBackgroundReadHandling(
            fRTCPInterface.gs()->socketNum());

    // Add the RTCP-over-TCP interface:
    fRTCPInterface.addStreamSocket(sockNum, streamChannelId);

    // Turn on background reading for this socket (in case it's not on already):
    TaskScheduler::BackgroundHandlerProc* handler =
            (TaskScheduler::BackgroundHandlerProc*)&incomingReportHandler;
    fRTCPInterface.startNetworkReading(handler);
}

void RTCPInstance ::injectReport(u_int8_t const* packet,
                                 unsigned packetSize,
                                 struct sockaddr_in const& fromAddress) {
    if (packetSize > maxRTCPPacketSize) packetSize = maxRTCPPacketSize;
    memmove(fInBuf, packet, packetSize);

    processIncomingReport(packetSize, fromAddress, -1,
                          0xFF);  // assume report received over UDP
}

static unsigned const IP_UDP_HDR_SIZE = 28;
// overhead (bytes) of IP and UDP hdrs

#define ADVANCE(n) \
    pkt += (n);    \
    packetSize -= (n)

void RTCPInstance::incomingReportHandler(RTCPInstance* instance, int /*mask*/) {
    instance->incomingReportHandler1();
}

void RTCPInstance::incomingReportHandler1() {
    do {
        if (fNumBytesAlreadyRead >= maxRTCPPacketSize) {
            envir() << "RTCPInstance error: Hit limit when reading incoming "
                       "packet over TCP. (fNumBytesAlreadyRead ("
                    << fNumBytesAlreadyRead << ") >= maxRTCPPacketSize ("
                    << maxRTCPPacketSize
                    << ")).  The remote endpoint is using a buggy "
                       "implementation of RTP/RTCP-over-TCP.  Please upgrade "
                       "it!\n";
            break;
        }

        unsigned numBytesRead;
        struct sockaddr_in fromAddress;
        int tcpSocketNum;
        unsigned char tcpStreamChannelId;
        Boolean packetReadWasIncomplete;
        Boolean readResult = fRTCPInterface.handleRead(
                &fInBuf[fNumBytesAlreadyRead],
                maxRTCPPacketSize - fNumBytesAlreadyRead, numBytesRead,
                fromAddress, tcpSocketNum, tcpStreamChannelId,
                packetReadWasIncomplete);

        unsigned packetSize = 0;
        if (packetReadWasIncomplete) {
            fNumBytesAlreadyRead += numBytesRead;
            return;  // more reads are needed to get the entire packet
        } else {     // normal case: We've read the entire packet
            packetSize = fNumBytesAlreadyRead + numBytesRead;
            fNumBytesAlreadyRead = 0;  // for next time
        }
        if (!readResult) break;

        // Ignore the packet if it was looped-back from ourself:
        Boolean packetWasFromOurHost = False;
        if (RTCPgs()->wasLoopedBackFromUs(envir(), fromAddress)) {
            packetWasFromOurHost = True;
            // However, we still want to handle incoming RTCP packets from
            // *other processes* on the same machine.  To distinguish this
            // case from a true loop-back, check whether we've just sent a
            // packet of the same size.  (This check isn't perfect, but it seems
            // to be the best we can do.)
            if (fHaveJustSentPacket && fLastPacketSentSize == packetSize) {
                // This is a true loop-back:
                fHaveJustSentPacket = False;
                break;  // ignore this packet
            }
        }

        if (fIsSSMSource && !packetWasFromOurHost) {
            // This packet is assumed to have been received via unicast (because
            // we're a SSM source, and SSM receivers send back RTCP "RR" packets
            // via unicast). 'Reflect' the packet by resending it to the
            // multicast group, so that any other receivers can also get to see
            // it.

            // NOTE: Denial-of-service attacks are possible here.
            // Users of this software may wish to add their own,
            // application-specific mechanism for 'authenticating' the
            // validity of this packet before reflecting it.

            // NOTE: The test for "!packetWasFromOurHost" means that we won't
            // reflect RTCP packets that come from other processes on the same
            // host as us.  The reason for this is that the 'packet size' test
            // above is not 100% reliable; some packets that were truly looped
            // back from us might not be detected as such, and this might lead
            // to infinite forwarding/receiving of some packets.  To avoid this
            // possibility, we reflect only RTCP packets that we know for sure
            // originated elsewhere. (Note, though, that if we ever re-enable
            // the code in "Groupsock::multicastSendOnly()", then we could
            // remove the test for "!packetWasFromOurHost".)
            fRTCPInterface.sendPacket(fInBuf, packetSize);
            fHaveJustSentPacket = True;
            fLastPacketSentSize = packetSize;
        }

        processIncomingReport(packetSize, fromAddress, tcpSocketNum,
                              tcpStreamChannelId);
    } while (0);
}

void RTCPInstance ::processIncomingReport(
        unsigned packetSize,
        struct sockaddr_in const& fromAddressAndPort,
        int tcpSocketNum,
        unsigned char tcpStreamChannelId) {
    do {
        Boolean callByeHandler = False;
        char* reason = NULL;  // by default, unless/until a BYE packet with a
                              // 'reason' arrives
        unsigned char* pkt = fInBuf;

#ifdef DEBUG
        fprintf(stderr, "[%p]saw incoming RTCP packet (from ", this);
        if (tcpSocketNum < 0) {
            // Note that "fromAddressAndPort" is valid only if we're receiving
            // over UDP (not over TCP):
            fprintf(stderr, "address %s, port %d",
                    AddressString(fromAddressAndPort).val(),
                    ntohs(fromAddressAndPort.sin_port));
        } else {
            fprintf(stderr, "TCP socket #%d, stream channel id %d",
                    tcpSocketNum, tcpStreamChannelId);
        }
        fprintf(stderr, ")\n");
        for (unsigned i = 0; i < packetSize; ++i) {
            if (i % 4 == 0) fprintf(stderr, " ");
            fprintf(stderr, "%02x", pkt[i]);
        }
        fprintf(stderr, "\n");
#endif
        int totPacketSize = IP_UDP_HDR_SIZE + packetSize;

        // Check the RTCP packet for validity:
        // It must at least contain a header (4 bytes), and this header
        // must be version=2, with no padding bit, and a payload type of
        // SR (200), RR (201), or APP (204):
        if (packetSize < 4) break;
        unsigned rtcpHdr = ntohl(*(u_int32_t*)pkt);
        if ((rtcpHdr & 0xE0FE0000) != (0x80000000 | (RTCP_PT_SR << 16)) &&
            (rtcpHdr & 0xE0FF0000) != (0x80000000 | (RTCP_PT_APP << 16))) {
#ifdef DEBUG
            fprintf(stderr, "rejected bad RTCP packet: header 0x%08x\n",
                    rtcpHdr);
#endif
            break;
        }

        // Process each of the individual RTCP 'subpackets' in (what may be)
        // a compound RTCP packet.
        int typeOfPacket = PACKET_UNKNOWN_TYPE;
        unsigned reportSenderSSRC = 0;
        Boolean packetOK = False;
        while (1) {
            u_int8_t rc = (rtcpHdr >> 24) & 0x1F;
            u_int8_t pt = (rtcpHdr >> 16) & 0xFF;
            unsigned length = 4 * (rtcpHdr & 0xFFFF);  // doesn't count hdr
            ADVANCE(4);                                // skip over the header
            if (length > packetSize) break;

            // Assume that each RTCP subpacket begins with a 4-byte SSRC:
            if (length < 4) break;
            length -= 4;
            reportSenderSSRC = ntohl(*(u_int32_t*)pkt);
            ADVANCE(4);
#ifdef HACK_FOR_CHROME_WEBRTC_BUG
            if (reportSenderSSRC == 0x00000001 && pt == RTCP_PT_RR) {
                // Chrome (and Opera) WebRTC receivers have a bug that causes
                // them to always send SSRC 1 in their "RR"s.  To work around
                // this (to help us distinguish between different receivers), we
                // use a fake SSRC in this case consisting of the IP address,
                // XORed with the port number:
                reportSenderSSRC = fromAddressAndPort.sin_addr.s_addr ^
                                   fromAddressAndPort.sin_port;
            }
#endif

            Boolean subPacketOK = False;
            switch (pt) {
                case RTCP_PT_SR: {
#ifdef DEBUG
                    fprintf(stderr, "SR\n");
#endif
                    if (length < 20) break;
                    length -= 20;

                    // Extract the NTP timestamp, and note this:
                    unsigned NTPmsw = ntohl(*(u_int32_t*)pkt);
                    ADVANCE(4);
                    unsigned NTPlsw = ntohl(*(u_int32_t*)pkt);
                    ADVANCE(4);
                    unsigned rtpTimestamp = ntohl(*(u_int32_t*)pkt);
                    ADVANCE(4);
                    if (fSource != NULL) {
                        RTPReceptionStatsDB& receptionStats =
                                fSource->receptionStatsDB();
                        receptionStats.noteIncomingSR(reportSenderSSRC, NTPmsw,
                                                      NTPlsw, rtpTimestamp);
                    }
                    ADVANCE(8);  // skip over packet count, octet count

                    // If a 'SR handler' was set, call it now:
                    if (fSRHandlerTask != NULL)
                        (*fSRHandlerTask)(fSRHandlerClientData);

                    // The rest of the SR is handled like a RR (so, no "break;"
                    // here)
                }
                case RTCP_PT_RR: {
#ifdef DEBUG
                    fprintf(stderr, "RR\n");
#endif
                    unsigned reportBlocksSize = rc * (6 * 4);
                    if (length < reportBlocksSize) break;
                    length -= reportBlocksSize;

                    if (fSink != NULL) {
                        // Use this information to update stats about our
                        // transmissions:
                        RTPTransmissionStatsDB& transmissionStats =
                                fSink->transmissionStatsDB();
                        for (unsigned i = 0; i < rc; ++i) {
                            unsigned senderSSRC = ntohl(*(u_int32_t*)pkt);
                            ADVANCE(4);
                            // We care only about reports about our own
                            // transmission, not others'
                            if (senderSSRC == fSink->SSRC()) {
                                unsigned lossStats = ntohl(*(u_int32_t*)pkt);
                                ADVANCE(4);
                                unsigned highestReceived =
                                        ntohl(*(u_int32_t*)pkt);
                                ADVANCE(4);
                                unsigned jitter = ntohl(*(u_int32_t*)pkt);
                                ADVANCE(4);
                                unsigned timeLastSR = ntohl(*(u_int32_t*)pkt);
                                ADVANCE(4);
                                unsigned timeSinceLastSR =
                                        ntohl(*(u_int32_t*)pkt);
                                ADVANCE(4);
                                transmissionStats.noteIncomingRR(
                                        reportSenderSSRC, fromAddressAndPort,
                                        lossStats, highestReceived, jitter,
                                        timeLastSR, timeSinceLastSR);
                            } else {
                                ADVANCE(4 * 5);
                            }
                        }
                    } else {
                        ADVANCE(reportBlocksSize);
                    }

                    if (pt ==
                        RTCP_PT_RR) {  // i.e., we didn't fall through from 'SR'
                        noteArrivingRR(fromAddressAndPort, tcpSocketNum,
                                       tcpStreamChannelId);
                    }

                    subPacketOK = True;
                    typeOfPacket = PACKET_RTCP_REPORT;
                    break;
                }
                case RTCP_PT_BYE: {
#ifdef DEBUG
                    fprintf(stderr, "BYE");
#endif
                    // Check whether there was a 'reason for leaving':
                    if (length > 0) {
                        u_int8_t reasonLength = *pkt;
                        if (reasonLength > length - 1) {
                            // The 'reason' length field is too large!
#ifdef DEBUG
                            fprintf(stderr,
                                    "\nError: The 'reason' length %d is too "
                                    "large (it should be <= %d)\n",
                                    reasonLength, length - 1);
#endif
                            reasonLength = length - 1;
                        }
                        reason = new char[reasonLength + 1];
                        for (unsigned i = 0; i < reasonLength; ++i) {
                            reason[i] = pkt[1 + i];
                        }
                        reason[reasonLength] = '\0';
#ifdef DEBUG
                        fprintf(stderr, " (reason:%s)", reason);
#endif
                    }
#ifdef DEBUG
                    fprintf(stderr, "\n");
#endif
                    // If a 'BYE handler' was set, arrange for it to be called
                    // at the end of this routine. (Note: We don't call it
                    // immediately, in case it happens to cause "this" to be
                    // deleted.)
                    if ((fByeHandlerTask != NULL ||
                         fByeWithReasonHandlerTask != NULL) &&
                        (!fByeHandleActiveParticipantsOnly ||
                         (fSource != NULL &&
                          fSource->receptionStatsDB().lookup(
                                  reportSenderSSRC) != NULL) ||
                         (fSink != NULL &&
                          fSink->transmissionStatsDB().lookup(
                                  reportSenderSSRC) != NULL))) {
                        callByeHandler = True;
                    }

                    // We should really check for & handle >1 SSRCs being
                    // present #####

                    subPacketOK = True;
                    typeOfPacket = PACKET_BYE;
                    break;
                }
                case RTCP_PT_APP: {
                    u_int8_t& subtype = rc;  // In "APP" packets, the "rc" field
                                             // gets used as "subtype"
#ifdef DEBUG
                    fprintf(stderr, "APP (subtype 0x%02x)\n", subtype);
#endif
                    if (length < 4) {
#ifdef DEBUG
                        fprintf(stderr, "\tError: No \"name\" field!\n");
#endif
                        break;
                    }
                    length -= 4;
#ifdef DEBUG
                    fprintf(stderr, "\tname:%c%c%c%c\n", pkt[0], pkt[1], pkt[2],
                            pkt[3]);
#endif
                    u_int32_t nameBytes = (pkt[0] << 24) | (pkt[1] << 16) |
                                          (pkt[2] << 8) | (pkt[3]);
                    ADVANCE(4);  // skip over "name", to the
                                 // 'application-dependent data'
#ifdef DEBUG
                    fprintf(stderr,
                            "\tapplication-dependent data size: %d bytes\n",
                            length);
#endif

                    // If an 'APP' packet handler was set, call it now:
                    if (fAppHandlerTask != NULL) {
                        (*fAppHandlerTask)(fAppHandlerClientData, subtype,
                                           nameBytes, pkt, length);
                    }
                    subPacketOK = True;
                    typeOfPacket = PACKET_RTCP_APP;
                    break;
                }
                // Other RTCP packet types that we don't yet handle:
                case RTCP_PT_SDES: {
#ifdef DEBUG
                    // 'Handle' SDES packets only in debugging code, by printing
                    // out the 'SDES items':
                    fprintf(stderr, "SDES\n");

                    // Process each 'chunk':
                    Boolean chunkOK = False;
                    ADVANCE(-4);
                    length +=
                            4;  // hack so that we see the first SSRC/CSRC again
                    while (length >=
                           8) {  // A valid chunk must be at least 8 bytes long
                        chunkOK = False;  // until we learn otherwise

                        u_int32_t SSRC_CSRC = ntohl(*(u_int32_t*)pkt);
                        ADVANCE(4);
                        length -= 4;
                        fprintf(stderr, "\tSSRC/CSRC: 0x%08x\n", SSRC_CSRC);

                        // Process each 'SDES item' in the chunk:
                        u_int8_t itemType = *pkt;
                        ADVANCE(1);
                        --length;
                        while (itemType != 0) {
                            unsigned itemLen = *pkt;
                            ADVANCE(1);
                            --length;
                            // Make sure "itemLen" allows for at least 1 zero
                            // byte at the end of the chunk:
                            if (itemLen + 1 > length || pkt[itemLen] != 0)
                                break;

                            fprintf(stderr, "\t\t%s:%s\n",
		      itemType == 1 ? "CNAME" :
		      itemType == 2 ? "NAME" :
		      itemType == 3 ? "EMAIL" :
		      itemType == 4 ? "PHONE" :
		      itemType == 5 ? "LOC" :
		      itemType == 6 ? "TOOL" :
		      itemType == 7 ? "NOTE" :
		      itemType == 8 ? "PRIV" :
		      "(unknown)",
		      itemType < 8 ? (char*)pkt // hack, because we know it's '\0'-terminated
		      : "???"/* don't try to print out PRIV or unknown items */);
                            ADVANCE(itemLen);
                            length -= itemLen;

                            itemType = *pkt;
                            ADVANCE(1);
                            --length;
                        }
                        if (itemType != 0) break;  // bad 'SDES item'

                        // Thus, itemType == 0.  This zero 'type' marks the end
                        // of the list of SDES items. Skip over remaining zero
                        // padding bytes, so that this chunk ends on a 4-byte
                        // boundary:
                        while (length % 4 > 0 && *pkt == 0) {
                            ADVANCE(1);
                            --length;
                        }
                        if (length % 4 > 0)
                            break;  // Bad (non-zero) padding byte

                        chunkOK = True;
                    }
                    if (!chunkOK || length > 0)
                        break;  // bad chunk, or not enough bytes for the last
                                // chunk
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_RTPFB: {
#ifdef DEBUG
                    fprintf(stderr, "RTPFB(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_PSFB: {
#ifdef DEBUG
                    fprintf(stderr, "PSFB(unhandled)\n");
                    // Temporary code to show "Receiver Estimated Maximum
                    // Bitrate" (REMB) feedback reports:
                    //#####
                    if (length >= 12 && pkt[4] == 'R' && pkt[5] == 'E' &&
                        pkt[6] == 'M' && pkt[7] == 'B') {
                        u_int8_t exp = pkt[9] >> 2;
                        u_int32_t mantissa = ((pkt[9] & 0x03) << 16) |
                                             (pkt[10] << 8) | pkt[11];
                        double remb = (double)mantissa;
                        while (exp > 0) {
                            remb *= 2.0;
                            exp /= 2;
                        }
                        fprintf(stderr,
                                "\tReceiver Estimated Max Bitrate (REMB): %g "
                                "bps\n",
                                remb);
                    }
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_XR: {
#ifdef DEBUG
                    fprintf(stderr, "XR(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_AVB: {
#ifdef DEBUG
                    fprintf(stderr, "AVB(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_RSI: {
#ifdef DEBUG
                    fprintf(stderr, "RSI(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_TOKEN: {
#ifdef DEBUG
                    fprintf(stderr, "TOKEN(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                case RTCP_PT_IDMS: {
#ifdef DEBUG
                    fprintf(stderr, "IDMS(unhandled)\n");
#endif
                    subPacketOK = True;
                    break;
                }
                default: {
#ifdef DEBUG
                    fprintf(stderr, "UNKNOWN TYPE(0x%x)\n", pt);
#endif
                    subPacketOK = True;
                    break;
                }
            }
            if (!subPacketOK) break;

                // need to check for (& handle) SSRC collision! #####

#ifdef DEBUG
            fprintf(stderr,
                    "validated RTCP subpacket: rc:%d, pt:%d, bytes "
                    "remaining:%d, report sender SSRC:0x%08x\n",
                    rc, pt, length, reportSenderSSRC);
#endif

            // Skip over any remaining bytes in this subpacket:
            ADVANCE(length);

            // Check whether another RTCP 'subpacket' follows:
            if (packetSize == 0) {
                packetOK = True;
                break;
            } else if (packetSize < 4) {
#ifdef DEBUG
                fprintf(stderr, "extraneous %d bytes at end of RTCP packet!\n",
                        packetSize);
#endif
                break;
            }
            rtcpHdr = ntohl(*(u_int32_t*)pkt);
            if ((rtcpHdr & 0xC0000000) != 0x80000000) {
#ifdef DEBUG
                fprintf(stderr, "bad RTCP subpacket: header 0x%08x\n", rtcpHdr);
#endif
                break;
            }
        }

        if (!packetOK) {
#ifdef DEBUG
            fprintf(stderr, "rejected bad RTCP subpacket: header 0x%08x\n",
                    rtcpHdr);
#endif
            break;
        } else {
#ifdef DEBUG
            fprintf(stderr, "validated entire RTCP packet\n");
#endif
        }

        onReceive(typeOfPacket, totPacketSize, reportSenderSSRC);

        // Finally, if we need to call a "BYE" handler, do so now (in case it
        // causes "this" to get deleted):
        if (callByeHandler) {
            if (fByeHandlerTask !=
                NULL) {  // call a BYE handler without including a 'reason'
                TaskFunc* byeHandler = fByeHandlerTask;
                fByeHandlerTask = NULL;  // because we call the handler only
                                         // once, by default
                (*byeHandler)(fByeHandlerClientData);
            } else if (fByeWithReasonHandlerTask !=
                       NULL) {  // call a BYE handler that includes a 'reason'
                ByeWithReasonHandlerFunc* byeHandler =
                        fByeWithReasonHandlerTask;
                fByeWithReasonHandlerTask =
                        NULL;  // because we call the handler only once, by
                               // default
                (*byeHandler)(fByeHandlerClientData, reason);
                // Note that the handler function is responsible for delete[]ing
                // "reason"
            }
        }
    } while (0);
}

void RTCPInstance::onReceive(int typeOfPacket,
                             int totPacketSize,
                             u_int32_t ssrc) {
    fTypeOfPacket = typeOfPacket;
    fLastReceivedSize = totPacketSize;
    fLastReceivedSSRC = ssrc;

    int members = (int)numMembers();
    int senders = (fSink != NULL) ? 1 : 0;

    OnReceive(this,              // p
              this,              // e
              &members,          // members
              &fPrevNumMembers,  // pmembers
              &senders,          // senders
              &fAveRTCPSize,     // avg_rtcp_size
              &fPrevReportTime,  // tp
              dTimeNow(),        // tc
              fNextReportTime);
}

void RTCPInstance::sendReport() {
#ifdef DEBUG
    fprintf(stderr, "sending REPORT\n");
#endif
    // Begin by including a SR and/or RR report:
    if (!addReport()) return;

    // Then, include a SDES:
    addSDES();

    // Send the report:
    sendBuiltPacket();

    // Periodically clean out old members from our SSRC membership database:
    const unsigned membershipReapPeriod = 5;
    if ((++fOutgoingReportCount) % membershipReapPeriod == 0) {
        unsigned threshold = fOutgoingReportCount - membershipReapPeriod;
        fKnownMembers->reapOldMembers(threshold);
    }
}

void RTCPInstance::sendBYE(char const* reason) {
#ifdef DEBUG
    if (reason != NULL) {
        fprintf(stderr, "sending BYE (reason:%s)\n", reason);
    } else {
        fprintf(stderr, "sending BYE\n");
    }
#endif
    // The packet must begin with a SR and/or RR report:
    (void)addReport(True);

    addBYE(reason);
    sendBuiltPacket();
}

void RTCPInstance::sendBuiltPacket() {
#ifdef DEBUG
    fprintf(stderr, "sending RTCP packet\n");
    unsigned char* p = fOutBuf->packet();
    for (unsigned i = 0; i < fOutBuf->curPacketSize(); ++i) {
        if (i % 4 == 0) fprintf(stderr, " ");
        fprintf(stderr, "%02x", p[i]);
    }
    fprintf(stderr, "\n");
#endif
    unsigned reportSize = fOutBuf->curPacketSize();
    fRTCPInterface.sendPacket(fOutBuf->packet(), reportSize);
    fOutBuf->resetOffset();

    fLastSentSize = IP_UDP_HDR_SIZE + reportSize;
    fHaveJustSentPacket = True;
    fLastPacketSentSize = reportSize;
}

int RTCPInstance::checkNewSSRC() {
    return fKnownMembers->noteMembership(fLastReceivedSSRC,
                                         fOutgoingReportCount);
}

void RTCPInstance::removeLastReceivedSSRC() {
    removeSSRC(fLastReceivedSSRC, False /*keep stats around*/);
}

void RTCPInstance::removeSSRC(u_int32_t ssrc, Boolean alsoRemoveStats) {
    fKnownMembers->remove(ssrc);

    if (alsoRemoveStats) {
        // Also, remove records of this SSRC from any reception or transmission
        // stats
        if (fSource != NULL) fSource->receptionStatsDB().removeRecord(ssrc);
        if (fSink != NULL) fSink->transmissionStatsDB().removeRecord(ssrc);
    }
}

void RTCPInstance::onExpire(RTCPInstance* instance) { instance->onExpire1(); }

// Member functions to build specific kinds of report:

Boolean RTCPInstance::addReport(Boolean alwaysAdd) {
    // Include a SR or a RR, depending on whether we have an associated sink or
    // source:
    if (fSink != NULL) {
        if (!alwaysAdd) {
            if (!fSink->enableRTCPReports()) return False;

            // Hack: Don't send a SR during those (brief) times when the
            // timestamp of the next outgoing RTP packet has been preset, to
            // ensure that that timestamp gets used for that outgoing packet.
            // (David Bertrand, 2006.07.18)
            if (fSink->nextTimestampHasBeenPreset()) return False;
        }

        addSR();
    }
    if (fSource != NULL) {
        if (!alwaysAdd) {
            if (!fSource->enableRTCPReports()) return False;
        }

        addRR();
    }

    return True;
}

void RTCPInstance::addSR() {
    // ASSERT: fSink != NULL

    enqueueCommonReportPrefix(RTCP_PT_SR, fSink->SSRC(),
                              5 /* extra words in a SR */);

    // Now, add the 'sender info' for our sink

    // Insert the NTP and RTP timestamps for the 'wallclock time':
    struct timeval timeNow;
    gettimeofday(&timeNow, NULL);
    fOutBuf->enqueueWord(timeNow.tv_sec + 0x83AA7E80);
    // NTP timestamp most-significant word (1970 epoch -> 1900 epoch)
    double fractionalPart =
            (timeNow.tv_usec / 15625.0) * 0x04000000;  // 2^32/10^6
    fOutBuf->enqueueWord((unsigned)(fractionalPart + 0.5));
    // NTP timestamp least-significant word
    unsigned rtpTimestamp = fSink->convertToRTPTimestamp(timeNow);
    fOutBuf->enqueueWord(rtpTimestamp);  // RTP ts

    // Insert the packet and byte counts:
    fOutBuf->enqueueWord(fSink->packetCount());
    fOutBuf->enqueueWord(fSink->octetCount());

    enqueueCommonReportSuffix();
}

void RTCPInstance::addRR() {
    // ASSERT: fSource != NULL

    enqueueCommonReportPrefix(RTCP_PT_RR, fSource->SSRC());
    enqueueCommonReportSuffix();
}

void RTCPInstance::enqueueCommonReportPrefix(unsigned char packetType,
                                             u_int32_t SSRC,
                                             unsigned numExtraWords) {
    unsigned numReportingSources;
    if (fSource == NULL) {
        numReportingSources = 0;  // we don't receive anything
    } else {
        RTPReceptionStatsDB& allReceptionStats = fSource->receptionStatsDB();
        numReportingSources =
                allReceptionStats.numActiveSourcesSinceLastReset();
        // This must be <32, to fit in 5 bits:
        if (numReportingSources >= 32) {
            numReportingSources = 32;
        }
        // Later: support adding more reports to handle >32 sources
        // (unlikely)#####
    }

    unsigned rtcpHdr = 0x80000000;  // version 2, no padding
    rtcpHdr |= (numReportingSources << 24);
    rtcpHdr |= (packetType << 16);
    rtcpHdr |= (1 + numExtraWords + 6 * numReportingSources);
    // each report block is 6 32-bit words long
    fOutBuf->enqueueWord(rtcpHdr);

    fOutBuf->enqueueWord(SSRC);
}

void RTCPInstance::enqueueCommonReportSuffix() {
    // Output the report blocks for each source:
    if (fSource != NULL) {
        RTPReceptionStatsDB& allReceptionStats = fSource->receptionStatsDB();

        RTPReceptionStatsDB::Iterator iterator(allReceptionStats);
        while (1) {
            RTPReceptionStats* receptionStats = iterator.next();
            if (receptionStats == NULL) break;
            enqueueReportBlock(receptionStats);
        }

        allReceptionStats.reset();  // because we have just generated a report
    }
}

void RTCPInstance::enqueueReportBlock(RTPReceptionStats* stats) {
    fOutBuf->enqueueWord(stats->SSRC());

    unsigned highestExtSeqNumReceived = stats->highestExtSeqNumReceived();

    unsigned totNumExpected =
            highestExtSeqNumReceived - stats->baseExtSeqNumReceived();
    int totNumLost = totNumExpected - stats->totNumPacketsReceived();
    // 'Clamp' this loss number to a 24-bit signed value:
    if (totNumLost > 0x007FFFFF) {
        totNumLost = 0x007FFFFF;
    } else if (totNumLost < 0) {
        if (totNumLost < -0x00800000)
            totNumLost = 0x00800000;  // unlikely, but...
        totNumLost &= 0x00FFFFFF;
    }

    unsigned numExpectedSinceLastReset =
            highestExtSeqNumReceived - stats->lastResetExtSeqNumReceived();
    int numLostSinceLastReset = numExpectedSinceLastReset -
                                stats->numPacketsReceivedSinceLastReset();
    unsigned char lossFraction;
    if (numExpectedSinceLastReset == 0 || numLostSinceLastReset < 0) {
        lossFraction = 0;
    } else {
        lossFraction = (unsigned char)((numLostSinceLastReset << 8) /
                                       numExpectedSinceLastReset);
    }

    fOutBuf->enqueueWord((lossFraction << 24) | totNumLost);
    fOutBuf->enqueueWord(highestExtSeqNumReceived);

    fOutBuf->enqueueWord(stats->jitter());

    unsigned NTPmsw = stats->lastReceivedSR_NTPmsw();
    unsigned NTPlsw = stats->lastReceivedSR_NTPlsw();
    unsigned LSR =
            ((NTPmsw & 0xFFFF) << 16) | (NTPlsw >> 16);  // middle 32 bits
    fOutBuf->enqueueWord(LSR);

    // Figure out how long has elapsed since the last SR rcvd from this src:
    struct timeval const& LSRtime = stats->lastReceivedSR_time();  // "last SR"
    struct timeval timeNow, timeSinceLSR;
    gettimeofday(&timeNow, NULL);
    if (timeNow.tv_usec < LSRtime.tv_usec) {
        timeNow.tv_usec += 1000000;
        timeNow.tv_sec -= 1;
    }
    timeSinceLSR.tv_sec = timeNow.tv_sec - LSRtime.tv_sec;
    timeSinceLSR.tv_usec = timeNow.tv_usec - LSRtime.tv_usec;
    // The enqueued time is in units of 1/65536 seconds.
    // (Note that 65536/1000000 == 1024/15625)
    unsigned DLSR;
    if (LSR == 0) {
        DLSR = 0;
    } else {
        DLSR = (timeSinceLSR.tv_sec << 16) |
               ((((timeSinceLSR.tv_usec << 11) + 15625) / 31250) & 0xFFFF);
    }
    fOutBuf->enqueueWord(DLSR);
}

void RTCPInstance::addSDES() {
    // For now we support only the CNAME item; later support more #####

    // Begin by figuring out the size of the entire SDES report:
    unsigned numBytes = 4;
    // counts the SSRC, but not the header; it'll get subtracted out
    numBytes += fCNAME.totalSize();  // includes id and length
    numBytes += 1;                   // the special END item

    unsigned num4ByteWords = (numBytes + 3) / 4;

    unsigned rtcpHdr = 0x81000000;  // version 2, no padding, 1 SSRC chunk
    rtcpHdr |= (RTCP_PT_SDES << 16);
    rtcpHdr |= num4ByteWords;
    fOutBuf->enqueueWord(rtcpHdr);

    if (fSource != NULL) {
        fOutBuf->enqueueWord(fSource->SSRC());
    } else if (fSink != NULL) {
        fOutBuf->enqueueWord(fSink->SSRC());
    }

    // Add the CNAME:
    fOutBuf->enqueue(fCNAME.data(), fCNAME.totalSize());

    // Add the 'END' item (i.e., a zero byte), plus any more needed to pad:
    unsigned numPaddingBytesNeeded = 4 - (fOutBuf->curPacketSize() % 4);
    unsigned char const zero = '\0';
    while (numPaddingBytesNeeded-- > 0) fOutBuf->enqueue(&zero, 1);
}

void RTCPInstance::addBYE(char const* reason) {
    u_int32_t rtcpHdr = 0x81000000;  // version 2, no padding, 1 SSRC
    rtcpHdr |= (RTCP_PT_BYE << 16);
    u_int16_t num32BitWords =
            2;  // by default, two 32-bit words total (i.e., with 1 SSRC)
    u_int8_t reasonLength8Bits = 0;  // by default
    if (reason != NULL) {
        // We need to add more 32-bit words for the 'length+reason':
        unsigned const reasonLength = strlen(reason);
        reasonLength8Bits = reasonLength < 0xFF ? (u_int8_t)reasonLength : 0xFF;
        unsigned numExtraWords =
                ((1 /*reason length field*/ + reasonLength8Bits) + 3) / 4;

        num32BitWords += numExtraWords;
    }
    rtcpHdr |= (num32BitWords - 1);  // length field
    fOutBuf->enqueueWord(rtcpHdr);

    if (fSource != NULL) {
        fOutBuf->enqueueWord(fSource->SSRC());
    } else if (fSink != NULL) {
        fOutBuf->enqueueWord(fSink->SSRC());
    }

    num32BitWords -= 2;  // ASSERT: num32BitWords >= 0
    if (num32BitWords > 0) {
        // Add a length+'reason for leaving':
        // First word:
        u_int32_t lengthPlusFirst3ReasonBytes = reasonLength8Bits << 24;
        unsigned index = 0;
        if (reasonLength8Bits > index)
            lengthPlusFirst3ReasonBytes |= ((u_int8_t)reason[index++]) << 16;
        if (reasonLength8Bits > index)
            lengthPlusFirst3ReasonBytes |= ((u_int8_t)reason[index++]) << 8;
        if (reasonLength8Bits > index)
            lengthPlusFirst3ReasonBytes |= (u_int8_t)reason[index++];
        fOutBuf->enqueueWord(lengthPlusFirst3ReasonBytes);

        // Any subsequent words:
        if (reasonLength8Bits > 3) {
            // ASSERT: num32BitWords > 1
            while (--num32BitWords > 0) {
                u_int32_t fourMoreReasonBytes = 0;
                if (reasonLength8Bits > index)
                    fourMoreReasonBytes |= ((u_int8_t)reason[index++]) << 24;
                if (reasonLength8Bits > index)
                    fourMoreReasonBytes |= ((u_int8_t)reason[index++]) << 16;
                if (reasonLength8Bits > index)
                    fourMoreReasonBytes |= ((u_int8_t)reason[index++]) << 8;
                if (reasonLength8Bits > index)
                    fourMoreReasonBytes |= (u_int8_t)reason[index++];
                fOutBuf->enqueueWord(fourMoreReasonBytes);
            }
        }
    }
}

void RTCPInstance::schedule(double nextTime) {
    fNextReportTime = nextTime;

    double secondsToDelay = nextTime - dTimeNow();
    if (secondsToDelay < 0) secondsToDelay = 0;
#ifdef DEBUG
    fprintf(stderr, "schedule(%f->%f)\n", secondsToDelay, nextTime);
#endif
    int64_t usToGo = (int64_t)(secondsToDelay * 1000000);
    nextTask() = envir().taskScheduler().scheduleDelayedTask(
            usToGo, (TaskFunc*)RTCPInstance::onExpire, this);
}

void RTCPInstance::reschedule(double nextTime) {
    envir().taskScheduler().unscheduleDelayedTask(nextTask());
    schedule(nextTime);
}

void RTCPInstance::onExpire1() {
    nextTask() = NULL;

    // Note: fTotSessionBW is kbits per second
    double rtcpBW = 0.05 * fTotSessionBW * 1024 / 8;  // -> bytes per second

    OnExpire(this,                     // event
             numMembers(),             // members
             (fSink != NULL) ? 1 : 0,  // senders
             rtcpBW,                   // rtcp_bw
             (fSink != NULL) ? 1 : 0,  // we_sent
             &fAveRTCPSize,            // ave_rtcp_size
             &fIsInitial,              // initial
             dTimeNow(),               // tc
             &fPrevReportTime,         // tp
             &fPrevNumMembers          // pmembers
    );
}

////////// SDESItem //////////

SDESItem::SDESItem(unsigned char tag, unsigned char const* value) {
    unsigned length = strlen((char const*)value);
    if (length > 0xFF) length = 0xFF;  // maximum data length for a SDES item

    fData[0] = tag;
    fData[1] = (unsigned char)length;
    memmove(&fData[2], value, length);
}

unsigned SDESItem::totalSize() const { return 2 + (unsigned)fData[1]; }

////////// Implementation of routines imported by the "rtcp_from_spec" C code

extern "C" void Schedule(double nextTime, event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return;

    instance->schedule(nextTime);
}

extern "C" void Reschedule(double nextTime, event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return;

    instance->reschedule(nextTime);
}

extern "C" void SendRTCPReport(event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return;

    instance->sendReport();
}

extern "C" void SendBYEPacket(event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return;

    instance->sendBYE();
}

extern "C" int TypeOfEvent(event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return EVENT_UNKNOWN;

    return instance->typeOfEvent();
}

extern "C" int SentPacketSize(event e) {
    RTCPInstance* instance = (RTCPInstance*)e;
    if (instance == NULL) return 0;

    return instance->sentPacketSize();
}

extern "C" int PacketType(packet p) {
    RTCPInstance* instance = (RTCPInstance*)p;
    if (instance == NULL) return PACKET_UNKNOWN_TYPE;

    return instance->packetType();
}

extern "C" int ReceivedPacketSize(packet p) {
    RTCPInstance* instance = (RTCPInstance*)p;
    if (instance == NULL) return 0;

    return instance->receivedPacketSize();
}

extern "C" int NewMember(packet p) {
    RTCPInstance* instance = (RTCPInstance*)p;
    if (instance == NULL) return 0;

    return instance->checkNewSSRC();
}

extern "C" int NewSender(packet /*p*/) {
    return 0;  // we don't yet recognize senders other than ourselves #####
}

extern "C" void AddMember(packet /*p*/) {
    // Do nothing; all of the real work was done when NewMember() was called
}

extern "C" void AddSender(packet /*p*/) {
    // we don't yet recognize senders other than ourselves #####
}

extern "C" void RemoveMember(packet p) {
    RTCPInstance* instance = (RTCPInstance*)p;
    if (instance == NULL) return;

    instance->removeLastReceivedSSRC();
}

extern "C" void RemoveSender(packet /*p*/) {
    // we don't yet recognize senders other than ourselves #####
}

extern "C" double drand30() {
    unsigned tmp = our_random() & 0x3FFFFFFF;  // a random 30-bit integer
    return tmp / (double)(1024 * 1024 * 1024);
}
