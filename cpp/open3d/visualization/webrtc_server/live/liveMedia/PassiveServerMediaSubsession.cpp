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
// A 'ServerMediaSubsession' object that represents an existing
// 'RTPSink', rather than one that creates new 'RTPSink's on demand.
// Implementation

#include "PassiveServerMediaSubsession.hh"

#include <GroupsockHelper.hh>

////////// PassiveServerMediaSubsession //////////

PassiveServerMediaSubsession* PassiveServerMediaSubsession::createNew(
        RTPSink& rtpSink, RTCPInstance* rtcpInstance) {
    return new PassiveServerMediaSubsession(rtpSink, rtcpInstance);
}

PassiveServerMediaSubsession ::PassiveServerMediaSubsession(
        RTPSink& rtpSink, RTCPInstance* rtcpInstance)
    : ServerMediaSubsession(rtpSink.envir()),
      fSDPLines(NULL),
      fRTPSink(rtpSink),
      fRTCPInstance(rtcpInstance) {
    fClientRTCPSourceRecords = HashTable::create(ONE_WORD_HASH_KEYS);
}

class RTCPSourceRecord {
public:
    RTCPSourceRecord(netAddressBits addr, Port const& port)
        : addr(addr), port(port) {}

    netAddressBits addr;
    Port port;
};

PassiveServerMediaSubsession::~PassiveServerMediaSubsession() {
    delete[] fSDPLines;

    // Clean out the RTCPSourceRecord table:
    while (1) {
        RTCPSourceRecord* source =
                (RTCPSourceRecord*)(fClientRTCPSourceRecords->RemoveNext());
        if (source == NULL) break;
        delete source;
    }

    delete fClientRTCPSourceRecords;
}

Boolean PassiveServerMediaSubsession::rtcpIsMuxed() {
    if (fRTCPInstance == NULL) return False;

    // Check whether RTP and RTCP use the same "groupsock" object:
    return &(fRTPSink.groupsockBeingUsed()) == fRTCPInstance->RTCPgs();
}

char const* PassiveServerMediaSubsession::sdpLines() {
    if (fSDPLines == NULL) {
        // Construct a set of SDP lines that describe this subsession:
        // Use the components from "rtpSink":
        Groupsock const& gs = fRTPSink.groupsockBeingUsed();
        AddressString groupAddressStr(gs.groupAddress());
        unsigned short portNum = ntohs(gs.port().num());
        unsigned char ttl = gs.ttl();
        unsigned char rtpPayloadType = fRTPSink.rtpPayloadType();
        char const* mediaType = fRTPSink.sdpMediaType();
        unsigned estBitrate =
                fRTCPInstance == NULL ? 50 : fRTCPInstance->totSessionBW();
        char* rtpmapLine = fRTPSink.rtpmapLine();
        char const* rtcpmuxLine = rtcpIsMuxed() ? "a=rtcp-mux\r\n" : "";
        char const* rangeLine = rangeSDPLine();
        char const* auxSDPLine = fRTPSink.auxSDPLine();
        if (auxSDPLine == NULL) auxSDPLine = "";

        char const* const sdpFmt =
                "m=%s %d RTP/AVP %d\r\n"
                "c=IN IP4 %s/%d\r\n"
                "b=AS:%u\r\n"
                "%s"
                "%s"
                "%s"
                "%s"
                "a=control:%s\r\n";
        unsigned sdpFmtSize =
                strlen(sdpFmt) + strlen(mediaType) + 5 /* max short len */ +
                3                                   /* max char len */
                + strlen(groupAddressStr.val()) + 3 /* max char len */
                + 20                                /* max int len */
                + strlen(rtpmapLine) + strlen(rtcpmuxLine) + strlen(rangeLine) +
                strlen(auxSDPLine) + strlen(trackId());
        char* sdpLines = new char[sdpFmtSize];
        sprintf(sdpLines, sdpFmt,
                mediaType,              // m= <media>
                portNum,                // m= <port>
                rtpPayloadType,         // m= <fmt list>
                groupAddressStr.val(),  // c= <connection address>
                ttl,                    // c= TTL
                estBitrate,             // b=AS:<bandwidth>
                rtpmapLine,             // a=rtpmap:... (if present)
                rtcpmuxLine,            // a=rtcp-mux:... (if present)
                rangeLine,              // a=range:... (if present)
                auxSDPLine,             // optional extra SDP line
                trackId());             // a=control:<track-id>
        delete[](char*) rangeLine;
        delete[] rtpmapLine;

        fSDPLines = strDup(sdpLines);
        delete[] sdpLines;
    }

    return fSDPLines;
}

void PassiveServerMediaSubsession ::getStreamParameters(
        unsigned clientSessionId,
        netAddressBits clientAddress,
        Port const& /*clientRTPPort*/,
        Port const& clientRTCPPort,
        int /*tcpSocketNum*/,
        unsigned char /*rtpChannelId*/,
        unsigned char /*rtcpChannelId*/,
        netAddressBits& destinationAddress,
        u_int8_t& destinationTTL,
        Boolean& isMulticast,
        Port& serverRTPPort,
        Port& serverRTCPPort,
        void*& streamToken) {
    isMulticast = True;
    Groupsock& gs = fRTPSink.groupsockBeingUsed();
    if (destinationTTL == 255) destinationTTL = gs.ttl();
    if (destinationAddress == 0) {  // normal case
        destinationAddress = gs.groupAddress().s_addr;
    } else {  // use the client-specified destination address instead:
        struct in_addr destinationAddr;
        destinationAddr.s_addr = destinationAddress;
        gs.changeDestinationParameters(destinationAddr, 0, destinationTTL);
        if (fRTCPInstance != NULL) {
            Groupsock* rtcpGS = fRTCPInstance->RTCPgs();
            rtcpGS->changeDestinationParameters(destinationAddr, 0,
                                                destinationTTL);
        }
    }
    serverRTPPort = gs.port();
    if (fRTCPInstance != NULL) {
        Groupsock* rtcpGS = fRTCPInstance->RTCPgs();
        serverRTCPPort = rtcpGS->port();
    }
    streamToken = NULL;  // not used

    // Make a record of this client's source - for RTCP RR handling:
    RTCPSourceRecord* source =
            new RTCPSourceRecord(clientAddress, clientRTCPPort);
    fClientRTCPSourceRecords->Add(
            reinterpret_cast<char const*>(clientSessionId), source);
}

void PassiveServerMediaSubsession::startStream(
        unsigned clientSessionId,
        void* /*streamToken*/,
        TaskFunc* rtcpRRHandler,
        void* rtcpRRHandlerClientData,
        unsigned short& rtpSeqNum,
        unsigned& rtpTimestamp,
        ServerRequestAlternativeByteHandler* /*serverRequestAlternativeByteHandler*/
        ,
        void* /*serverRequestAlternativeByteHandlerClientData*/) {
    rtpSeqNum = fRTPSink.currentSeqNo();
    rtpTimestamp = fRTPSink.presetNextTimestamp();

    // Try to use a big send buffer for RTP -  at least 0.1 second of
    // specified bandwidth and at least 50 KB
    unsigned streamBitrate =
            fRTCPInstance == NULL ? 50
                                  : fRTCPInstance->totSessionBW();  // in kbps
    unsigned rtpBufSize =
            streamBitrate * 25 / 2;  // 1 kbps * 0.1 s = 12.5 bytes
    if (rtpBufSize < 50 * 1024) rtpBufSize = 50 * 1024;
    increaseSendBufferTo(envir(), fRTPSink.groupsockBeingUsed().socketNum(),
                         rtpBufSize);

    if (fRTCPInstance != NULL) {
        // Hack: Send a RTCP "SR" packet now, so that receivers will (likely) be
        // able to get RTCP-synchronized presentation times immediately:
        fRTCPInstance->sendReport();

        // Set up the handler for incoming RTCP "RR" packets from this client:
        RTCPSourceRecord* source =
                (RTCPSourceRecord*)(fClientRTCPSourceRecords->Lookup(
                        reinterpret_cast<char const*>(clientSessionId)));
        if (source != NULL) {
            fRTCPInstance->setSpecificRRHandler(source->addr, source->port,
                                                rtcpRRHandler,
                                                rtcpRRHandlerClientData);
        }
    }
}

float PassiveServerMediaSubsession::getCurrentNPT(void* streamToken) {
    // Return the elapsed time between our "RTPSink"s creation time, and the
    // current time:
    struct timeval const& creationTime = fRTPSink.creationTime();  // alias

    struct timeval timeNow;
    gettimeofday(&timeNow, NULL);

    return (float)(timeNow.tv_sec - creationTime.tv_sec +
                   (timeNow.tv_usec - creationTime.tv_usec) / 1000000.0);
}

void PassiveServerMediaSubsession ::getRTPSinkandRTCP(
        void* streamToken, RTPSink const*& rtpSink, RTCPInstance const*& rtcp) {
    rtpSink = &fRTPSink;
    rtcp = fRTCPInstance;
}

void PassiveServerMediaSubsession::deleteStream(unsigned clientSessionId,
                                                void*& /*streamToken*/) {
    // Lookup and remove the 'RTCPSourceRecord' for this client.  Also turn off
    // RTCP "RR" handling:
    RTCPSourceRecord* source =
            (RTCPSourceRecord*)(fClientRTCPSourceRecords->Lookup(
                    reinterpret_cast<char const*>(clientSessionId)));
    if (source != NULL) {
        if (fRTCPInstance != NULL) {
            fRTCPInstance->unsetSpecificRRHandler(source->addr, source->port);
        }

        fClientRTCPSourceRecords->Remove(
                reinterpret_cast<char const*>(clientSessionId));
        delete source;
    }
}
