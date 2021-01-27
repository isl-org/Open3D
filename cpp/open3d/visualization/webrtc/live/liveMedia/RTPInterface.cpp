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
// An abstraction of a network interface used for RTP (or RTCP).
// (This allows the RTP-over-TCP hack (RFC 2326, section 10.12) to
// be implemented transparently.)
// Implementation

#include "RTPInterface.hh"

#include <stdio.h>

#include <GroupsockHelper.hh>

////////// Helper Functions - Definition //////////

// Helper routines and data structures, used to implement
// sending/receiving RTP/RTCP over a TCP socket:

// Reading RTP-over-TCP is implemented using two levels of hash tables.
// The top-level hash table maps TCP socket numbers to a
// "SocketDescriptor" that contains a hash table for each of the
// sub-channels that are reading from this socket.

static HashTable* socketHashTable(UsageEnvironment& env,
                                  Boolean createIfNotPresent = True) {
    _Tables* ourTables = _Tables::getOurTables(env, createIfNotPresent);
    if (ourTables == NULL) return NULL;

    if (ourTables->socketTable == NULL) {
        // Create a new socket number -> SocketDescriptor mapping table:
        ourTables->socketTable = HashTable::create(ONE_WORD_HASH_KEYS);
    }
    return (HashTable*)(ourTables->socketTable);
}

class SocketDescriptor {
public:
    SocketDescriptor(UsageEnvironment& env, int socketNum);
    virtual ~SocketDescriptor();

    void registerRTPInterface(unsigned char streamChannelId,
                              RTPInterface* rtpInterface);
    RTPInterface* lookupRTPInterface(unsigned char streamChannelId);
    void deregisterRTPInterface(unsigned char streamChannelId);

    void setServerRequestAlternativeByteHandler(
            ServerRequestAlternativeByteHandler* handler, void* clientData) {
        fServerRequestAlternativeByteHandler = handler;
        fServerRequestAlternativeByteHandlerClientData = clientData;
    }

private:
    static void tcpReadHandler(SocketDescriptor*, int mask);
    Boolean tcpReadHandler1(int mask);

private:
    UsageEnvironment& fEnv;
    int fOurSocketNum;
    HashTable* fSubChannelHashTable;
    ServerRequestAlternativeByteHandler* fServerRequestAlternativeByteHandler;
    void* fServerRequestAlternativeByteHandlerClientData;
    u_int8_t fStreamChannelId, fSizeByte1;
    Boolean fReadErrorOccurred, fDeleteMyselfNext, fAreInReadHandlerLoop;
    enum {
        AWAITING_DOLLAR,
        AWAITING_STREAM_CHANNEL_ID,
        AWAITING_SIZE1,
        AWAITING_SIZE2,
        AWAITING_PACKET_DATA
    } fTCPReadingState;
};

static SocketDescriptor* lookupSocketDescriptor(
        UsageEnvironment& env, int sockNum, Boolean createIfNotFound = True) {
    HashTable* table = socketHashTable(env, createIfNotFound);
    if (table == NULL) return NULL;

    char const* key = (char const*)(long)sockNum;
    SocketDescriptor* socketDescriptor =
            (SocketDescriptor*)(table->Lookup(key));
    if (socketDescriptor == NULL) {
        if (createIfNotFound) {
            socketDescriptor = new SocketDescriptor(env, sockNum);
            table->Add((char const*)(long)(sockNum), socketDescriptor);
        } else if (table->IsEmpty()) {
            // We can also delete the table (to reclaim space):
            _Tables* ourTables = _Tables::getOurTables(env);
            delete table;
            ourTables->socketTable = NULL;
            ourTables->reclaimIfPossible();
        }
    }

    return socketDescriptor;
}

static void removeSocketDescription(UsageEnvironment& env, int sockNum) {
    char const* key = (char const*)(long)sockNum;
    HashTable* table = socketHashTable(env);
    table->Remove(key);

    if (table->IsEmpty()) {
        // We can also delete the table (to reclaim space):
        _Tables* ourTables = _Tables::getOurTables(env);
        delete table;
        ourTables->socketTable = NULL;
        ourTables->reclaimIfPossible();
    }
}

////////// RTPInterface - Implementation //////////

RTPInterface::RTPInterface(Medium* owner, Groupsock* gs)
    : fOwner(owner),
      fGS(gs),
      fTCPStreams(NULL),
      fNextTCPReadSize(0),
      fNextTCPReadStreamSocketNum(-1),
      fNextTCPReadStreamChannelId(0xFF),
      fReadHandlerProc(NULL),
      fAuxReadHandlerFunc(NULL),
      fAuxReadHandlerClientData(NULL) {
    // Make the socket non-blocking, even though it will be read from only
    // asynchronously, when packets arrive. The reason for this is that, in some
    // OSs, reads on a blocking socket can (allegedly) sometimes block, even if
    // the socket was previously reported (e.g., by "select()") as having data
    // available. (This can supposedly happen if the UDP checksum fails, for
    // example.)
    makeSocketNonBlocking(fGS->socketNum());
    increaseSendBufferTo(envir(), fGS->socketNum(), 50 * 1024);
}

RTPInterface::~RTPInterface() {
    stopNetworkReading();
    delete fTCPStreams;
}

void RTPInterface::setStreamSocket(int sockNum, unsigned char streamChannelId) {
    fGS->removeAllDestinations();
    envir().taskScheduler().disableBackgroundHandling(
            fGS->socketNum());  // turn off any reading on our datagram socket
    fGS->reset();  // and close our datagram socket, because we won't be using
                   // it anymore

    addStreamSocket(sockNum, streamChannelId);
}

void RTPInterface::addStreamSocket(int sockNum, unsigned char streamChannelId) {
    if (sockNum < 0) return;

    for (tcpStreamRecord* streams = fTCPStreams; streams != NULL;
         streams = streams->fNext) {
        if (streams->fStreamSocketNum == sockNum &&
            streams->fStreamChannelId == streamChannelId) {
            return;  // we already have it
        }
    }

    fTCPStreams = new tcpStreamRecord(sockNum, streamChannelId, fTCPStreams);

    // Also, make sure this new socket is set up for receiving
    // RTP/RTCP-over-TCP:
    SocketDescriptor* socketDescriptor =
            lookupSocketDescriptor(envir(), sockNum);
    socketDescriptor->registerRTPInterface(streamChannelId, this);
}

static void deregisterSocket(UsageEnvironment& env,
                             int sockNum,
                             unsigned char streamChannelId) {
    SocketDescriptor* socketDescriptor =
            lookupSocketDescriptor(env, sockNum, False);
    if (socketDescriptor != NULL) {
        socketDescriptor->deregisterRTPInterface(streamChannelId);
        // Note: This may delete "socketDescriptor",
        // if no more interfaces are using this socket
    }
}

void RTPInterface::removeStreamSocket(int sockNum,
                                      unsigned char streamChannelId) {
    // Remove - from our list of 'TCP streams' - the record of the
    // (sockNum,streamChannelId) pair. (However "streamChannelId" == 0xFF is a
    // special case, meaning remove all
    //  (sockNum,*) pairs.)

    while (1) {
        tcpStreamRecord** streamsPtr = &fTCPStreams;

        while (*streamsPtr != NULL) {
            if ((*streamsPtr)->fStreamSocketNum == sockNum &&
                (streamChannelId == 0xFF ||
                 streamChannelId == (*streamsPtr)->fStreamChannelId)) {
                // Delete the record pointed to by *streamsPtr :
                unsigned char streamChannelIdToRemove =
                        (*streamsPtr)->fStreamChannelId;
                tcpStreamRecord* next = (*streamsPtr)->fNext;
                (*streamsPtr)->fNext = NULL;
                delete (*streamsPtr);
                *streamsPtr = next;

                // And 'deregister' this socket,channelId pair:
                deregisterSocket(envir(), sockNum, streamChannelIdToRemove);

                if (streamChannelId != 0xFF) return;  // we're done
                break;  // start again from the beginning of the list, in case
                        // the list has changed
            } else {
                streamsPtr = &((*streamsPtr)->fNext);
            }
        }
        if (*streamsPtr == NULL) break;
    }
}

void RTPInterface::setServerRequestAlternativeByteHandler(
        UsageEnvironment& env,
        int socketNum,
        ServerRequestAlternativeByteHandler* handler,
        void* clientData) {
    SocketDescriptor* socketDescriptor =
            lookupSocketDescriptor(env, socketNum, False);

    if (socketDescriptor != NULL)
        socketDescriptor->setServerRequestAlternativeByteHandler(handler,
                                                                 clientData);
}

void RTPInterface::clearServerRequestAlternativeByteHandler(
        UsageEnvironment& env, int socketNum) {
    setServerRequestAlternativeByteHandler(env, socketNum, NULL, NULL);
}

Boolean RTPInterface::sendPacket(unsigned char* packet, unsigned packetSize) {
    Boolean success =
            True;  // we'll return False instead if any of the sends fail

    // Normal case: Send as a UDP packet:
    if (!fGS->output(envir(), packet, packetSize)) success = False;

    // Also, send over each of our TCP sockets:
    tcpStreamRecord* nextStream;
    for (tcpStreamRecord* stream = fTCPStreams; stream != NULL;
         stream = nextStream) {
        nextStream = stream->fNext;  // Set this now, in case the following
                                     // deletes "stream":
        if (!sendRTPorRTCPPacketOverTCP(packet, packetSize,
                                        stream->fStreamSocketNum,
                                        stream->fStreamChannelId)) {
            success = False;
        }
    }

    return success;
}

void RTPInterface ::startNetworkReading(
        TaskScheduler::BackgroundHandlerProc* handlerProc) {
    // Normal case: Arrange to read UDP packets:
    envir().taskScheduler().turnOnBackgroundReadHandling(fGS->socketNum(),
                                                         handlerProc, fOwner);

    // Also, receive RTP over TCP, on each of our TCP connections:
    fReadHandlerProc = handlerProc;
    for (tcpStreamRecord* streams = fTCPStreams; streams != NULL;
         streams = streams->fNext) {
        // Get a socket descriptor for "streams->fStreamSocketNum":
        SocketDescriptor* socketDescriptor =
                lookupSocketDescriptor(envir(), streams->fStreamSocketNum);

        // Tell it about our subChannel:
        socketDescriptor->registerRTPInterface(streams->fStreamChannelId, this);
    }
}

Boolean RTPInterface::handleRead(unsigned char* buffer,
                                 unsigned bufferMaxSize,
                                 unsigned& bytesRead,
                                 struct sockaddr_in& fromAddress,
                                 int& tcpSocketNum,
                                 unsigned char& tcpStreamChannelId,
                                 Boolean& packetReadWasIncomplete) {
    packetReadWasIncomplete = False;  // by default
    Boolean readSuccess;
    if (fNextTCPReadStreamSocketNum < 0) {
        // Normal case: read from the (datagram) 'groupsock':
        tcpSocketNum = -1;
        readSuccess =
                fGS->handleRead(buffer, bufferMaxSize, bytesRead, fromAddress);
    } else {
        // Read from the TCP connection:
        tcpSocketNum = fNextTCPReadStreamSocketNum;
        tcpStreamChannelId = fNextTCPReadStreamChannelId;

        bytesRead = 0;
        unsigned totBytesToRead = fNextTCPReadSize;
        if (totBytesToRead > bufferMaxSize) totBytesToRead = bufferMaxSize;
        unsigned curBytesToRead = totBytesToRead;
        int curBytesRead;
        while ((curBytesRead = readSocket(envir(), fNextTCPReadStreamSocketNum,
                                          &buffer[bytesRead], curBytesToRead,
                                          fromAddress)) > 0) {
            bytesRead += curBytesRead;
            if (bytesRead >= totBytesToRead) break;
            curBytesToRead -= curBytesRead;
        }
        fNextTCPReadSize -= bytesRead;
        if (fNextTCPReadSize == 0) {
            // We've read all of the data that we asked for
            readSuccess = True;
        } else if (curBytesRead < 0) {
            // There was an error reading the socket
            bytesRead = 0;
            readSuccess = False;
        } else {
            // We need to read more bytes, and there was not an error reading
            // the socket
            packetReadWasIncomplete = True;
            return True;
        }
        fNextTCPReadStreamSocketNum = -1;  // default, for next time
    }

    if (readSuccess && fAuxReadHandlerFunc != NULL) {
        // Also pass the newly-read packet data to our auxilliary handler:
        (*fAuxReadHandlerFunc)(fAuxReadHandlerClientData, buffer, bytesRead);
    }
    return readSuccess;
}

void RTPInterface::stopNetworkReading() {
    // Normal case
    if (fGS != NULL)
        envir().taskScheduler().turnOffBackgroundReadHandling(fGS->socketNum());

    // Also turn off read handling on each of our TCP connections:
    for (tcpStreamRecord* streams = fTCPStreams; streams != NULL;
         streams = streams->fNext) {
        deregisterSocket(envir(), streams->fStreamSocketNum,
                         streams->fStreamChannelId);
    }
}

////////// Helper Functions - Implementation /////////

Boolean RTPInterface::sendRTPorRTCPPacketOverTCP(
        u_int8_t* packet,
        unsigned packetSize,
        int socketNum,
        unsigned char streamChannelId) {
#ifdef DEBUG_SEND
    fprintf(stderr,
            "sendRTPorRTCPPacketOverTCP: %d bytes over channel %d (socket "
            "%d)\n",
            packetSize, streamChannelId, socketNum);
    fflush(stderr);
#endif
    // Send a RTP/RTCP packet over TCP, using the encoding defined in RFC 2326,
    // section 10.12:
    //     $<streamChannelId><packetSize><packet>
    // (If the initial "send()" of '$<streamChannelId><packetSize>' succeeds,
    // then we force the subsequent "send()" for the <packet> data to succeed,
    // even if we have to do so with a blocking "send()".)
    do {
        u_int8_t framingHeader[4];
        framingHeader[0] = '$';
        framingHeader[1] = streamChannelId;
        framingHeader[2] = (u_int8_t)((packetSize & 0xFF00) >> 8);
        framingHeader[3] = (u_int8_t)(packetSize & 0xFF);
        if (!sendDataOverTCP(socketNum, framingHeader, 4, False)) break;

        if (!sendDataOverTCP(socketNum, packet, packetSize, True)) break;
#ifdef DEBUG_SEND
        fprintf(stderr, "sendRTPorRTCPPacketOverTCP: completed\n");
        fflush(stderr);
#endif

        return True;
    } while (0);

#ifdef DEBUG_SEND
    fprintf(stderr, "sendRTPorRTCPPacketOverTCP: failed! (errno %d)\n",
            envir().getErrno());
    fflush(stderr);
#endif
    return False;
}

#ifndef RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS
#define RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS 500
#endif

Boolean RTPInterface::sendDataOverTCP(int socketNum,
                                      u_int8_t const* data,
                                      unsigned dataSize,
                                      Boolean forceSendToSucceed) {
    int sendResult = send(socketNum, (char const*)data, dataSize, 0 /*flags*/);
    if (sendResult < (int)dataSize) {
        // The TCP send() failed - at least partially.

        unsigned numBytesSentSoFar = sendResult < 0 ? 0 : (unsigned)sendResult;
        if (numBytesSentSoFar > 0 ||
            (forceSendToSucceed && envir().getErrno() == EAGAIN)) {
            // The OS's TCP send buffer has filled up (because the stream's
            // bitrate has exceeded the capacity of the TCP connection!). Force
            // this data write to succeed, by blocking if necessary until it
            // does:
            unsigned numBytesRemainingToSend = dataSize - numBytesSentSoFar;
#ifdef DEBUG_SEND
            fprintf(stderr,
                    "sendDataOverTCP: resending %d-byte send (blocking)\n",
                    numBytesRemainingToSend);
            fflush(stderr);
#endif
            makeSocketBlocking(socketNum,
                               RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS);
            sendResult =
                    send(socketNum, (char const*)(&data[numBytesSentSoFar]),
                         numBytesRemainingToSend, 0 /*flags*/);
            if ((unsigned)sendResult != numBytesRemainingToSend) {
                // The blocking "send()" failed, or timed out.  In either case,
                // we assume that the TCP connection has failed (or is 'hanging'
                // indefinitely), and we stop using it (for both RTP and RTP).
                // (If we kept using the socket here, the RTP or RTCP packet
                // write would be in an
                //  incomplete, inconsistent state.)
#ifdef DEBUG_SEND
                fprintf(stderr,
                        "sendDataOverTCP: blocking send() failed (delivering "
                        "%d bytes out of %d); closing socket %d\n",
                        sendResult, numBytesRemainingToSend, socketNum);
                fflush(stderr);
#endif
                removeStreamSocket(socketNum, 0xFF);
                return False;
            }
            makeSocketNonBlocking(socketNum);

            return True;
        } else if (sendResult < 0 && envir().getErrno() != EAGAIN) {
            // Because the "send()" call failed, assume that the socket is now
            // unusable, so stop using it (for both RTP and RTCP):
            removeStreamSocket(socketNum, 0xFF);
        }

        return False;
    }

    return True;
}

SocketDescriptor::SocketDescriptor(UsageEnvironment& env, int socketNum)
    : fEnv(env),
      fOurSocketNum(socketNum),
      fSubChannelHashTable(HashTable::create(ONE_WORD_HASH_KEYS)),
      fServerRequestAlternativeByteHandler(NULL),
      fServerRequestAlternativeByteHandlerClientData(NULL),
      fReadErrorOccurred(False),
      fDeleteMyselfNext(False),
      fAreInReadHandlerLoop(False),
      fTCPReadingState(AWAITING_DOLLAR) {}

SocketDescriptor::~SocketDescriptor() {
    fEnv.taskScheduler().turnOffBackgroundReadHandling(fOurSocketNum);
    removeSocketDescription(fEnv, fOurSocketNum);

    if (fSubChannelHashTable != NULL) {
        // Remove knowledge of this socket from any "RTPInterface"s that are
        // using it:
        HashTable::Iterator* iter =
                HashTable::Iterator::create(*fSubChannelHashTable);
        RTPInterface* rtpInterface;
        char const* key;

        while ((rtpInterface = (RTPInterface*)(iter->next(key))) != NULL) {
            u_int64_t streamChannelIdLong = (u_int64_t)key;
            unsigned char streamChannelId = (unsigned char)streamChannelIdLong;

            rtpInterface->removeStreamSocket(fOurSocketNum, streamChannelId);
        }
        delete iter;

        // Then remove the hash table entries themselves, and then remove the
        // hash table:
        while (fSubChannelHashTable->RemoveNext() != NULL) {
        }
        delete fSubChannelHashTable;
    }

    // Finally:
    if (fServerRequestAlternativeByteHandler != NULL) {
        // Hack: Pass a special character to our alternative byte handler, to
        // tell it that either
        // - an error occurred when reading the TCP socket, or
        // - no error occurred, but it needs to take over control of the TCP
        // socket once again.
        u_int8_t specialChar = fReadErrorOccurred ? 0xFF : 0xFE;
        (*fServerRequestAlternativeByteHandler)(
                fServerRequestAlternativeByteHandlerClientData, specialChar);
    }
}

void SocketDescriptor::registerRTPInterface(unsigned char streamChannelId,
                                            RTPInterface* rtpInterface) {
    Boolean isFirstRegistration = fSubChannelHashTable->IsEmpty();
#if defined(DEBUG_SEND) || defined(DEBUG_RECEIVE)
    fprintf(stderr,
            "SocketDescriptor(socket %d)::registerRTPInterface(channel %d): "
            "isFirstRegistration %d\n",
            fOurSocketNum, streamChannelId, isFirstRegistration);
#endif
    fSubChannelHashTable->Add((char const*)(long)streamChannelId, rtpInterface);

    if (isFirstRegistration) {
        // Arrange to handle reads on this TCP socket:
        TaskScheduler::BackgroundHandlerProc* handler =
                (TaskScheduler::BackgroundHandlerProc*)&tcpReadHandler;
        fEnv.taskScheduler().setBackgroundHandling(
                fOurSocketNum, SOCKET_READABLE | SOCKET_EXCEPTION, handler,
                this);
    }
}

RTPInterface* SocketDescriptor ::lookupRTPInterface(
        unsigned char streamChannelId) {
    char const* lookupArg = (char const*)(long)streamChannelId;
    return (RTPInterface*)(fSubChannelHashTable->Lookup(lookupArg));
}

void SocketDescriptor ::deregisterRTPInterface(unsigned char streamChannelId) {
#if defined(DEBUG_SEND) || defined(DEBUG_RECEIVE)
    fprintf(stderr,
            "SocketDescriptor(socket %d)::deregisterRTPInterface(channel %d)\n",
            fOurSocketNum, streamChannelId);
#endif
    fSubChannelHashTable->Remove((char const*)(long)streamChannelId);

    if (fSubChannelHashTable->IsEmpty()) {
        // No more interfaces are using us, so it's curtains for us now:
        if (fAreInReadHandlerLoop) {
            fDeleteMyselfNext = True;  // we can't delete ourself yet, but we'll
                                       // do so from "tcpReadHandler()" below
        } else {
            delete this;
        }
    }
}

void SocketDescriptor::tcpReadHandler(SocketDescriptor* socketDescriptor,
                                      int mask) {
    // Call the read handler until it returns false, with a limit to avoid
    // starving other sockets
    unsigned count = 2000;
    socketDescriptor->fAreInReadHandlerLoop = True;
    while (!socketDescriptor->fDeleteMyselfNext &&
           socketDescriptor->tcpReadHandler1(mask) && --count > 0) {
    }
    socketDescriptor->fAreInReadHandlerLoop = False;
    if (socketDescriptor->fDeleteMyselfNext) delete socketDescriptor;
}

Boolean SocketDescriptor::tcpReadHandler1(int mask) {
    // We expect the following data over the TCP channel:
    //   optional RTSP command or response bytes (before the first '$'
    //   character) a '$' character a 1-byte channel id a 2-byte packet size (in
    //   network byte order) the packet data.
    // However, because the socket is being read asynchronously, this data might
    // arrive in pieces.

    u_int8_t c;
    struct sockaddr_in fromAddress;
    if (fTCPReadingState != AWAITING_PACKET_DATA) {
        int result = readSocket(fEnv, fOurSocketNum, &c, 1, fromAddress);
        if (result == 0) {  // There was no more data to read
            return False;
        } else if (result != 1) {  // error reading TCP socket, so we will no
                                   // longer handle it
#ifdef DEBUG_RECEIVE
            fprintf(stderr,
                    "SocketDescriptor(socket %d)::tcpReadHandler(): "
                    "readSocket(1 byte) returned %d (error)\n",
                    fOurSocketNum, result);
#endif
            fReadErrorOccurred = True;
            fDeleteMyselfNext = True;
            return False;
        }
    }

    Boolean callAgain = True;
    switch (fTCPReadingState) {
        case AWAITING_DOLLAR: {
            if (c == '$') {
#ifdef DEBUG_RECEIVE
                fprintf(stderr,
                        "SocketDescriptor(socket %d)::tcpReadHandler(): Saw "
                        "'$'\n",
                        fOurSocketNum);
#endif
                fTCPReadingState = AWAITING_STREAM_CHANNEL_ID;
            } else {
                // This character is part of a RTSP request or command, which is
                // handled separately:
                if (fServerRequestAlternativeByteHandler != NULL && c != 0xFF &&
                    c != 0xFE) {
                    // Hack: 0xFF and 0xFE are used as special signaling
                    // characters, so don't send them
                    (*fServerRequestAlternativeByteHandler)(
                            fServerRequestAlternativeByteHandlerClientData, c);
                }
            }
            break;
        }
        case AWAITING_STREAM_CHANNEL_ID: {
            // The byte that we read is the stream channel id.
            if (lookupRTPInterface(c) != NULL) {  // sanity check
                fStreamChannelId = c;
                fTCPReadingState = AWAITING_SIZE1;
            } else {
                // This wasn't a stream channel id that we expected.  We're
                // (somehow) in a strange state.  Try to recover:
#ifdef DEBUG_RECEIVE
                fprintf(stderr,
                        "SocketDescriptor(socket %d)::tcpReadHandler(): Saw "
                        "nonexistent stream channel id: 0x%02x\n",
                        fOurSocketNum, c);
#endif
                fTCPReadingState = AWAITING_DOLLAR;
            }
            break;
        }
        case AWAITING_SIZE1: {
            // The byte that we read is the first (high) byte of the 16-bit RTP
            // or RTCP packet 'size'.
            fSizeByte1 = c;
            fTCPReadingState = AWAITING_SIZE2;
            break;
        }
        case AWAITING_SIZE2: {
            // The byte that we read is the second (low) byte of the 16-bit RTP
            // or RTCP packet 'size'.
            unsigned short size = (fSizeByte1 << 8) | c;

            // Record the information about the packet data that will be read
            // next:
            RTPInterface* rtpInterface = lookupRTPInterface(fStreamChannelId);
            if (rtpInterface != NULL) {
                rtpInterface->fNextTCPReadSize = size;
                rtpInterface->fNextTCPReadStreamSocketNum = fOurSocketNum;
                rtpInterface->fNextTCPReadStreamChannelId = fStreamChannelId;
            }
            fTCPReadingState = AWAITING_PACKET_DATA;
            break;
        }
        case AWAITING_PACKET_DATA: {
            callAgain = False;
            fTCPReadingState =
                    AWAITING_DOLLAR;  // the next state, unless we end up having
                                      // to read more data in the current state
            // Call the appropriate read handler to get the packet data from the
            // TCP stream:
            RTPInterface* rtpInterface = lookupRTPInterface(fStreamChannelId);
            if (rtpInterface != NULL) {
                if (rtpInterface->fNextTCPReadSize == 0) {
                    // We've already read all the data for this packet.
                    break;
                }
                if (rtpInterface->fReadHandlerProc != NULL) {
#ifdef DEBUG_RECEIVE
                    fprintf(stderr,
                            "SocketDescriptor(socket %d)::tcpReadHandler(): "
                            "reading %d bytes on channel %d\n",
                            fOurSocketNum, rtpInterface->fNextTCPReadSize,
                            rtpInterface->fNextTCPReadStreamChannelId);
#endif
                    fTCPReadingState = AWAITING_PACKET_DATA;
                    rtpInterface->fReadHandlerProc(rtpInterface->fOwner, mask);
                } else {
#ifdef DEBUG_RECEIVE
                    fprintf(stderr,
                            "SocketDescriptor(socket %d)::tcpReadHandler(): No "
                            "handler proc for \"rtpInterface\" for channel %d; "
                            "need to skip %d remaining bytes\n",
                            fOurSocketNum, fStreamChannelId,
                            rtpInterface->fNextTCPReadSize);
#endif
                    int result =
                            readSocket(fEnv, fOurSocketNum, &c, 1, fromAddress);
                    if (result < 0) {  // error reading TCP socket, so we will
                                       // no longer handle it
#ifdef DEBUG_RECEIVE
                        fprintf(stderr,
                                "SocketDescriptor(socket "
                                "%d)::tcpReadHandler(): readSocket(1 byte) "
                                "returned %d (error)\n",
                                fOurSocketNum, result);
#endif
                        fReadErrorOccurred = True;
                        fDeleteMyselfNext = True;
                        return False;
                    } else {
                        fTCPReadingState = AWAITING_PACKET_DATA;
                        if (result == 1) {
                            --rtpInterface->fNextTCPReadSize;
                            callAgain = True;
                        }
                    }
                }
            }
#ifdef DEBUG_RECEIVE
            else
                fprintf(stderr,
                        "SocketDescriptor(socket %d)::tcpReadHandler(): No "
                        "\"rtpInterface\" for channel %d\n",
                        fOurSocketNum, fStreamChannelId);
#endif
        }
    }

    return callAgain;
}

////////// tcpStreamRecord implementation //////////

tcpStreamRecord ::tcpStreamRecord(int streamSocketNum,
                                  unsigned char streamChannelId,
                                  tcpStreamRecord* next)
    : fNext(next),
      fStreamSocketNum(streamSocketNum),
      fStreamChannelId(streamChannelId) {}

tcpStreamRecord::~tcpStreamRecord() { delete fNext; }
