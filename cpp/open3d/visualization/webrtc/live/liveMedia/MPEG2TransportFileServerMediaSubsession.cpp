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
// A 'ServerMediaSubsession' object that creates new, unicast, "RTPSink"s
// on demand, from a MPEG-2 Transport Stream file.
// Implementation

#include "MPEG2TransportFileServerMediaSubsession.hh"

#include "SimpleRTPSink.hh"

MPEG2TransportFileServerMediaSubsession*
MPEG2TransportFileServerMediaSubsession::createNew(UsageEnvironment& env,
                                                   char const* fileName,
                                                   char const* indexFileName,
                                                   Boolean reuseFirstSource) {
    MPEG2TransportStreamIndexFile* indexFile;
    if (indexFileName != NULL && reuseFirstSource) {
        // It makes no sense to support trick play if all clients use the same
        // source.  Fix this:
        env << "MPEG2TransportFileServerMediaSubsession::createNew(): ignoring "
               "the index file name, because \"reuseFirstSource\" is set\n";
        indexFile = NULL;
    } else {
        indexFile =
                MPEG2TransportStreamIndexFile::createNew(env, indexFileName);
    }
    return new MPEG2TransportFileServerMediaSubsession(env, fileName, indexFile,
                                                       reuseFirstSource);
}

MPEG2TransportFileServerMediaSubsession ::
        MPEG2TransportFileServerMediaSubsession(
                UsageEnvironment& env,
                char const* fileName,
                MPEG2TransportStreamIndexFile* indexFile,
                Boolean reuseFirstSource)
    : FileServerMediaSubsession(env, fileName, reuseFirstSource),
      fIndexFile(indexFile),
      fDuration(0.0),
      fClientSessionHashTable(NULL) {
    if (fIndexFile != NULL) {  // we support 'trick play'
        fDuration = fIndexFile->getPlayingDuration();
        fClientSessionHashTable = HashTable::create(ONE_WORD_HASH_KEYS);
    }
}

MPEG2TransportFileServerMediaSubsession ::
        ~MPEG2TransportFileServerMediaSubsession() {
    if (fIndexFile != NULL) {  // we support 'trick play'
        Medium::close(fIndexFile);

        // Clean out the client session hash table:
        while (1) {
            ClientTrickPlayState* client =
                    (ClientTrickPlayState*)(fClientSessionHashTable
                                                    ->RemoveNext());
            if (client == NULL) break;
            delete client;
        }
        delete fClientSessionHashTable;
    }
}

#define TRANSPORT_PACKET_SIZE 188
#define TRANSPORT_PACKETS_PER_NETWORK_PACKET 7
// The product of these two numbers must be enough to fit within a network
// packet

void MPEG2TransportFileServerMediaSubsession ::startStream(
        unsigned clientSessionId,
        void* streamToken,
        TaskFunc* rtcpRRHandler,
        void* rtcpRRHandlerClientData,
        unsigned short& rtpSeqNum,
        unsigned& rtpTimestamp,
        ServerRequestAlternativeByteHandler*
                serverRequestAlternativeByteHandler,
        void* serverRequestAlternativeByteHandlerClientData) {
    if (fIndexFile != NULL) {  // we support 'trick play'
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client != NULL && client->areChangingScale()) {
            // First, handle this like a "PAUSE", except that we back up to the
            // previous VSH
            client->updateStateOnPlayChange(True);
            OnDemandServerMediaSubsession::pauseStream(clientSessionId,
                                                       streamToken);

            // Then, adjust for the change of scale:
            client->updateStateOnScaleChange();
        }
    }

    // Call the original, default version of this routine:
    OnDemandServerMediaSubsession::startStream(
            clientSessionId, streamToken, rtcpRRHandler,
            rtcpRRHandlerClientData, rtpSeqNum, rtpTimestamp,
            serverRequestAlternativeByteHandler,
            serverRequestAlternativeByteHandlerClientData);
}

void MPEG2TransportFileServerMediaSubsession ::pauseStream(
        unsigned clientSessionId, void* streamToken) {
    if (fIndexFile != NULL) {  // we support 'trick play'
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client != NULL) {
            client->updateStateOnPlayChange(False);
        }
    }

    // Call the original, default version of this routine:
    OnDemandServerMediaSubsession::pauseStream(clientSessionId, streamToken);
}

void MPEG2TransportFileServerMediaSubsession ::seekStream(
        unsigned clientSessionId,
        void* streamToken,
        double& seekNPT,
        double streamDuration,
        u_int64_t& numBytes) {
    // Begin by calling the original, default version of this routine:
    OnDemandServerMediaSubsession::seekStream(
            clientSessionId, streamToken, seekNPT, streamDuration, numBytes);

    // Then, special handling specific to indexed Transport Stream files:
    if (fIndexFile != NULL) {  // we support 'trick play'
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client != NULL) {
            unsigned long numTSPacketsToStream =
                    client->updateStateFromNPT(seekNPT, streamDuration);
            numBytes = numTSPacketsToStream * TRANSPORT_PACKET_SIZE;
        }
    }
}

void MPEG2TransportFileServerMediaSubsession ::setStreamScale(
        unsigned clientSessionId, void* streamToken, float scale) {
    if (fIndexFile != NULL) {  // we support 'trick play'
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client != NULL) {
            client->setNextScale(
                    scale);  // scale won't take effect until the next "PLAY"
        }
    }

    // Call the original, default version of this routine:
    OnDemandServerMediaSubsession::setStreamScale(clientSessionId, streamToken,
                                                  scale);
}

void MPEG2TransportFileServerMediaSubsession ::deleteStream(
        unsigned clientSessionId, void*& streamToken) {
    if (fIndexFile != NULL) {  // we support 'trick play'
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client != NULL) {
            client->updateStateOnPlayChange(False);
        }
    }

    // Call the original, default version of this routine:
    OnDemandServerMediaSubsession::deleteStream(clientSessionId, streamToken);
}

ClientTrickPlayState*
MPEG2TransportFileServerMediaSubsession::newClientTrickPlayState() {
    return new ClientTrickPlayState(fIndexFile);
}

FramedSource* MPEG2TransportFileServerMediaSubsession ::createNewStreamSource(
        unsigned clientSessionId, unsigned& estBitrate) {
    // Create the video source:
    unsigned const inputDataChunkSize =
            TRANSPORT_PACKETS_PER_NETWORK_PACKET * TRANSPORT_PACKET_SIZE;
    ByteStreamFileSource* fileSource = ByteStreamFileSource::createNew(
            envir(), fFileName, inputDataChunkSize);
    if (fileSource == NULL) return NULL;
    fFileSize = fileSource->fileSize();

    // Use the file size and the duration to estimate the stream's bitrate:
    if (fFileSize > 0 && fDuration > 0.0) {
        estBitrate = (unsigned)((int64_t)fFileSize / (125 * fDuration) +
                                0.5);  // kbps, rounded
    } else {
        estBitrate = 5000;  // kbps, estimate
    }

    // Create a framer for the Transport Stream:
    MPEG2TransportStreamFramer* framer =
            MPEG2TransportStreamFramer::createNew(envir(), fileSource);

    if (fIndexFile != NULL) {  // we support 'trick play'
        // Keep state for this client (if we don't already have it):
        ClientTrickPlayState* client = lookupClient(clientSessionId);
        if (client == NULL) {
            client = newClientTrickPlayState();
            fClientSessionHashTable->Add(
                    reinterpret_cast<char const*>(clientSessionId), client);
        }
        client->setSource(framer);
    }

    return framer;
}

RTPSink* MPEG2TransportFileServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char /*rtpPayloadTypeIfDynamic*/,
        FramedSource* /*inputSource*/) {
    return SimpleRTPSink::createNew(envir(), rtpGroupsock, 33, 90000, "video",
                                    "MP2T", 1, True, False /*no 'M' bit*/);
}

void MPEG2TransportFileServerMediaSubsession::testScaleFactor(float& scale) {
    if (fIndexFile != NULL && fDuration > 0.0) {
        // We support any integral scale, other than 0
        int iScale = scale < 0.0 ? (int)(scale - 0.5f)
                                 : (int)(scale + 0.5f);  // round
        if (iScale == 0) iScale = 1;
        scale = (float)iScale;
    } else {
        scale = 1.0f;
    }
}

float MPEG2TransportFileServerMediaSubsession::duration() const {
    return fDuration;
}

ClientTrickPlayState* MPEG2TransportFileServerMediaSubsession ::lookupClient(
        unsigned clientSessionId) {
    return (ClientTrickPlayState*)(fClientSessionHashTable->Lookup(
            reinterpret_cast<char const*>(clientSessionId)));
}

////////// ClientTrickPlayState implementation //////////

ClientTrickPlayState::ClientTrickPlayState(
        MPEG2TransportStreamIndexFile* indexFile)
    : fIndexFile(indexFile),
      fOriginalTransportStreamSource(NULL),
      fTrickModeFilter(NULL),
      fTrickPlaySource(NULL),
      fFramer(NULL),
      fScale(1.0f),
      fNextScale(1.0f),
      fNPT(0.0f),
      fTSRecordNum(0),
      fIxRecordNum(0) {}

unsigned long ClientTrickPlayState::updateStateFromNPT(double npt,
                                                       double streamDuration) {
    fNPT = (float)npt;
    // Map "fNPT" to the corresponding Transport Stream and Index record
    // numbers:
    unsigned long tsRecordNum, ixRecordNum;
    fIndexFile->lookupTSPacketNumFromNPT(fNPT, tsRecordNum, ixRecordNum);

    updateTSRecordNum();
    if (tsRecordNum != fTSRecordNum) {
        fTSRecordNum = tsRecordNum;
        fIxRecordNum = ixRecordNum;

        // Seek the source to the new record number:
        reseekOriginalTransportStreamSource();
        // Note: We assume that we're asked to seek only in normal
        // (i.e., non trick play) mode, so we don't seek within the trick
        // play source (if any).

        fFramer->clearPIDStatusTable();
    }

    unsigned long numTSRecordsToStream = 0;
    float pcrLimit = 0.0;
    if (streamDuration > 0.0) {
        // fNPT might have changed when we looked it up in the index file.
        // Adjust "streamDuration" accordingly:
        streamDuration += npt - (double)fNPT;

        if (streamDuration > 0.0) {
            // Specify that we want to stream no more data than this.

            if (fNextScale == 1.0f) {
                // We'll be streaming from the original file.
                // Use the index file to figure out how many Transport Packets
                // we get to stream:
                unsigned long toTSRecordNum, toIxRecordNum;
                float toNPT = (float)(fNPT + streamDuration);
                fIndexFile->lookupTSPacketNumFromNPT(toNPT, toTSRecordNum,
                                                     toIxRecordNum);
                if (toTSRecordNum > tsRecordNum) {  // sanity check
                    numTSRecordsToStream = toTSRecordNum - tsRecordNum;
                }
            } else {
                // We'll be streaming from the trick play stream.
                // It'd be difficult to figure out how many Transport Packets we
                // need to stream, so instead set a PCR limit in the trick play
                // stream.  (We rely upon the fact that PCRs in the trick play
                // stream start at 0.0)
                int direction = fNextScale < 0.0 ? -1 : 1;
                pcrLimit = (float)(streamDuration / (fNextScale * direction));
            }
        }
    }
    fFramer->setNumTSPacketsToStream(numTSRecordsToStream);
    fFramer->setPCRLimit(pcrLimit);

    return numTSRecordsToStream;
}

void ClientTrickPlayState::updateStateOnScaleChange() {
    fScale = fNextScale;

    // Change our source objects to reflect the change in scale:
    // First, close the existing trick play source (if any):
    if (fTrickPlaySource != NULL) {
        fTrickModeFilter->forgetInputSource();
        // so that the underlying Transport Stream source doesn't get deleted
        // by:
        Medium::close(fTrickPlaySource);
        fTrickPlaySource = NULL;
        fTrickModeFilter = NULL;
    }
    if (fNextScale != 1.0f) {
        // Create a new trick play filter from the original Transport Stream
        // source:
        UsageEnvironment& env = fIndexFile->envir();  // alias
        fTrickModeFilter = MPEG2TransportStreamTrickModeFilter ::createNew(
                env, fOriginalTransportStreamSource, fIndexFile,
                int(fNextScale));
        fTrickModeFilter->seekTo(fTSRecordNum, fIxRecordNum);

        // And generate a Transport Stream from this:
        fTrickPlaySource = MPEG2TransportStreamFromESSource::createNew(env);
        fTrickPlaySource->addNewVideoSource(fTrickModeFilter,
                                            fIndexFile->mpegVersion());

        fFramer->changeInputSource(fTrickPlaySource);
    } else {
        // Switch back to the original Transport Stream source:
        reseekOriginalTransportStreamSource();
        fFramer->changeInputSource(fOriginalTransportStreamSource);
    }
}

void ClientTrickPlayState::updateStateOnPlayChange(
        Boolean reverseToPreviousVSH) {
    updateTSRecordNum();
    if (fTrickPlaySource == NULL) {
        // We were in regular (1x) play. Use the index file to look up the
        // index record number and npt from the current transport number:
        fIndexFile->lookupPCRFromTSPacketNum(fTSRecordNum, reverseToPreviousVSH,
                                             fNPT, fIxRecordNum);
    } else {
        // We were in trick mode, and so already have the index record number.
        // Get the transport record number and npt from this:
        fIxRecordNum = fTrickModeFilter->nextIndexRecordNum();
        if ((long)fIxRecordNum < 0)
            fIxRecordNum = 0;  // we were at the start of the file
        unsigned long transportRecordNum;
        float pcr;
        u_int8_t offset, size, recordType;  // all dummy
        if (fIndexFile->readIndexRecordValues(fIxRecordNum, transportRecordNum,
                                              offset, size, pcr, recordType)) {
            fTSRecordNum = transportRecordNum;
            fNPT = pcr;
        }
    }
}

void ClientTrickPlayState::setSource(MPEG2TransportStreamFramer* framer) {
    fFramer = framer;
    fOriginalTransportStreamSource =
            (ByteStreamFileSource*)(framer->inputSource());
}

void ClientTrickPlayState::updateTSRecordNum() {
    if (fFramer != NULL)
        fTSRecordNum += (unsigned long)(fFramer->tsPacketCount());
}

void ClientTrickPlayState::reseekOriginalTransportStreamSource() {
    u_int64_t tsRecordNum64 = (u_int64_t)fTSRecordNum;
    fOriginalTransportStreamSource->seekToByteAbsolute(tsRecordNum64 *
                                                       TRANSPORT_PACKET_SIZE);
}
