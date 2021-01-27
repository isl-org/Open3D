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
// on demand, from a MP3 audio file.
// (Actually, any MPEG-1 or MPEG-2 audio file should work.)
// Implementation

#include "MP3AudioFileServerMediaSubsession.hh"

#include "MP3ADU.hh"
#include "MP3ADURTPSink.hh"
#include "MP3FileSource.hh"
#include "MPEG1or2AudioRTPSink.hh"

MP3AudioFileServerMediaSubsession*
MP3AudioFileServerMediaSubsession ::createNew(UsageEnvironment& env,
                                              char const* fileName,
                                              Boolean reuseFirstSource,
                                              Boolean generateADUs,
                                              Interleaving* interleaving) {
    return new MP3AudioFileServerMediaSubsession(
            env, fileName, reuseFirstSource, generateADUs, interleaving);
}

MP3AudioFileServerMediaSubsession ::MP3AudioFileServerMediaSubsession(
        UsageEnvironment& env,
        char const* fileName,
        Boolean reuseFirstSource,
        Boolean generateADUs,
        Interleaving* interleaving)
    : FileServerMediaSubsession(env, fileName, reuseFirstSource),
      fGenerateADUs(generateADUs),
      fInterleaving(interleaving),
      fFileDuration(0.0) {}

MP3AudioFileServerMediaSubsession ::~MP3AudioFileServerMediaSubsession() {
    delete fInterleaving;
}

FramedSource* MP3AudioFileServerMediaSubsession ::createNewStreamSourceCommon(
        FramedSource* baseMP3Source,
        unsigned mp3NumBytes,
        unsigned& estBitrate) {
    FramedSource* streamSource;
    do {
        streamSource = baseMP3Source;  // by default
        if (streamSource == NULL) break;

        // Use the MP3 file size, plus the duration, to estimate the stream's
        // bitrate:
        if (mp3NumBytes > 0 && fFileDuration > 0.0) {
            estBitrate = (unsigned)(mp3NumBytes / (125 * fFileDuration) +
                                    0.5);  // kbps, rounded
        } else {
            estBitrate = 128;  // kbps, estimate
        }

        if (fGenerateADUs) {
            // Add a filter that converts the source MP3s to ADUs:
            streamSource = ADUFromMP3Source::createNew(envir(), streamSource);
            if (streamSource == NULL) break;

            if (fInterleaving != NULL) {
                // Add another filter that interleaves the ADUs before
                // packetizing:
                streamSource = MP3ADUinterleaver::createNew(
                        envir(), *fInterleaving, streamSource);
                if (streamSource == NULL) break;
            }
        } else if (fFileDuration > 0.0) {
            // Because this is a seekable file, insert a pair of filters: one
            // that converts the input MP3 stream to ADUs; another that converts
            // these ADUs back to MP3.  This allows us to seek within the input
            // stream without tripping over the MP3 'bit reservoir':
            streamSource = ADUFromMP3Source::createNew(envir(), streamSource);
            if (streamSource == NULL) break;

            streamSource = MP3FromADUSource::createNew(envir(), streamSource);
            if (streamSource == NULL) break;
        }
    } while (0);

    return streamSource;
}

void MP3AudioFileServerMediaSubsession::getBaseStreams(
        FramedSource* frontStream,
        FramedSource*& sourceMP3Stream,
        ADUFromMP3Source*& aduStream /*if any*/) {
    if (fGenerateADUs) {
        // There's an ADU stream.
        if (fInterleaving != NULL) {
            // There's an interleaving filter in front of the ADU stream.  So go
            // back one, to reach the ADU stream:
            aduStream = (ADUFromMP3Source*)(((FramedFilter*)frontStream)
                                                    ->inputSource());
        } else {
            aduStream = (ADUFromMP3Source*)frontStream;
        }

        // Then, go back one more, to reach the MP3 source:
        sourceMP3Stream = (MP3FileSource*)(aduStream->inputSource());
    } else if (fFileDuration > 0.0) {
        // There are a pair of filters - MP3->ADU and ADU->MP3 - in front of the
        // original MP3 source.  So, go back one, to reach the ADU source:
        aduStream = (ADUFromMP3Source*)(((FramedFilter*)frontStream)
                                                ->inputSource());

        // Then, go back one more, to reach the MP3 source:
        sourceMP3Stream = (MP3FileSource*)(aduStream->inputSource());
    } else {
        // There's no filter in front of the source MP3 stream (and there's no
        // ADU stream):
        aduStream = NULL;
        sourceMP3Stream = frontStream;
    }
}

void MP3AudioFileServerMediaSubsession ::seekStreamSource(
        FramedSource* inputSource,
        double& seekNPT,
        double streamDuration,
        u_int64_t& /*numBytes*/) {
    FramedSource* sourceMP3Stream;
    ADUFromMP3Source* aduStream;
    getBaseStreams(inputSource, sourceMP3Stream, aduStream);

    if (aduStream != NULL)
        aduStream->resetInput();  // because we're about to seek within its
                                  // source
    ((MP3FileSource*)sourceMP3Stream)->seekWithinFile(seekNPT, streamDuration);
}

void MP3AudioFileServerMediaSubsession ::setStreamSourceScale(
        FramedSource* inputSource, float scale) {
    FramedSource* sourceMP3Stream;
    ADUFromMP3Source* aduStream;
    getBaseStreams(inputSource, sourceMP3Stream, aduStream);

    if (aduStream == NULL)
        return;  // because, in this case, the stream's not scalable

    int iScale = (int)scale;
    aduStream->setScaleFactor(iScale);
    ((MP3FileSource*)sourceMP3Stream)->setPresentationTimeScale(iScale);
}

FramedSource* MP3AudioFileServerMediaSubsession ::createNewStreamSource(
        unsigned /*clientSessionId*/, unsigned& estBitrate) {
    MP3FileSource* mp3Source = MP3FileSource::createNew(envir(), fFileName);
    if (mp3Source == NULL) return NULL;
    fFileDuration = mp3Source->filePlayTime();

    return createNewStreamSourceCommon(mp3Source, mp3Source->fileSize(),
                                       estBitrate);
}

RTPSink* MP3AudioFileServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic,
        FramedSource* /*inputSource*/) {
    if (fGenerateADUs) {
        return MP3ADURTPSink::createNew(envir(), rtpGroupsock,
                                        rtpPayloadTypeIfDynamic);
    } else {
        return MPEG1or2AudioRTPSink::createNew(envir(), rtpGroupsock);
    }
}

void MP3AudioFileServerMediaSubsession::testScaleFactor(float& scale) {
    if (fFileDuration <= 0.0) {
        // The file is non-seekable, so is probably a live input source.
        // We don't support scale factors other than 1
        scale = 1;
    } else {
        // We support any integral scale >= 1
        int iScale = (int)(scale + 0.5);  // round
        if (iScale < 1) iScale = 1;
        scale = (float)iScale;
    }
}

float MP3AudioFileServerMediaSubsession::duration() const {
    return fFileDuration;
}
