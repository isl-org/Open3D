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
// on demand, from an WAV audio file.
// Implementation

#include "WAVAudioFileServerMediaSubsession.hh"

#include "SimpleRTPSink.hh"
#include "WAVAudioFileSource.hh"
#include "uLawAudioFilter.hh"

WAVAudioFileServerMediaSubsession*
WAVAudioFileServerMediaSubsession ::createNew(UsageEnvironment& env,
                                              char const* fileName,
                                              Boolean reuseFirstSource,
                                              Boolean convertToULaw) {
    return new WAVAudioFileServerMediaSubsession(
            env, fileName, reuseFirstSource, convertToULaw);
}

WAVAudioFileServerMediaSubsession ::WAVAudioFileServerMediaSubsession(
        UsageEnvironment& env,
        char const* fileName,
        Boolean reuseFirstSource,
        Boolean convertToULaw)
    : FileServerMediaSubsession(env, fileName, reuseFirstSource),
      fConvertToULaw(convertToULaw) {}

WAVAudioFileServerMediaSubsession ::~WAVAudioFileServerMediaSubsession() {}

void WAVAudioFileServerMediaSubsession ::seekStreamSource(
        FramedSource* inputSource,
        double& seekNPT,
        double streamDuration,
        u_int64_t& numBytes) {
    WAVAudioFileSource* wavSource;
    if (fBitsPerSample > 8) {
        // "inputSource" is a filter; its input source is the original WAV file
        // source:
        wavSource = (WAVAudioFileSource*)(((FramedFilter*)inputSource)
                                                  ->inputSource());
    } else {
        // "inputSource" is the original WAV file source:
        wavSource = (WAVAudioFileSource*)inputSource;
    }

    unsigned seekSampleNumber = (unsigned)(seekNPT * fSamplingFrequency);
    unsigned seekByteNumber =
            seekSampleNumber * ((fNumChannels * fBitsPerSample) / 8);

    wavSource->seekToPCMByte(seekByteNumber);

    setStreamSourceDuration(inputSource, streamDuration, numBytes);
}

void WAVAudioFileServerMediaSubsession ::setStreamSourceDuration(
        FramedSource* inputSource, double streamDuration, u_int64_t& numBytes) {
    WAVAudioFileSource* wavSource;
    if (fBitsPerSample > 8) {
        // "inputSource" is a filter; its input source is the original WAV file
        // source:
        wavSource = (WAVAudioFileSource*)(((FramedFilter*)inputSource)
                                                  ->inputSource());
    } else {
        // "inputSource" is the original WAV file source:
        wavSource = (WAVAudioFileSource*)inputSource;
    }

    unsigned numDurationSamples =
            (unsigned)(streamDuration * fSamplingFrequency);
    unsigned numDurationBytes =
            numDurationSamples * ((fNumChannels * fBitsPerSample) / 8);
    numBytes = (u_int64_t)numDurationBytes;

    wavSource->limitNumBytesToStream(numDurationBytes);
}

void WAVAudioFileServerMediaSubsession ::setStreamSourceScale(
        FramedSource* inputSource, float scale) {
    int iScale = (int)scale;
    WAVAudioFileSource* wavSource;
    if (fBitsPerSample > 8) {
        // "inputSource" is a filter; its input source is the original WAV file
        // source:
        wavSource = (WAVAudioFileSource*)(((FramedFilter*)inputSource)
                                                  ->inputSource());
    } else {
        // "inputSource" is the original WAV file source:
        wavSource = (WAVAudioFileSource*)inputSource;
    }

    wavSource->setScaleFactor(iScale);
}

FramedSource* WAVAudioFileServerMediaSubsession ::createNewStreamSource(
        unsigned /*clientSessionId*/, unsigned& estBitrate) {
    FramedSource* resultSource = NULL;
    do {
        WAVAudioFileSource* wavSource =
                WAVAudioFileSource::createNew(envir(), fFileName);
        if (wavSource == NULL) break;

        // Get attributes of the audio source:

        fAudioFormat = wavSource->getAudioFormat();
        fBitsPerSample = wavSource->bitsPerSample();
        // We handle only 4,8,16,20,24 bits-per-sample audio:
        if (fBitsPerSample % 4 != 0 || fBitsPerSample < 4 ||
            fBitsPerSample > 24 || fBitsPerSample == 12) {
            envir() << "The input file contains " << fBitsPerSample
                    << " bit-per-sample audio, which we don't handle\n";
            break;
        }
        fSamplingFrequency = wavSource->samplingFrequency();
        fNumChannels = wavSource->numChannels();
        unsigned bitsPerSecond =
                fSamplingFrequency * fBitsPerSample * fNumChannels;

        fFileDuration =
                (float)((8.0 * wavSource->numPCMBytes()) /
                        (fSamplingFrequency * fNumChannels * fBitsPerSample));

        // Add in any filter necessary to transform the data prior to streaming:
        resultSource = wavSource;  // by default
        if (fAudioFormat == WA_PCM) {
            if (fBitsPerSample == 16) {
                // Note that samples in the WAV audio file are in little-endian
                // order.
                if (fConvertToULaw) {
                    // Add a filter that converts from raw 16-bit PCM audio to
                    // 8-bit u-law audio:
                    resultSource = uLawFromPCMAudioSource::createNew(
                            envir(), wavSource, 1 /*little-endian*/);
                    bitsPerSecond /= 2;
                } else {
                    // Add a filter that converts from little-endian to network
                    // (big-endian) order:
                    resultSource = EndianSwap16::createNew(envir(), wavSource);
                }
            } else if (fBitsPerSample == 20 || fBitsPerSample == 24) {
                // Add a filter that converts from little-endian to network
                // (big-endian) order:
                resultSource = EndianSwap24::createNew(envir(), wavSource);
            }
        }

        estBitrate = (bitsPerSecond + 500) / 1000;  // kbps
        return resultSource;
    } while (0);

    // An error occurred:
    Medium::close(resultSource);
    return NULL;
}

RTPSink* WAVAudioFileServerMediaSubsession ::createNewRTPSink(
        Groupsock* rtpGroupsock,
        unsigned char rtpPayloadTypeIfDynamic,
        FramedSource* /*inputSource*/) {
    do {
        char const* mimeType;
        unsigned char payloadFormatCode =
                rtpPayloadTypeIfDynamic;  // by default, unless a static RTP
                                          // payload type can be used
        if (fAudioFormat == WA_PCM) {
            if (fBitsPerSample == 16) {
                if (fConvertToULaw) {
                    mimeType = "PCMU";
                    if (fSamplingFrequency == 8000 && fNumChannels == 1) {
                        payloadFormatCode = 0;  // a static RTP payload type
                    }
                } else {
                    mimeType = "L16";
                    if (fSamplingFrequency == 44100 && fNumChannels == 2) {
                        payloadFormatCode = 10;  // a static RTP payload type
                    } else if (fSamplingFrequency == 44100 &&
                               fNumChannels == 1) {
                        payloadFormatCode = 11;  // a static RTP payload type
                    }
                }
            } else if (fBitsPerSample == 20) {
                mimeType = "L20";
            } else if (fBitsPerSample == 24) {
                mimeType = "L24";
            } else {  // fBitsPerSample == 8 (we assume that fBitsPerSample == 4
                      // is only for WA_IMA_ADPCM)
                mimeType = "L8";
            }
        } else if (fAudioFormat == WA_PCMU) {
            mimeType = "PCMU";
            if (fSamplingFrequency == 8000 && fNumChannels == 1) {
                payloadFormatCode = 0;  // a static RTP payload type
            }
        } else if (fAudioFormat == WA_PCMA) {
            mimeType = "PCMA";
            if (fSamplingFrequency == 8000 && fNumChannels == 1) {
                payloadFormatCode = 8;  // a static RTP payload type
            }
        } else if (fAudioFormat == WA_IMA_ADPCM) {
            mimeType = "DVI4";
            // Use a static payload type, if one is defined:
            if (fNumChannels == 1) {
                if (fSamplingFrequency == 8000) {
                    payloadFormatCode = 5;  // a static RTP payload type
                } else if (fSamplingFrequency == 16000) {
                    payloadFormatCode = 6;  // a static RTP payload type
                } else if (fSamplingFrequency == 11025) {
                    payloadFormatCode = 16;  // a static RTP payload type
                } else if (fSamplingFrequency == 22050) {
                    payloadFormatCode = 17;  // a static RTP payload type
                }
            }
        } else {  // unknown format
            break;
        }

        return SimpleRTPSink::createNew(envir(), rtpGroupsock,
                                        payloadFormatCode, fSamplingFrequency,
                                        "audio", mimeType, fNumChannels);
    } while (0);

    // An error occurred:
    return NULL;
}

void WAVAudioFileServerMediaSubsession::testScaleFactor(float& scale) {
    if (fFileDuration <= 0.0) {
        // The file is non-seekable, so is probably a live input source.
        // We don't support scale factors other than 1
        scale = 1;
    } else {
        // We support any integral scale, other than 0
        int iScale =
                scale < 0.0 ? (int)(scale - 0.5) : (int)(scale + 0.5);  // round
        if (iScale == 0) iScale = 1;
        scale = (float)iScale;
    }
}

float WAVAudioFileServerMediaSubsession::duration() const {
    return fFileDuration;
}
