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
// A WAV audio file source
// Implementation

#include "WAVAudioFileSource.hh"

#include "GroupsockHelper.hh"
#include "InputFile.hh"

////////// WAVAudioFileSource //////////

WAVAudioFileSource* WAVAudioFileSource::createNew(UsageEnvironment& env,
                                                  char const* fileName) {
    do {
        FILE* fid = OpenInputFile(env, fileName);
        if (fid == NULL) break;

        WAVAudioFileSource* newSource = new WAVAudioFileSource(env, fid);
        if (newSource != NULL && newSource->bitsPerSample() == 0) {
            // The WAV file header was apparently invalid.
            Medium::close(newSource);
            break;
        }

        newSource->fFileSize = (unsigned)GetFileSize(fileName, fid);

        return newSource;
    } while (0);

    return NULL;
}

unsigned WAVAudioFileSource::numPCMBytes() const {
    if (fFileSize < fWAVHeaderSize) return 0;
    return fFileSize - fWAVHeaderSize;
}

void WAVAudioFileSource::setScaleFactor(int scale) {
    if (!fFidIsSeekable)
        return;  // we can't do 'trick play' operations on non-seekable files

    fScaleFactor = scale;

    if (fScaleFactor < 0 && TellFile64(fFid) > 0) {
        // Because we're reading backwards, seek back one sample, to ensure that
        // (i)  we start reading the last sample before the start point, and
        // (ii) we don't hit end-of-file on the first read.
        int bytesPerSample = (fNumChannels * fBitsPerSample) / 8;
        if (bytesPerSample == 0) bytesPerSample = 1;
        SeekFile64(fFid, -bytesPerSample, SEEK_CUR);
    }
}

void WAVAudioFileSource::seekToPCMByte(unsigned byteNumber) {
    byteNumber += fWAVHeaderSize;
    if (byteNumber > fFileSize) byteNumber = fFileSize;

    SeekFile64(fFid, byteNumber, SEEK_SET);
}

void WAVAudioFileSource::limitNumBytesToStream(unsigned numBytesToStream) {
    fNumBytesToStream = numBytesToStream;
    fLimitNumBytesToStream = fNumBytesToStream > 0;
}

unsigned char WAVAudioFileSource::getAudioFormat() { return fAudioFormat; }

#define nextc fgetc(fid)

static Boolean get4Bytes(FILE* fid, u_int32_t& result) {  // little-endian
    int c0, c1, c2, c3;
    if ((c0 = nextc) == EOF || (c1 = nextc) == EOF || (c2 = nextc) == EOF ||
        (c3 = nextc) == EOF)
        return False;
    result = (c3 << 24) | (c2 << 16) | (c1 << 8) | c0;
    return True;
}

static Boolean get2Bytes(FILE* fid, u_int16_t& result) {  // little-endian
    int c0, c1;
    if ((c0 = nextc) == EOF || (c1 = nextc) == EOF) return False;
    result = (c1 << 8) | c0;
    return True;
}

static Boolean skipBytes(FILE* fid, int num) {
    while (num-- > 0) {
        if (nextc == EOF) return False;
    }
    return True;
}

WAVAudioFileSource::WAVAudioFileSource(UsageEnvironment& env, FILE* fid)
    : AudioInputDevice(env, 0, 0, 0, 0) /* set the real parameters later */,
      fFid(fid),
      fFidIsSeekable(False),
      fLastPlayTime(0),
      fHaveStartedReading(False),
      fWAVHeaderSize(0),
      fFileSize(0),
      fScaleFactor(1),
      fLimitNumBytesToStream(False),
      fNumBytesToStream(0),
      fAudioFormat(WA_UNKNOWN) {
    // Check the WAV file header for validity.
    // Note: The following web pages contain info about the WAV format:
    // http://www.ringthis.com/dev/wave_format.htm
    // http://www.lightlink.com/tjweber/StripWav/Canon.html
    // http://www.onicos.com/staff/iz/formats/wav.html

    Boolean success = False;  // until we learn otherwise
    do {
        // RIFF Chunk:
        if (nextc != 'R' || nextc != 'I' || nextc != 'F' || nextc != 'F') break;
        if (!skipBytes(fid, 4)) break;
        if (nextc != 'W' || nextc != 'A' || nextc != 'V' || nextc != 'E') break;

        // Skip over any chunk that's not a FORMAT ('fmt ') chunk:
        u_int32_t tmp;
        if (!get4Bytes(fid, tmp)) break;
        while (tmp != 0x20746d66 /*'fmt ', little-endian*/) {
            // Skip this chunk:
            u_int32_t chunkLength;
            if (!get4Bytes(fid, chunkLength)) break;
            if (!skipBytes(fid, chunkLength)) break;
            if (!get4Bytes(fid, tmp)) break;
        }

        // FORMAT Chunk (the 4-byte header code has already been parsed):
        unsigned formatLength;
        if (!get4Bytes(fid, formatLength)) break;
        unsigned short audioFormat;
        if (!get2Bytes(fid, audioFormat)) break;

        fAudioFormat = (unsigned char)audioFormat;
        if (fAudioFormat != WA_PCM && fAudioFormat != WA_PCMA &&
            fAudioFormat != WA_PCMU && fAudioFormat != WA_IMA_ADPCM) {
            // It's a format that we don't (yet) understand
            env.setResultMsg(
                    "Audio format is not one that we handle (PCM/PCMU/PCMA or "
                    "IMA ADPCM)");
            break;
        }
        unsigned short numChannels;
        if (!get2Bytes(fid, numChannels)) break;
        fNumChannels = (unsigned char)numChannels;
        if (fNumChannels < 1 || fNumChannels > 2) {  // invalid # channels
            char errMsg[100];
            sprintf(errMsg, "Bad # channels: %d", fNumChannels);
            env.setResultMsg(errMsg);
            break;
        }
        if (!get4Bytes(fid, fSamplingFrequency)) break;
        if (fSamplingFrequency == 0) {
            env.setResultMsg("Bad sampling frequency: 0");
            break;
        }
        if (!skipBytes(fid, 6))
            break;  // "nAvgBytesPerSec" (4 bytes) + "nBlockAlign" (2 bytes)
        unsigned short bitsPerSample;
        if (!get2Bytes(fid, bitsPerSample)) break;
        fBitsPerSample = (unsigned char)bitsPerSample;
        if (fBitsPerSample == 0) {
            env.setResultMsg("Bad bits-per-sample: 0");
            break;
        }
        if (!skipBytes(fid, formatLength - 16)) break;

        // FACT chunk (optional):
        int c = nextc;
        if (c == 'f') {
            if (nextc != 'a' || nextc != 'c' || nextc != 't') break;
            unsigned factLength;
            if (!get4Bytes(fid, factLength)) break;
            if (!skipBytes(fid, factLength)) break;
            c = nextc;
        }

        // EYRE chunk (optional):
        if (c == 'e') {
            if (nextc != 'y' || nextc != 'r' || nextc != 'e') break;
            unsigned eyreLength;
            if (!get4Bytes(fid, eyreLength)) break;
            if (!skipBytes(fid, eyreLength)) break;
            c = nextc;
        }

        // DATA Chunk:
        if (c != 'd' || nextc != 'a' || nextc != 't' || nextc != 'a') break;
        if (!skipBytes(fid, 4)) break;

        // The header is good; the remaining data are the sample bytes.
        fWAVHeaderSize = (unsigned)TellFile64(fid);
        success = True;
    } while (0);

    if (!success) {
        env.setResultMsg("Bad WAV file format");
        // Set "fBitsPerSample" to zero, to indicate failure:
        fBitsPerSample = 0;
        return;
    }

    fPlayTimePerSample = 1e6 / (double)fSamplingFrequency;

    // Although PCM is a sample-based format, we group samples into
    // 'frames' for efficient delivery to clients.  Set up our preferred
    // frame size to be close to 20 ms, if possible, but always no greater
    // than 1400 bytes (to ensure that it will fit in a single RTP packet)
    unsigned maxSamplesPerFrame = (1400 * 8) / (fNumChannels * fBitsPerSample);
    unsigned desiredSamplesPerFrame = (unsigned)(0.02 * fSamplingFrequency);
    unsigned samplesPerFrame = desiredSamplesPerFrame < maxSamplesPerFrame
                                       ? desiredSamplesPerFrame
                                       : maxSamplesPerFrame;
    fPreferredFrameSize = (samplesPerFrame * fNumChannels * fBitsPerSample) / 8;

    fFidIsSeekable = FileIsSeekable(fFid);
#ifndef READ_FROM_FILES_SYNCHRONOUSLY
    // Now that we've finished reading the WAV header, all future reads (of
    // audio samples) from the file will be asynchronous:
    makeSocketNonBlocking(fileno(fFid));
#endif
}

WAVAudioFileSource::~WAVAudioFileSource() {
    if (fFid == NULL) return;

#ifndef READ_FROM_FILES_SYNCHRONOUSLY
    envir().taskScheduler().turnOffBackgroundReadHandling(fileno(fFid));
#endif

    CloseInputFile(fFid);
}

void WAVAudioFileSource::doGetNextFrame() {
    if (feof(fFid) || ferror(fFid) ||
        (fLimitNumBytesToStream && fNumBytesToStream == 0)) {
        handleClosure();
        return;
    }

    fFrameSize = 0;  // until it's set later
#ifdef READ_FROM_FILES_SYNCHRONOUSLY
    doReadFromFile();
#else
    if (!fHaveStartedReading) {
        // Await readable data from the file:
        envir().taskScheduler().turnOnBackgroundReadHandling(
                fileno(fFid),
                (TaskScheduler::BackgroundHandlerProc*)&fileReadableHandler,
                this);
        fHaveStartedReading = True;
    }
#endif
}

void WAVAudioFileSource::doStopGettingFrames() {
    envir().taskScheduler().unscheduleDelayedTask(nextTask());
#ifndef READ_FROM_FILES_SYNCHRONOUSLY
    envir().taskScheduler().turnOffBackgroundReadHandling(fileno(fFid));
    fHaveStartedReading = False;
#endif
}

void WAVAudioFileSource::fileReadableHandler(WAVAudioFileSource* source,
                                             int /*mask*/) {
    if (!source->isCurrentlyAwaitingData()) {
        source->doStopGettingFrames();  // we're not ready for the data yet
        return;
    }
    source->doReadFromFile();
}

void WAVAudioFileSource::doReadFromFile() {
    // Try to read as many bytes as will fit in the buffer provided (or
    // "fPreferredFrameSize" if less)
    if (fLimitNumBytesToStream && fNumBytesToStream < fMaxSize) {
        fMaxSize = fNumBytesToStream;
    }
    if (fPreferredFrameSize < fMaxSize) {
        fMaxSize = fPreferredFrameSize;
    }
    unsigned bytesPerSample = (fNumChannels * fBitsPerSample) / 8;
    if (bytesPerSample == 0)
        bytesPerSample = 1;  // because we can't read less than a byte at a time

    // For 'trick play', read one sample at a time; otherwise (normal case) read
    // samples in bulk:
    unsigned bytesToRead = fScaleFactor == 1
                                   ? fMaxSize - fMaxSize % bytesPerSample
                                   : bytesPerSample;
    unsigned numBytesRead;
    while (1) {  // loop for 'trick play' only
#ifdef READ_FROM_FILES_SYNCHRONOUSLY
        numBytesRead = fread(fTo, 1, bytesToRead, fFid);
#else
        if (fFidIsSeekable) {
            numBytesRead = fread(fTo, 1, bytesToRead, fFid);
        } else {
            // For non-seekable files (e.g., pipes), call "read()" rather than
            // "fread()", to ensure that the read doesn't block:
            numBytesRead = read(fileno(fFid), fTo, bytesToRead);
        }
#endif
        if (numBytesRead == 0) {
            handleClosure();
            return;
        }
        fFrameSize += numBytesRead;
        fTo += numBytesRead;
        fMaxSize -= numBytesRead;
        fNumBytesToStream -= numBytesRead;

        // If we did an asynchronous read, and didn't read an integral number of
        // samples, then we need to wait for another read:
#ifndef READ_FROM_FILES_SYNCHRONOUSLY
        if (fFrameSize % bytesPerSample > 0) return;
#endif

        // If we're doing 'trick play', then seek to the appropriate place for
        // reading the next sample, and keep reading until we fill the provided
        // buffer:
        if (fScaleFactor != 1) {
            SeekFile64(fFid, (fScaleFactor - 1) * bytesPerSample, SEEK_CUR);
            if (fMaxSize < bytesPerSample) break;
        } else {
            break;  // from the loop (normal case)
        }
    }

    // Set the 'presentation time' and 'duration' of this frame:
    if (fPresentationTime.tv_sec == 0 && fPresentationTime.tv_usec == 0) {
        // This is the first frame, so use the current time:
        gettimeofday(&fPresentationTime, NULL);
    } else {
        // Increment by the play time of the previous data:
        unsigned uSeconds = fPresentationTime.tv_usec + fLastPlayTime;
        fPresentationTime.tv_sec += uSeconds / 1000000;
        fPresentationTime.tv_usec = uSeconds % 1000000;
    }

    // Remember the play time of this data:
    fDurationInMicroseconds = fLastPlayTime =
            (unsigned)((fPlayTimePerSample * fFrameSize) / bytesPerSample);

    // Inform the reader that he has data:
#ifdef READ_FROM_FILES_SYNCHRONOUSLY
    // To avoid possible infinite recursion, we need to return to the event loop
    // to do this:
    nextTask() = envir().taskScheduler().scheduleDelayedTask(
            0, (TaskFunc*)FramedSource::afterGetting, this);
#else
    // Because the file read was done from the event loop, we can call the
    // 'after getting' function directly, without risk of infinite recursion:
    FramedSource::afterGetting(this);
#endif
}

Boolean WAVAudioFileSource::setInputPort(int /*portIndex*/) { return True; }

double WAVAudioFileSource::getAverageLevel() const {
    return 0.0;  //##### fix this later
}
