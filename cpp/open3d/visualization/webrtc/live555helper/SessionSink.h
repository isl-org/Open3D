/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** SessionSink.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <stdint.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <live555helper/environment.h>

#include "liveMedia.hh"

static uint8_t H26X_marker[] = {0, 0, 0, 1};

/* ---------------------------------------------------------------------------
**  Media client callback interface
** -------------------------------------------------------------------------*/
class SessionCallback {
public:
    virtual bool onNewSession(const char* id,
                              const char* media,
                              const char* codec,
                              const char* sdp) {
        return true;
    }
    virtual bool onData(const char* id,
                        unsigned char* buffer,
                        ssize_t size,
                        struct timeval presentationTime) = 0;
    virtual ssize_t onNewBuffer(const char* id,
                                const char* mime,
                                unsigned char* buffer,
                                ssize_t size) {
        ssize_t markerSize = 0;
        if ((strcmp(mime, "video/H264") == 0) ||
            (strcmp(mime, "video/H265") == 0)) {
            if ((unsigned long)size > sizeof(H26X_marker)) {
                memcpy(buffer, H26X_marker, sizeof(H26X_marker));
                markerSize = sizeof(H26X_marker);
            }
        }
        return markerSize;
    }
};

/* ---------------------------------------------------------------------------
**  Media client Sink
** -------------------------------------------------------------------------*/
class SessionSink : public MediaSink {
public:
    static SessionSink* createNew(UsageEnvironment& env,
                                  SessionCallback* callback) {
        return new SessionSink(env, callback);
    }

private:
    SessionSink(UsageEnvironment& env, SessionCallback* callback);
    virtual ~SessionSink();

    void allocate(ssize_t bufferSize);

    static void afterGettingFrame(void* clientData,
                                  unsigned frameSize,
                                  unsigned numTruncatedBytes,
                                  struct timeval presentationTime,
                                  unsigned durationInMicroseconds) {
        static_cast<SessionSink*>(clientData)
                ->afterGettingFrame(frameSize, numTruncatedBytes,
                                    presentationTime, durationInMicroseconds);
    }

    void afterGettingFrame(unsigned frameSize,
                           unsigned numTruncatedBytes,
                           struct timeval presentationTime,
                           unsigned durationInMicroseconds);

    virtual Boolean continuePlaying();

private:
    u_int8_t* m_buffer;
    size_t m_bufferSize;
    SessionCallback* m_callback;
    ssize_t m_markerSize;
};
