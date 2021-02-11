/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** SessionSink.cpp
**
** -------------------------------------------------------------------------*/

#include "SessionSink.h"

#include <iostream>

SessionSink::SessionSink(UsageEnvironment& env, SessionCallback* callback)
    : MediaSink(env),
      m_buffer(NULL),
      m_bufferSize(0),
      m_callback(callback),
      m_markerSize(0) {}

SessionSink::~SessionSink() { delete[] m_buffer; }

void SessionSink::allocate(ssize_t bufferSize) {
    m_bufferSize = bufferSize;
    m_buffer = new u_int8_t[m_bufferSize];
    if (m_callback) {
        m_markerSize = m_callback->onNewBuffer(this->name(),
                                               this->source()->MIMEtype(),
                                               m_buffer, m_bufferSize);
        envir() << "markerSize:" << (int)m_markerSize << "\n";
    }
}

void SessionSink::afterGettingFrame(unsigned frameSize,
                                    unsigned numTruncatedBytes,
                                    struct timeval presentationTime,
                                    unsigned durationInMicroseconds) {
    if (numTruncatedBytes != 0) {
        delete[] m_buffer;
        envir() << "buffer too small " << (int)m_bufferSize
                << " allocate bigger one\n";
        allocate(m_bufferSize * 2);
    } else if (m_callback) {
        std::cout << "SessionSink::afterGettingFrame" << std::endl;
        if (!m_callback->onData(this->name(), m_buffer,
                                frameSize + m_markerSize, presentationTime)) {
            envir() << "NOTIFY failed\n";
        }
    }
    this->continuePlaying();
}

Boolean SessionSink::continuePlaying() {
    if (m_buffer == NULL) {
        allocate(1024 * 1024);
    }
    Boolean ret = False;
    if (source() != NULL) {
        source()->getNextFrame(m_buffer + m_markerSize,
                               m_bufferSize - m_markerSize, afterGettingFrame,
                               this, onSourceClosure, this);
        ret = True;
    }
    return ret;
}
