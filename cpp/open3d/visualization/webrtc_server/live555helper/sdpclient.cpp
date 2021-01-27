/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** sdpclient.cpp
**
** Interface to an SDP client
**
** -------------------------------------------------------------------------*/

#include <live555helper/sdpclient.h>

SDPClient::SDPClient(Environment& env,
                     Callback* callback,
                     const char* sdp,
                     const std::map<std::string, std::string>& opts,
                     int verbosityLevel)
    : m_env(env), m_callback(callback) {
    m_session = MediaSession::createNew(m_env, sdp);

    if (m_session == NULL) {
        m_env << "Failed to create session from \"" << sdp
              << "\" error: " << m_env.getResultMsg() << "\n";
    } else {
        MediaSubsessionIterator iter(*m_session);
        MediaSubsession* subsession = NULL;
        while ((subsession = iter.next()) != NULL) {
            if (!subsession->initiate()) {
                m_env << "Failed to create sink for \""
                      << subsession->mediumName() << "/"
                      << subsession->codecName()
                      << "\" subsession error: " << m_env.getResultMsg()
                      << "\n";
            } else {
                MediaSink* sink = SessionSink::createNew(m_env, m_callback);
                if (sink == NULL) {
                    m_env << "Failed to create sink for \""
                          << subsession->mediumName() << "/"
                          << subsession->codecName()
                          << "\" subsession error: " << m_env.getResultMsg()
                          << "\n";
                    m_callback->onError(*this, m_env.getResultMsg());
                } else if (m_callback->onNewSession(
                                   sink->name(), subsession->mediumName(),
                                   subsession->codecName(),
                                   subsession->savedSDPLines())) {
                    m_env << "Start playing sink for \""
                          << subsession->mediumName() << "/"
                          << subsession->codecName() << "\" subsession"
                          << "\n";
                    subsession->sink = sink;
                    subsession->sink->startPlaying(*(subsession->readSource()),
                                                   NULL, NULL);
                } else {
                    Medium::close(sink);
                }
            }
        }
    }
}

SDPClient::~SDPClient() {
    // free subsession
    if (m_session != NULL) {
        MediaSubsessionIterator iter(*m_session);
        MediaSubsession* subsession;
        while ((subsession = iter.next()) != NULL) {
            if (subsession->sink) {
                m_env << "Close session: " << subsession->mediumName() << "/"
                      << subsession->codecName() << "\n";
                Medium::close(subsession->sink);
                subsession->sink = NULL;
            }
        }
        Medium::close(m_session);
    }
}
