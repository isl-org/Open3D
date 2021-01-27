/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** environment.cpp
**
** -------------------------------------------------------------------------*/

#include <live555helper/environment.h>

#include <iostream>

Environment::Environment() : Environment(m_stopRef) {}

Environment::Environment(char& stop)
    : BasicUsageEnvironment(*BasicTaskScheduler::createNew()), m_stop(stop) {
    m_stop = 0;
}

Environment::~Environment() {
    TaskScheduler* scheduler = &this->taskScheduler();
    delete scheduler;
}

void Environment::mainloop() { this->taskScheduler().doEventLoop(&m_stop); }

void Environment::stop() { m_stop = 1; }
