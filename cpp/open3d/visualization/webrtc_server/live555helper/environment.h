/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** environment.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include "BasicUsageEnvironment.hh"

class Environment : public BasicUsageEnvironment {
public:
    Environment();
    Environment(char& stop);
    virtual ~Environment();

    void mainloop();
    void stop();

protected:
    char& m_stop;
    char m_stopRef;
};
