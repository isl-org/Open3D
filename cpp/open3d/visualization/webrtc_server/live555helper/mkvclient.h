/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** mkvclient.h
**
** Interface to an MKV client
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/environment.h>

#include <map>
#include <string>

#include "SessionSink.h"
#include "liveMedia.hh"

/* ---------------------------------------------------------------------------
**  MKV client connection interface
** -------------------------------------------------------------------------*/
class MKVClient {
public:
    /* ---------------------------------------------------------------------------
    **  MKV client callback interface
    ** -------------------------------------------------------------------------*/
    class Callback : public SessionCallback {
    public:
        virtual void onError(MKVClient&, const char*) {}
        virtual void onEndOfFile(MKVClient& client) { client.stop(); }
    };

public:
    MKVClient(Environment& env,
              Callback* callback,
              const char* path,
              const std::map<std::string, std::string>& opts,
              int verbosityLevel = 1);
    virtual ~MKVClient();

    void stop() { m_env.stop(); }

private:
    void onMatroskaFileCreation(MatroskaFile* newFile);
    static void onMatroskaFileCreation(MatroskaFile* newFile,
                                       void* clientData) {
        ((MKVClient*)(clientData))->onMatroskaFileCreation(newFile);
    }
    void onEndOfFile();
    static void onEndOfFile(void* clientData) {
        ((MKVClient*)(clientData))->onEndOfFile();
    }

protected:
    Environment& m_env;
    Callback* m_callback;
    MatroskaFile* m_mkvfile;
    MatroskaDemux* m_demux;
};
