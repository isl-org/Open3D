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
// A RTSP server
// Implementation of functionality related to the "REGISTER" and "DEREGISTER"
// commands

#include "GroupsockHelper.hh"
#include "ProxyServerMediaSession.hh"
#include "RTSPCommon.hh"
#include "RTSPRegisterSender.hh"
#include "RTSPServer.hh"

////////// Implementation of "RTSPServer::registerStream()": //////////

static void rtspRegisterResponseHandler(RTSPClient* rtspClient,
                                        int resultCode,
                                        char* resultString);  // forward

// A class that represents the state of a "REGISTER" request in progress:
class RegisterRequestRecord : public RTSPRegisterSender {
public:
    RegisterRequestRecord(
            RTSPServer& ourServer,
            unsigned requestId,
            char const* remoteClientNameOrAddress,
            portNumBits remoteClientPortNum,
            char const* rtspURLToRegister,
            RTSPServer::responseHandlerForREGISTER* responseHandler,
            Authenticator* authenticator,
            Boolean requestStreamingViaTCP,
            char const* proxyURLSuffix)
        : RTSPRegisterSender(ourServer.envir(),
                             remoteClientNameOrAddress,
                             remoteClientPortNum,
                             rtspURLToRegister,
                             rtspRegisterResponseHandler,
                             authenticator,
                             requestStreamingViaTCP,
                             proxyURLSuffix,
                             True /*reuseConnection*/,
#ifdef DEBUG
                             1 /*verbosityLevel*/,
#else
                             0 /*verbosityLevel*/,
#endif
                             NULL),
          fOurServer(ourServer),
          fRequestId(requestId),
          fResponseHandler(responseHandler) {
        // Add ourself to our server's 'pending REGISTER or DEREGISTER requests'
        // table:
        ourServer.fPendingRegisterOrDeregisterRequests->Add((char const*)this,
                                                            this);
    }

    virtual ~RegisterRequestRecord() {
        // Remove ourself from the server's 'pending REGISTER or DEREGISTER
        // requests' hash table before we go:
        fOurServer.fPendingRegisterOrDeregisterRequests->Remove(
                (char const*)this);
    }

    void handleResponse(int resultCode, char* resultString) {
        if (resultCode == 0) {
            // The "REGISTER" request succeeded, so use the still-open RTSP
            // socket to await incoming commands from the remote endpoint:
            int sock;
            struct sockaddr_in remoteAddress;

            grabConnection(sock, remoteAddress);
            if (sock >= 0) {
                increaseSendBufferTo(
                        envir(), sock,
                        50 * 1024);  // in anticipation of streaming over it
                (void)fOurServer.createNewClientConnection(sock, remoteAddress);
            }
        }

        if (fResponseHandler != NULL) {
            // Call our (REGISTER-specific) response handler now:
            (*fResponseHandler)(&fOurServer, fRequestId, resultCode,
                                resultString);
        } else {
            // We need to delete[] "resultString" before we leave:
            delete[] resultString;
        }

        // We're completely done with the REGISTER command now, so delete
        // ourself now:
        Medium::close(this);
    }

private:
    RTSPServer& fOurServer;
    unsigned fRequestId;
    RTSPServer::responseHandlerForREGISTER* fResponseHandler;
};

static void rtspRegisterResponseHandler(RTSPClient* rtspClient,
                                        int resultCode,
                                        char* resultString) {
    RegisterRequestRecord* registerRequestRecord =
            (RegisterRequestRecord*)rtspClient;

    registerRequestRecord->handleResponse(resultCode, resultString);
}

unsigned RTSPServer::registerStream(ServerMediaSession* serverMediaSession,
                                    char const* remoteClientNameOrAddress,
                                    portNumBits remoteClientPortNum,
                                    responseHandlerForREGISTER* responseHandler,
                                    char const* username,
                                    char const* password,
                                    Boolean receiveOurStreamViaTCP,
                                    char const* proxyURLSuffix) {
    // Create a new "RegisterRequestRecord" that will send the "REGISTER"
    // command. (This object will automatically get deleted after we get a
    // response to the "REGISTER" command, or if we're deleted.)
    Authenticator* authenticator = NULL;
    if (username != NULL) {
        if (password == NULL) password = "";
        authenticator = new Authenticator(username, password);
    }
    unsigned requestId = ++fRegisterOrDeregisterRequestCounter;
    char const* url = rtspURL(serverMediaSession);
    new RegisterRequestRecord(*this, requestId, remoteClientNameOrAddress,
                              remoteClientPortNum, url, responseHandler,
                              authenticator, receiveOurStreamViaTCP,
                              proxyURLSuffix);

    delete[] url;          // we can do this here because it was copied to the
                           // "RegisterRequestRecord"
    delete authenticator;  // ditto
    return requestId;
}

////////// Implementation of "RTSPServer::deregisterStream()": //////////

static void rtspDeregisterResponseHandler(RTSPClient* rtspClient,
                                          int resultCode,
                                          char* resultString);  // forward

// A class that represents the state of a "DEREGISTER" request in progress:
class DeregisterRequestRecord : public RTSPDeregisterSender {
public:
    DeregisterRequestRecord(
            RTSPServer& ourServer,
            unsigned requestId,
            char const* remoteClientNameOrAddress,
            portNumBits remoteClientPortNum,
            char const* rtspURLToDeregister,
            RTSPServer::responseHandlerForDEREGISTER* responseHandler,
            Authenticator* authenticator,
            char const* proxyURLSuffix)
        : RTSPDeregisterSender(ourServer.envir(),
                               remoteClientNameOrAddress,
                               remoteClientPortNum,
                               rtspURLToDeregister,
                               rtspDeregisterResponseHandler,
                               authenticator,
                               proxyURLSuffix,
#ifdef DEBUG
                               1 /*verbosityLevel*/,
#else
                               0 /*verbosityLevel*/,
#endif
                               NULL),
          fOurServer(ourServer),
          fRequestId(requestId),
          fResponseHandler(responseHandler) {
        // Add ourself to our server's 'pending REGISTER or DEREGISTER requests'
        // table:
        ourServer.fPendingRegisterOrDeregisterRequests->Add((char const*)this,
                                                            this);
    }

    virtual ~DeregisterRequestRecord() {
        // Remove ourself from the server's 'pending REGISTER or DEREGISTER
        // requests' hash table before we go:
        fOurServer.fPendingRegisterOrDeregisterRequests->Remove(
                (char const*)this);
    }

    void handleResponse(int resultCode, char* resultString) {
        if (fResponseHandler != NULL) {
            // Call our (DEREGISTER-specific) response handler now:
            (*fResponseHandler)(&fOurServer, fRequestId, resultCode,
                                resultString);
        } else {
            // We need to delete[] "resultString" before we leave:
            delete[] resultString;
        }

        // We're completely done with the DEREGISTER command now, so delete
        // ourself now:
        Medium::close(this);
    }

private:
    RTSPServer& fOurServer;
    unsigned fRequestId;
    RTSPServer::responseHandlerForDEREGISTER* fResponseHandler;
};

static void rtspDeregisterResponseHandler(RTSPClient* rtspClient,
                                          int resultCode,
                                          char* resultString) {
    DeregisterRequestRecord* deregisterRequestRecord =
            (DeregisterRequestRecord*)rtspClient;

    deregisterRequestRecord->handleResponse(resultCode, resultString);
}

unsigned RTSPServer::deregisterStream(
        ServerMediaSession* serverMediaSession,
        char const* remoteClientNameOrAddress,
        portNumBits remoteClientPortNum,
        responseHandlerForDEREGISTER* responseHandler,
        char const* username,
        char const* password,
        char const* proxyURLSuffix) {
    // Create a new "DeregisterRequestRecord" that will send the "DEREGISTER"
    // command. (This object will automatically get deleted after we get a
    // response to the "DEREGISTER" command, or if we're deleted.)
    Authenticator* authenticator = NULL;
    if (username != NULL) {
        if (password == NULL) password = "";
        authenticator = new Authenticator(username, password);
    }
    unsigned requestId = ++fRegisterOrDeregisterRequestCounter;
    char const* url = rtspURL(serverMediaSession);
    new DeregisterRequestRecord(*this, requestId, remoteClientNameOrAddress,
                                remoteClientPortNum, url, responseHandler,
                                authenticator, proxyURLSuffix);

    delete[] url;          // we can do this here because it was copied to the
                           // "DeregisterRequestRecord"
    delete authenticator;  // ditto
    return requestId;
}

Boolean RTSPServer::weImplementREGISTER(
        char const* /*cmd*/ /*"REGISTER" or "DEREGISTER"*/,
        char const* /*proxyURLSuffix*/,
        char*& responseStr) {
    // By default, servers do not implement our custom "REGISTER"/"DEREGISTER"
    // commands:
    responseStr = NULL;
    return False;
}

void RTSPServer::implementCmd_REGISTER(
        char const* /*cmd*/ /*"REGISTER" or "DEREGISTER"*/,
        char const* /*url*/,
        char const* /*urlSuffix*/,
        int /*socketToRemoteServer*/,
        Boolean /*deliverViaTCP*/,
        char const* /*proxyURLSuffix*/) {
    // By default, this function is a 'noop'
}

// Special mechanism for handling our custom "REGISTER" command:

RTSPServer::RTSPClientConnection::ParamsForREGISTER ::ParamsForREGISTER(
        char const* cmd /*"REGISTER" or "DEREGISTER"*/,
        RTSPServer::RTSPClientConnection* ourConnection,
        char const* url,
        char const* urlSuffix,
        Boolean reuseConnection,
        Boolean deliverViaTCP,
        char const* proxyURLSuffix)
    : fCmd(strDup(cmd)),
      fOurConnection(ourConnection),
      fURL(strDup(url)),
      fURLSuffix(strDup(urlSuffix)),
      fReuseConnection(reuseConnection),
      fDeliverViaTCP(deliverViaTCP),
      fProxyURLSuffix(strDup(proxyURLSuffix)) {}

RTSPServer::RTSPClientConnection::ParamsForREGISTER::~ParamsForREGISTER() {
    delete[](char*) fCmd;
    delete[] fURL;
    delete[] fURLSuffix;
    delete[] fProxyURLSuffix;
}

#define DELAY_USECS_AFTER_REGISTER_RESPONSE 100000 /*100ms*/

void RTSPServer ::RTSPClientConnection::handleCmd_REGISTER(
        char const* cmd /*"REGISTER" or "DEREGISTER"*/,
        char const* url,
        char const* urlSuffix,
        char const* fullRequestStr,
        Boolean reuseConnection,
        Boolean deliverViaTCP,
        char const* proxyURLSuffix) {
    char* responseStr;
    if (fOurRTSPServer.weImplementREGISTER(cmd, proxyURLSuffix, responseStr)) {
        // The "REGISTER"/"DEREGISTER" command - if we implement it - may
        // require access control:
        if (!authenticationOK(cmd, urlSuffix, fullRequestStr)) return;

        // We implement the "REGISTER"/"DEREGISTER" command by first replying to
        // it, then actually handling it (in a separate event-loop task, that
        // will get called after the reply has been done). Hack: If we're going
        // to reuse the command's connection for subsequent RTSP commands, then
        // we delay the actual handling of the command slightly, to make it less
        // likely that the first subsequent RTSP command (e.g., "DESCRIBE") will
        // end up in the client's reponse buffer before the socket (at the far
        // end) gets reused for RTSP command handling.
        setRTSPResponse(responseStr == NULL ? "200 OK" : responseStr);
        delete[] responseStr;

        ParamsForREGISTER* registerParams = new ParamsForREGISTER(
                cmd, this, url, urlSuffix, reuseConnection, deliverViaTCP,
                proxyURLSuffix);
        envir().taskScheduler().scheduleDelayedTask(
                reuseConnection ? DELAY_USECS_AFTER_REGISTER_RESPONSE : 0,
                (TaskFunc*)continueHandlingREGISTER, registerParams);
    } else if (responseStr != NULL) {
        setRTSPResponse(responseStr);
        delete[] responseStr;
    } else {
        handleCmd_notSupported();
    }
}

// A special version of "parseTransportHeader()", used just for parsing the
// "Transport:" header in an incoming "REGISTER" command:
void parseTransportHeaderForREGISTER(char const* buf,
                                     Boolean& reuseConnection,
                                     Boolean& deliverViaTCP,
                                     char*& proxyURLSuffix) {
    // Initialize the result parameters to default values:
    reuseConnection = False;
    deliverViaTCP = False;
    proxyURLSuffix = NULL;

    // First, find "Transport:"
    while (1) {
        if (*buf == '\0') return;  // not found
        if (*buf == '\r' && *(buf + 1) == '\n' && *(buf + 2) == '\r')
            return;  // end of the headers => not found
        if (_strncasecmp(buf, "Transport:", 10) == 0) break;
        ++buf;
    }

    // Then, run through each of the fields, looking for ones we handle:
    char const* fields = buf + 10;
    while (*fields == ' ') ++fields;
    char* field = strDupSize(fields);
    while (sscanf(fields, "%[^;\r\n]", field) == 1) {
        if (strcmp(field, "reuse_connection") == 0) {
            reuseConnection = True;
        } else if (_strncasecmp(field, "preferred_delivery_protocol=udp", 31) ==
                   0) {
            deliverViaTCP = False;
        } else if (_strncasecmp(field,
                                "preferred_delivery_protocol=interleaved",
                                39) == 0) {
            deliverViaTCP = True;
        } else if (_strncasecmp(field, "proxy_url_suffix=", 17) == 0) {
            delete[] proxyURLSuffix;
            proxyURLSuffix = strDup(field + 17);
        }

        fields += strlen(field);
        while (*fields == ';' || *fields == ' ' || *fields == '\t')
            ++fields;  // skip over separating ';' chars or whitespace
        if (*fields == '\0' || *fields == '\r' || *fields == '\n') break;
    }
    delete[] field;
}

void RTSPServer::RTSPClientConnection::continueHandlingREGISTER(
        ParamsForREGISTER* params) {
    params->fOurConnection->continueHandlingREGISTER1(params);
}

void RTSPServer::RTSPClientConnection::continueHandlingREGISTER1(
        ParamsForREGISTER* params) {
    // Reuse our socket if requested:
    int socketNumToBackEndServer =
            params->fReuseConnection ? fClientOutputSocket : -1;

    RTSPServer* ourServer = &fOurRTSPServer;  // copy the pointer now, in case
                                              // we "delete this" below

    if (socketNumToBackEndServer >= 0) {
        // Because our socket will no longer be used by the server to handle
        // incoming requests, we can now delete this "RTSPClientConnection"
        // object.  We do this now, in case the "implementCmd_REGISTER()" call
        // below would also end up deleting this.
        fClientInputSocket = fClientOutputSocket =
                -1;  // so the socket doesn't get closed when we get deleted
        delete this;
    }

    ourServer->implementCmd_REGISTER(
            params->fCmd, params->fURL, params->fURLSuffix,
            socketNumToBackEndServer, params->fDeliverViaTCP,
            params->fProxyURLSuffix);
    delete params;
}

///////// RTSPServerWithREGISTERProxying implementation /////////

RTSPServerWithREGISTERProxying* RTSPServerWithREGISTERProxying ::createNew(
        UsageEnvironment& env,
        Port ourPort,
        UserAuthenticationDatabase* authDatabase,
        UserAuthenticationDatabase* authDatabaseForREGISTER,
        unsigned reclamationSeconds,
        Boolean streamRTPOverTCP,
        int verbosityLevelForProxying,
        char const* backEndUsername,
        char const* backEndPassword) {
    int ourSocket = setUpOurSocket(env, ourPort);
    if (ourSocket == -1) return NULL;

    return new RTSPServerWithREGISTERProxying(
            env, ourSocket, ourPort, authDatabase, authDatabaseForREGISTER,
            reclamationSeconds, streamRTPOverTCP, verbosityLevelForProxying,
            backEndUsername, backEndPassword);
}

RTSPServerWithREGISTERProxying ::RTSPServerWithREGISTERProxying(
        UsageEnvironment& env,
        int ourSocket,
        Port ourPort,
        UserAuthenticationDatabase* authDatabase,
        UserAuthenticationDatabase* authDatabaseForREGISTER,
        unsigned reclamationSeconds,
        Boolean streamRTPOverTCP,
        int verbosityLevelForProxying,
        char const* backEndUsername,
        char const* backEndPassword)
    : RTSPServer(env, ourSocket, ourPort, authDatabase, reclamationSeconds),
      fStreamRTPOverTCP(streamRTPOverTCP),
      fVerbosityLevelForProxying(verbosityLevelForProxying),
      fRegisteredProxyCounter(0),
      fAllowedCommandNames(NULL),
      fAuthDBForREGISTER(authDatabaseForREGISTER),
      fBackEndUsername(strDup(backEndUsername)),
      fBackEndPassword(strDup(backEndPassword)) {}

RTSPServerWithREGISTERProxying::~RTSPServerWithREGISTERProxying() {
    delete[] fAllowedCommandNames;
    delete[] fBackEndUsername;
    delete[] fBackEndPassword;
}

char const* RTSPServerWithREGISTERProxying::allowedCommandNames() {
    if (fAllowedCommandNames == NULL) {
        char const* baseAllowedCommandNames = RTSPServer::allowedCommandNames();
        char const* newAllowedCommandName = ", REGISTER, DEREGISTER";
        fAllowedCommandNames =
                new char[strlen(baseAllowedCommandNames) +
                         strlen(newAllowedCommandName) + 1 /* for '\0' */];
        sprintf(fAllowedCommandNames, "%s%s", baseAllowedCommandNames,
                newAllowedCommandName);
    }
    return fAllowedCommandNames;
}

Boolean RTSPServerWithREGISTERProxying ::weImplementREGISTER(
        char const* cmd /*"REGISTER" or "DEREGISTER"*/,
        char const* proxyURLSuffix,
        char*& responseStr) {
    // First, check whether we have already proxied a stream as
    // "proxyURLSuffix":
    if (proxyURLSuffix != NULL) {
        ServerMediaSession* sms = lookupServerMediaSession(proxyURLSuffix);
        if ((strcmp(cmd, "REGISTER") == 0 && sms != NULL) ||
            (strcmp(cmd, "DEREGISTER") == 0 && sms == NULL)) {
            responseStr = strDup("451 Invalid parameter");
            return False;
        }
    }

    // Otherwise, we will implement it:
    responseStr = NULL;
    return True;
}

void RTSPServerWithREGISTERProxying ::implementCmd_REGISTER(
        char const* cmd /*"REGISTER" or "DEREGISTER"*/,
        char const* url,
        char const* /*urlSuffix*/,
        int socketToRemoteServer,
        Boolean deliverViaTCP,
        char const* proxyURLSuffix) {
    // Continue setting up proxying for the specified URL.
    // By default:
    //    - We use "registeredProxyStream-N" as the (front-end) stream name
    //    (ignoring the back-end stream's 'urlSuffix'),
    //      unless "proxyURLSuffix" is non-NULL (in which case we use that)
    //    - There is no 'username' and 'password' for the back-end stream.
    //    (Thus, access-controlled back-end streams will fail.)
    //    - If "fStreamRTPOverTCP" is True, then we request delivery over TCP,
    //    regardless of the value of "deliverViaTCP".
    //      (Otherwise, if "fStreamRTPOverTCP" is False, we use the value of
    //      "deliverViaTCP" to decide this.)
    // To change this default behavior, you will need to subclass
    // "RTSPServerWithREGISTERProxying", and reimplement this function.

    char const* proxyStreamName;
    char proxyStreamNameBuf[100];
    if (proxyURLSuffix == NULL) {
        sprintf(proxyStreamNameBuf, "registeredProxyStream-%u",
                ++fRegisteredProxyCounter);
        proxyStreamName = proxyStreamNameBuf;
    } else {
        proxyStreamName = proxyURLSuffix;
    }

    if (strcmp(cmd, "REGISTER") == 0) {
        if (fStreamRTPOverTCP) deliverViaTCP = True;
        portNumBits tunnelOverHTTPPortNum =
                deliverViaTCP ? (portNumBits)(~0) : 0;
        // We don't support streaming from the back-end via
        // RTSP/RTP/RTCP-over-HTTP; only via RTP/RTCP-over-TCP or
        // RTP/RTCP-over-UDP

        ServerMediaSession* sms = ProxyServerMediaSession::createNew(
                envir(), this, url, proxyStreamName, fBackEndUsername,
                fBackEndPassword, tunnelOverHTTPPortNum,
                fVerbosityLevelForProxying, socketToRemoteServer);
        addServerMediaSession(sms);

        // (Regardless of the verbosity level) announce the fact that we're
        // proxying this new stream, and the URL to use to access it:
        char* proxyStreamURL = rtspURL(sms);
        envir() << "Proxying the registered back-end stream \"" << url
                << "\".\n";
        envir() << "\tPlay this stream using the URL: " << proxyStreamURL
                << "\n";
        delete[] proxyStreamURL;
    } else {  // "DEREGISTER"
        deleteServerMediaSession(lookupServerMediaSession(proxyStreamName));
    }
}

UserAuthenticationDatabase*
RTSPServerWithREGISTERProxying::getAuthenticationDatabaseForCommand(
        char const* cmdName) {
    if (strcmp(cmdName, "REGISTER") == 0) return fAuthDBForREGISTER;

    return RTSPServer::getAuthenticationDatabaseForCommand(cmdName);
}
