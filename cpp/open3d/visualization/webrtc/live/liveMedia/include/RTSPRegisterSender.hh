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
// Special objects which, when created, sends a custom RTSP "REGISTER" (or "DEREGISTER") command
// to a specified client.
// C++ header

#ifndef _RTSP_REGISTER_SENDER_HH
#define _RTSP_REGISTER_SENDER_HH

#ifndef _RTSP_CLIENT_HH
#include "RTSPClient.hh"
#endif

class RTSPRegisterOrDeregisterSender: public RTSPClient {
public:
  virtual ~RTSPRegisterOrDeregisterSender();
protected: // we're a virtual base class
  RTSPRegisterOrDeregisterSender(UsageEnvironment& env,
				 char const* remoteClientNameOrAddress, portNumBits remoteClientPortNum,
				 Authenticator* authenticator,
				 int verbosityLevel, char const* applicationName);

public: // Some compilers complain if this is "protected:"
  // A subclass of "RTSPClient::RequestRecord", specific to our "REGISTER" and "DEREGISTER" commands:
  class RequestRecord_REGISTER_or_DEREGISTER: public RTSPClient::RequestRecord {
  public:
    RequestRecord_REGISTER_or_DEREGISTER(unsigned cseq, char const* cmdName, RTSPClient::responseHandler* rtspResponseHandler, char const* rtspURLToRegisterOrDeregister, char const* proxyURLSuffix);
    virtual ~RequestRecord_REGISTER_or_DEREGISTER();

    char const* proxyURLSuffix() const { return fProxyURLSuffix; }

  protected:
    char* fRTSPURLToRegisterOrDeregister;
    char* fProxyURLSuffix;
  };

protected:
  portNumBits fRemoteClientPortNum;
};

//////////

class RTSPRegisterSender: public RTSPRegisterOrDeregisterSender {
public:
  static RTSPRegisterSender*
  createNew(UsageEnvironment& env,
	    char const* remoteClientNameOrAddress, portNumBits remoteClientPortNum, char const* rtspURLToRegister,
	    RTSPClient::responseHandler* rtspResponseHandler, Authenticator* authenticator = NULL,
	    Boolean requestStreamingViaTCP = False, char const* proxyURLSuffix = NULL, Boolean reuseConnection = False,
	    int verbosityLevel = 0, char const* applicationName = NULL);

  void grabConnection(int& sock, struct sockaddr_in& remoteAddress); // so that the socket doesn't get closed when we're deleted

protected:
  RTSPRegisterSender(UsageEnvironment& env,
		     char const* remoteClientNameOrAddress, portNumBits remoteClientPortNum, char const* rtspURLToRegister,
		     RTSPClient::responseHandler* rtspResponseHandler, Authenticator* authenticator,
		     Boolean requestStreamingViaTCP, char const* proxyURLSuffix, Boolean reuseConnection,
		     int verbosityLevel, char const* applicationName);
    // called only by "createNew()"
  virtual ~RTSPRegisterSender();

  // Redefined virtual functions:
  virtual Boolean setRequestFields(RequestRecord* request,
                                   char*& cmdURL, Boolean& cmdURLWasAllocated,
                                   char const*& protocolStr,
                                   char*& extraHeaders, Boolean& extraHeadersWereAllocated);

public: // Some compilers complain if this is "protected:"
  // A subclass of "RequestRecord_REGISTER_or_DEREGISTER", specific to our "REGISTER" command:
  class RequestRecord_REGISTER: public RTSPRegisterOrDeregisterSender::RequestRecord_REGISTER_or_DEREGISTER {
  public:
    RequestRecord_REGISTER(unsigned cseq, RTSPClient::responseHandler* rtspResponseHandler, char const* rtspURLToRegister,
			   Boolean reuseConnection, Boolean requestStreamingViaTCP, char const* proxyURLSuffix);
    virtual ~RequestRecord_REGISTER();

    char const* rtspURLToRegister() const { return fRTSPURLToRegisterOrDeregister; }
    Boolean reuseConnection() const { return fReuseConnection; }
    Boolean requestStreamingViaTCP() const { return fRequestStreamingViaTCP; }

  private:
    Boolean fReuseConnection, fRequestStreamingViaTCP;
  };
};

//////////

class RTSPDeregisterSender: public RTSPRegisterOrDeregisterSender {
public:
  static RTSPDeregisterSender*
  createNew(UsageEnvironment& env,
	    char const* remoteClientNameOrAddress, portNumBits remoteClientPortNum, char const* rtspURLToDeregister,
	    RTSPClient::responseHandler* rtspResponseHandler, Authenticator* authenticator = NULL,
	    char const* proxyURLSuffix = NULL,
	    int verbosityLevel = 0, char const* applicationName = NULL);

protected:
  RTSPDeregisterSender(UsageEnvironment& env,
		       char const* remoteClientNameOrAddress, portNumBits remoteClientPortNum, char const* rtspURLToDeregister,
		     RTSPClient::responseHandler* rtspResponseHandler, Authenticator* authenticator,
		     char const* proxyURLSuffix,
		     int verbosityLevel, char const* applicationName);
    // called only by "createNew()"
  virtual ~RTSPDeregisterSender();

  // Redefined virtual functions:
  virtual Boolean setRequestFields(RequestRecord* request,
                                   char*& cmdURL, Boolean& cmdURLWasAllocated,
                                   char const*& protocolStr,
                                   char*& extraHeaders, Boolean& extraHeadersWereAllocated);

public: // Some compilers complain if this is "protected:"
  // A subclass of "RequestRecord_REGISTER_or_DEREGISTER", specific to our "DEREGISTER" command:
  class RequestRecord_DEREGISTER: public RTSPRegisterOrDeregisterSender::RequestRecord_REGISTER_or_DEREGISTER {
  public:
    RequestRecord_DEREGISTER(unsigned cseq, RTSPClient::responseHandler* rtspResponseHandler, char const* rtspURLToDeregister, char const* proxyURLSuffix);
    virtual ~RequestRecord_DEREGISTER();

    char const* rtspURLToDeregister() const { return fRTSPURLToRegisterOrDeregister; }
  };
};

#endif
