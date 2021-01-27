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
// "mTunnel" multicast access service
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// Helper routines to implement 'group sockets'
// Implementation

#include "GroupsockHelper.hh"

#if (defined(__WIN32__) || defined(_WIN32)) && !defined(__MINGW32__)
#include <time.h>
extern "C" int initializeWinsockIfNecessary();
#else
#include <stdarg.h>
#include <sys/time.h>
#include <time.h>
#if !defined(_WIN32)
#include <netinet/tcp.h>
#ifdef __ANDROID_NDK__
#include <android/ndk-version.h>
#define ANDROID_OLD_NDK __NDK_MAJOR__ < 17
#endif
#endif
#include <fcntl.h>
#define initializeWinsockIfNecessary() 1
#endif
#if defined(__WIN32__) || defined(_WIN32) || defined(_QNX4)
#else
#include <signal.h>
#define USE_SIGNALS 1
#endif
#include <stdio.h>

// By default, use INADDR_ANY for the sending and receiving interfaces:
netAddressBits SendingInterfaceAddr = INADDR_ANY;
netAddressBits ReceivingInterfaceAddr = INADDR_ANY;

static void socketErr(UsageEnvironment& env, char const* errorMsg) {
    env.setResultErrMsg(errorMsg);
}

NoReuse::NoReuse(UsageEnvironment& env) : fEnv(env) {
    groupsockPriv(fEnv)->reuseFlag = 0;
}

NoReuse::~NoReuse() {
    groupsockPriv(fEnv)->reuseFlag = 1;
    reclaimGroupsockPriv(fEnv);
}

_groupsockPriv* groupsockPriv(UsageEnvironment& env) {
    if (env.groupsockPriv == NULL) {  // We need to create it
        _groupsockPriv* result = new _groupsockPriv;
        result->socketTable = NULL;
        result->reuseFlag =
                1;  // default value => allow reuse of socket numbers
        env.groupsockPriv = result;
    }
    return (_groupsockPriv*)(env.groupsockPriv);
}

void reclaimGroupsockPriv(UsageEnvironment& env) {
    _groupsockPriv* priv = (_groupsockPriv*)(env.groupsockPriv);
    if (priv->socketTable == NULL && priv->reuseFlag == 1 /*default value*/) {
        // We can delete the structure (to save space); it will get created
        // again, if needed:
        delete priv;
        env.groupsockPriv = NULL;
    }
}

static int createSocket(int type) {
    // Call "socket()" to create a (IPv4) socket of the specified type.
    // But also set it to have the 'close on exec' property (if we can)
    int sock;

#ifdef SOCK_CLOEXEC
    sock = socket(AF_INET, type | SOCK_CLOEXEC, 0);
    if (sock != -1 || errno != EINVAL) return sock;
        // An "errno" of EINVAL likely means that the system wasn't happy with
        // the SOCK_CLOEXEC; fall through and try again without it:
#endif

    sock = socket(AF_INET, type, 0);
#ifdef FD_CLOEXEC
    if (sock != -1) fcntl(sock, F_SETFD, FD_CLOEXEC);
#endif
    return sock;
}

int setupDatagramSocket(UsageEnvironment& env, Port port) {
    if (!initializeWinsockIfNecessary()) {
        socketErr(env, "Failed to initialize 'winsock': ");
        return -1;
    }

    int newSocket = createSocket(SOCK_DGRAM);
    if (newSocket < 0) {
        socketErr(env, "unable to create datagram socket: ");
        return newSocket;
    }

    int reuseFlag = groupsockPriv(env)->reuseFlag;
    reclaimGroupsockPriv(env);
    if (setsockopt(newSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuseFlag,
                   sizeof reuseFlag) < 0) {
        socketErr(env, "setsockopt(SO_REUSEADDR) error: ");
        closeSocket(newSocket);
        return -1;
    }

#if defined(__WIN32__) || defined(_WIN32)
    // Windoze doesn't properly handle SO_REUSEPORT or IP_MULTICAST_LOOP
#else
#ifdef SO_REUSEPORT
    if (setsockopt(newSocket, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuseFlag,
                   sizeof reuseFlag) < 0) {
        socketErr(env, "setsockopt(SO_REUSEPORT) error: ");
        closeSocket(newSocket);
        return -1;
    }
#endif

#ifdef IP_MULTICAST_LOOP
    const u_int8_t loop = 1;
    if (setsockopt(newSocket, IPPROTO_IP, IP_MULTICAST_LOOP, (const char*)&loop,
                   sizeof loop) < 0) {
        socketErr(env, "setsockopt(IP_MULTICAST_LOOP) error: ");
        closeSocket(newSocket);
        return -1;
    }
#endif
#endif

    // Note: Windoze requires binding, even if the port number is 0
    netAddressBits addr = INADDR_ANY;
#if defined(__WIN32__) || defined(_WIN32)
#else
    if (port.num() != 0 || ReceivingInterfaceAddr != INADDR_ANY) {
#endif
    if (port.num() == 0) addr = ReceivingInterfaceAddr;
    MAKE_SOCKADDR_IN(name, addr, port.num());
    if (bind(newSocket, (struct sockaddr*)&name, sizeof name) != 0) {
        char tmpBuffer[100];
        sprintf(tmpBuffer,
                "bind() error (port number: %d): ", ntohs(port.num()));
        socketErr(env, tmpBuffer);
        closeSocket(newSocket);
        return -1;
    }
#if defined(__WIN32__) || defined(_WIN32)
#else
    }
#endif

    // Set the sending interface for multicasts, if it's not the default:
    if (SendingInterfaceAddr != INADDR_ANY) {
        struct in_addr addr;
        addr.s_addr = SendingInterfaceAddr;

        if (setsockopt(newSocket, IPPROTO_IP, IP_MULTICAST_IF,
                       (const char*)&addr, sizeof addr) < 0) {
            socketErr(env, "error setting outgoing multicast interface: ");
            closeSocket(newSocket);
            return -1;
        }
    }

    return newSocket;
}

Boolean makeSocketNonBlocking(int sock) {
#if defined(__WIN32__) || defined(_WIN32)
    unsigned long arg = 1;
    return ioctlsocket(sock, FIONBIO, &arg) == 0;
#elif defined(VXWORKS)
    int arg = 1;
    return ioctl(sock, FIONBIO, (int)&arg) == 0;
#else
    int curFlags = fcntl(sock, F_GETFL, 0);
    return fcntl(sock, F_SETFL, curFlags | O_NONBLOCK) >= 0;
#endif
}

Boolean makeSocketBlocking(int sock, unsigned writeTimeoutInMilliseconds) {
    Boolean result;
#if defined(__WIN32__) || defined(_WIN32)
    unsigned long arg = 0;
    result = ioctlsocket(sock, FIONBIO, &arg) == 0;
#elif defined(VXWORKS)
    int arg = 0;
    result = ioctl(sock, FIONBIO, (int)&arg) == 0;
#else
    int curFlags = fcntl(sock, F_GETFL, 0);
    result = fcntl(sock, F_SETFL, curFlags & (~O_NONBLOCK)) >= 0;
#endif

    if (writeTimeoutInMilliseconds > 0) {
#ifdef SO_SNDTIMEO
#if defined(__WIN32__) || defined(_WIN32)
        DWORD msto = (DWORD)writeTimeoutInMilliseconds;
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&msto, sizeof(msto));
#else
        struct timeval tv;
        tv.tv_sec = writeTimeoutInMilliseconds / 1000;
        tv.tv_usec = (writeTimeoutInMilliseconds % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, sizeof tv);
#endif
#endif
    }

    return result;
}

Boolean setSocketKeepAlive(int sock) {
#if defined(__WIN32__) || defined(_WIN32)
    // How do we do this in Windows?  For now, just make this a no-op in
    // Windows:
#else
    int const keepalive_enabled = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (void*)&keepalive_enabled,
                   sizeof keepalive_enabled) < 0) {
        return False;
    }

#ifdef TCP_KEEPIDLE
    int const keepalive_time = 180;
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, (void*)&keepalive_time,
                   sizeof keepalive_time) < 0) {
        return False;
    }
#endif

    int const keepalive_count = 5;
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, (void*)&keepalive_count,
                   sizeof keepalive_count) < 0) {
        return False;
    }

    int const keepalive_interval = 20;
    if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, (void*)&keepalive_interval,
                   sizeof keepalive_interval) < 0) {
        return False;
    }
#endif

    return True;
}

int setupStreamSocket(UsageEnvironment& env,
                      Port port,
                      Boolean makeNonBlocking,
                      Boolean setKeepAlive) {
    if (!initializeWinsockIfNecessary()) {
        socketErr(env, "Failed to initialize 'winsock': ");
        return -1;
    }

    int newSocket = createSocket(SOCK_STREAM);
    if (newSocket < 0) {
        socketErr(env, "unable to create stream socket: ");
        return newSocket;
    }

    int reuseFlag = groupsockPriv(env)->reuseFlag;
    reclaimGroupsockPriv(env);
    if (setsockopt(newSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuseFlag,
                   sizeof reuseFlag) < 0) {
        socketErr(env, "setsockopt(SO_REUSEADDR) error: ");
        closeSocket(newSocket);
        return -1;
    }

    // SO_REUSEPORT doesn't really make sense for TCP sockets, so we
    // normally don't set them.  However, if you really want to do this
    // #define REUSE_FOR_TCP
#ifdef REUSE_FOR_TCP
#if defined(__WIN32__) || defined(_WIN32)
    // Windoze doesn't properly handle SO_REUSEPORT
#else
#ifdef SO_REUSEPORT
    if (setsockopt(newSocket, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuseFlag,
                   sizeof reuseFlag) < 0) {
        socketErr(env, "setsockopt(SO_REUSEPORT) error: ");
        closeSocket(newSocket);
        return -1;
    }
#endif
#endif
#endif

    // Note: Windoze requires binding, even if the port number is 0
#if defined(__WIN32__) || defined(_WIN32)
#else
    if (port.num() != 0 || ReceivingInterfaceAddr != INADDR_ANY) {
#endif
    MAKE_SOCKADDR_IN(name, ReceivingInterfaceAddr, port.num());
    if (bind(newSocket, (struct sockaddr*)&name, sizeof name) != 0) {
        char tmpBuffer[100];
        sprintf(tmpBuffer,
                "bind() error (port number: %d): ", ntohs(port.num()));
        socketErr(env, tmpBuffer);
        closeSocket(newSocket);
        return -1;
    }
#if defined(__WIN32__) || defined(_WIN32)
#else
    }
#endif

    if (makeNonBlocking) {
        if (!makeSocketNonBlocking(newSocket)) {
            socketErr(env, "failed to make non-blocking: ");
            closeSocket(newSocket);
            return -1;
        }
    }

    // Set the keep alive mechanism for the TCP socket, to avoid "ghost sockets"
    //    that remain after an interrupted communication.
    if (setKeepAlive) {
        if (!setSocketKeepAlive(newSocket)) {
            socketErr(env, "failed to set keep alive: ");
            closeSocket(newSocket);
            return -1;
        }
    }

    return newSocket;
}

int readSocket(UsageEnvironment& env,
               int socket,
               unsigned char* buffer,
               unsigned bufferSize,
               struct sockaddr_in& fromAddress) {
    SOCKLEN_T addressSize = sizeof fromAddress;
    int bytesRead = recvfrom(socket, (char*)buffer, bufferSize, 0,
                             (struct sockaddr*)&fromAddress, &addressSize);
    if (bytesRead < 0) {
        //##### HACK to work around bugs in Linux and Windows:
        int err = env.getErrno();
        if (err == 111 /*ECONNREFUSED (Linux)*/
#if defined(__WIN32__) || defined(_WIN32)
            // What a piece of crap Windows is.  Sometimes
            // recvfrom() returns -1, but with an 'errno' of 0.
            // This appears not to be a real error; just treat
            // it as if it were a read of zero bytes, and hope
            // we don't have to do anything else to 'reset'
            // this alleged error:
            || err == 0 || err == EWOULDBLOCK
#else
            || err == EAGAIN
#endif
            ||
            err == 113 /*EHOSTUNREACH (Linux)*/) {  // Why does Linux return
                                                    // this for datagram sock?
            fromAddress.sin_addr.s_addr = 0;
            return 0;
        }
        //##### END HACK
        socketErr(env, "recvfrom() error: ");
    } else if (bytesRead == 0) {
        // "recvfrom()" on a stream socket can return 0 if the remote end has
        // closed the connection.  Treat this as an error:
        return -1;
    }

    return bytesRead;
}

Boolean writeSocket(UsageEnvironment& env,
                    int socket,
                    struct in_addr address,
                    portNumBits portNum,
                    u_int8_t ttlArg,
                    unsigned char* buffer,
                    unsigned bufferSize) {
    // Before sending, set the socket's TTL:
#if defined(__WIN32__) || defined(_WIN32)
#define TTL_TYPE int
#else
#define TTL_TYPE u_int8_t
#endif
    TTL_TYPE ttl = (TTL_TYPE)ttlArg;
    if (setsockopt(socket, IPPROTO_IP, IP_MULTICAST_TTL, (const char*)&ttl,
                   sizeof ttl) < 0) {
        socketErr(env, "setsockopt(IP_MULTICAST_TTL) error: ");
        return False;
    }

    return writeSocket(env, socket, address, portNum, buffer, bufferSize);
}

Boolean writeSocket(UsageEnvironment& env,
                    int socket,
                    struct in_addr address,
                    portNumBits portNum,
                    unsigned char* buffer,
                    unsigned bufferSize) {
    do {
        MAKE_SOCKADDR_IN(dest, address.s_addr, portNum);
        int bytesSent = sendto(socket, (char*)buffer, bufferSize, 0,
                               (struct sockaddr*)&dest, sizeof dest);
        if (bytesSent != (int)bufferSize) {
            char tmpBuf[100];
            sprintf(tmpBuf,
                    "writeSocket(%d), sendTo() error: wrote %d bytes instead "
                    "of %u: ",
                    socket, bytesSent, bufferSize);
            socketErr(env, tmpBuf);
            break;
        }

        return True;
    } while (0);

    return False;
}

void ignoreSigPipeOnSocket(int socketNum) {
#ifdef USE_SIGNALS
#ifdef SO_NOSIGPIPE
    int set_option = 1;
    setsockopt(socketNum, SOL_SOCKET, SO_NOSIGPIPE, &set_option,
               sizeof set_option);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
#endif
}

static unsigned getBufferSize(UsageEnvironment& env,
                              int bufOptName,
                              int socket) {
    unsigned curSize;
    SOCKLEN_T sizeSize = sizeof curSize;
    if (getsockopt(socket, SOL_SOCKET, bufOptName, (char*)&curSize, &sizeSize) <
        0) {
        socketErr(env, "getBufferSize() error: ");
        return 0;
    }

    return curSize;
}
unsigned getSendBufferSize(UsageEnvironment& env, int socket) {
    return getBufferSize(env, SO_SNDBUF, socket);
}
unsigned getReceiveBufferSize(UsageEnvironment& env, int socket) {
    return getBufferSize(env, SO_RCVBUF, socket);
}

static unsigned setBufferTo(UsageEnvironment& env,
                            int bufOptName,
                            int socket,
                            unsigned requestedSize) {
    SOCKLEN_T sizeSize = sizeof requestedSize;
    setsockopt(socket, SOL_SOCKET, bufOptName, (char*)&requestedSize, sizeSize);

    // Get and return the actual, resulting buffer size:
    return getBufferSize(env, bufOptName, socket);
}
unsigned setSendBufferTo(UsageEnvironment& env,
                         int socket,
                         unsigned requestedSize) {
    return setBufferTo(env, SO_SNDBUF, socket, requestedSize);
}
unsigned setReceiveBufferTo(UsageEnvironment& env,
                            int socket,
                            unsigned requestedSize) {
    return setBufferTo(env, SO_RCVBUF, socket, requestedSize);
}

static unsigned increaseBufferTo(UsageEnvironment& env,
                                 int bufOptName,
                                 int socket,
                                 unsigned requestedSize) {
    // First, get the current buffer size.  If it's already at least
    // as big as what we're requesting, do nothing.
    unsigned curSize = getBufferSize(env, bufOptName, socket);

    // Next, try to increase the buffer to the requested size,
    // or to some smaller size, if that's not possible:
    while (requestedSize > curSize) {
        SOCKLEN_T sizeSize = sizeof requestedSize;
        if (setsockopt(socket, SOL_SOCKET, bufOptName, (char*)&requestedSize,
                       sizeSize) >= 0) {
            // success
            return requestedSize;
        }
        requestedSize = (requestedSize + curSize) / 2;
    }

    return getBufferSize(env, bufOptName, socket);
}
unsigned increaseSendBufferTo(UsageEnvironment& env,
                              int socket,
                              unsigned requestedSize) {
    return increaseBufferTo(env, SO_SNDBUF, socket, requestedSize);
}
unsigned increaseReceiveBufferTo(UsageEnvironment& env,
                                 int socket,
                                 unsigned requestedSize) {
    return increaseBufferTo(env, SO_RCVBUF, socket, requestedSize);
}

static void clearMulticastAllSocketOption(int socket) {
#ifdef IP_MULTICAST_ALL
    // This option is defined in modern versions of Linux to overcome a bug in
    // the Linux kernel's default behavior. When set to 0, it ensures that we
    // receive only packets that were sent to the specified IP multicast
    // address, even if some other process on the same system has joined a
    // different multicast group with the same port number.
    int multicastAll = 0;
    (void)setsockopt(socket, IPPROTO_IP, IP_MULTICAST_ALL, (void*)&multicastAll,
                     sizeof multicastAll);
    // Ignore the call's result.  Should it fail, we'll still receive packets
    // (just perhaps more than intended)
#endif
}

Boolean socketJoinGroup(UsageEnvironment& env,
                        int socket,
                        netAddressBits groupAddress) {
    if (!IsMulticastAddress(groupAddress)) return True;  // ignore this case

    struct ip_mreq imr;
    imr.imr_multiaddr.s_addr = groupAddress;
    imr.imr_interface.s_addr = ReceivingInterfaceAddr;
    if (setsockopt(socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char*)&imr,
                   sizeof(struct ip_mreq)) < 0) {
#if defined(__WIN32__) || defined(_WIN32)
        if (env.getErrno() != 0) {
            // That piece-of-shit toy operating system (Windows) sometimes lies
            // about setsockopt() failing!
#endif
            socketErr(env, "setsockopt(IP_ADD_MEMBERSHIP) error: ");
            return False;
#if defined(__WIN32__) || defined(_WIN32)
        }
#endif
    }

    clearMulticastAllSocketOption(socket);

    return True;
}

Boolean socketLeaveGroup(UsageEnvironment&,
                         int socket,
                         netAddressBits groupAddress) {
    if (!IsMulticastAddress(groupAddress)) return True;  // ignore this case

    struct ip_mreq imr;
    imr.imr_multiaddr.s_addr = groupAddress;
    imr.imr_interface.s_addr = ReceivingInterfaceAddr;
    if (setsockopt(socket, IPPROTO_IP, IP_DROP_MEMBERSHIP, (const char*)&imr,
                   sizeof(struct ip_mreq)) < 0) {
        return False;
    }

    return True;
}

// The source-specific join/leave operations require special setsockopt()
// commands, and a special structure (ip_mreq_source).  If the include files
// didn't define these, we do so here:
#if !defined(IP_ADD_SOURCE_MEMBERSHIP)
struct ip_mreq_source {
    struct in_addr imr_multiaddr;  /* IP multicast address of group */
    struct in_addr imr_sourceaddr; /* IP address of source */
    struct in_addr imr_interface;  /* local IP address of interface */
};
#endif

#ifndef IP_ADD_SOURCE_MEMBERSHIP

#ifdef LINUX
#define IP_ADD_SOURCE_MEMBERSHIP 39
#define IP_DROP_SOURCE_MEMBERSHIP 40
#else
#define IP_ADD_SOURCE_MEMBERSHIP 25
#define IP_DROP_SOURCE_MEMBERSHIP 26
#endif

#endif

Boolean socketJoinGroupSSM(UsageEnvironment& env,
                           int socket,
                           netAddressBits groupAddress,
                           netAddressBits sourceFilterAddr) {
    if (!IsMulticastAddress(groupAddress)) return True;  // ignore this case

    struct ip_mreq_source imr;
#if ANDROID_OLD_NDK
    imr.imr_multiaddr = groupAddress;
    imr.imr_sourceaddr = sourceFilterAddr;
    imr.imr_interface = ReceivingInterfaceAddr;
#else
    imr.imr_multiaddr.s_addr = groupAddress;
    imr.imr_sourceaddr.s_addr = sourceFilterAddr;
    imr.imr_interface.s_addr = ReceivingInterfaceAddr;
#endif
    if (setsockopt(socket, IPPROTO_IP, IP_ADD_SOURCE_MEMBERSHIP,
                   (const char*)&imr, sizeof(struct ip_mreq_source)) < 0) {
        socketErr(env, "setsockopt(IP_ADD_SOURCE_MEMBERSHIP) error: ");
        return False;
    }

    clearMulticastAllSocketOption(socket);

    return True;
}

Boolean socketLeaveGroupSSM(UsageEnvironment& /*env*/,
                            int socket,
                            netAddressBits groupAddress,
                            netAddressBits sourceFilterAddr) {
    if (!IsMulticastAddress(groupAddress)) return True;  // ignore this case

    struct ip_mreq_source imr;
#if ANDROID_OLD_NDK
    imr.imr_multiaddr = groupAddress;
    imr.imr_sourceaddr = sourceFilterAddr;
    imr.imr_interface = ReceivingInterfaceAddr;
#else
    imr.imr_multiaddr.s_addr = groupAddress;
    imr.imr_sourceaddr.s_addr = sourceFilterAddr;
    imr.imr_interface.s_addr = ReceivingInterfaceAddr;
#endif
    if (setsockopt(socket, IPPROTO_IP, IP_DROP_SOURCE_MEMBERSHIP,
                   (const char*)&imr, sizeof(struct ip_mreq_source)) < 0) {
        return False;
    }

    return True;
}

static Boolean getSourcePort0(int socket,
                              portNumBits& resultPortNum /*host order*/) {
    sockaddr_in test;
    test.sin_port = 0;
    SOCKLEN_T len = sizeof test;
    if (getsockname(socket, (struct sockaddr*)&test, &len) < 0) return False;

    resultPortNum = ntohs(test.sin_port);
    return True;
}

Boolean getSourcePort(UsageEnvironment& env, int socket, Port& port) {
    portNumBits portNum = 0;
    if (!getSourcePort0(socket, portNum) || portNum == 0) {
        // Hack - call bind(), then try again:
        MAKE_SOCKADDR_IN(name, INADDR_ANY, 0);
        bind(socket, (struct sockaddr*)&name, sizeof name);

        if (!getSourcePort0(socket, portNum) || portNum == 0) {
            socketErr(env, "getsockname() error: ");
            return False;
        }
    }

    port = Port(portNum);
    return True;
}

static Boolean badAddressForUs(netAddressBits addr) {
    // Check for some possible erroneous addresses:
    netAddressBits nAddr = htonl(addr);
    return (nAddr == 0x7F000001 /* 127.0.0.1 */
            || nAddr == 0 || nAddr == (netAddressBits)(~0));
}

Boolean loopbackWorks = 1;

netAddressBits ourIPAddress(UsageEnvironment& env) {
    static netAddressBits ourAddress = 0;
    int sock = -1;
    struct in_addr testAddr;

    if (ReceivingInterfaceAddr != INADDR_ANY) {
        // Hack: If we were told to receive on a specific interface address,
        // then define this to be our ip address:
        ourAddress = ReceivingInterfaceAddr;
    }

    if (ourAddress == 0) {
        // We need to find our source address
        struct sockaddr_in fromAddr;
        fromAddr.sin_addr.s_addr = 0;

        // Get our address by sending a (0-TTL) multicast packet,
        // receiving it, and looking at the source address used.
        // (This is kinda bogus, but it provides the best guarantee
        // that other nodes will think our address is the same as we do.)
        do {
            loopbackWorks = 0;  // until we learn otherwise

#ifndef DISABLE_LOOPBACK_IP_ADDRESS_CHECK
            testAddr.s_addr = our_inet_addr("228.67.43.91");  // arbitrary
            Port testPort(15947);                             // ditto

            sock = setupDatagramSocket(env, testPort);
            if (sock < 0) break;

            if (!socketJoinGroup(env, sock, testAddr.s_addr)) break;

            unsigned char testString[] = "hostIdTest";
            unsigned testStringLength = sizeof testString;

            if (!writeSocket(env, sock, testAddr, testPort.num(), 0, testString,
                             testStringLength))
                break;

            // Block until the socket is readable (with a 5-second timeout):
            fd_set rd_set;
            FD_ZERO(&rd_set);
            FD_SET((unsigned)sock, &rd_set);
            const unsigned numFds = sock + 1;
            struct timeval timeout;
            timeout.tv_sec = 5;
            timeout.tv_usec = 0;
            int result = select(numFds, &rd_set, NULL, NULL, &timeout);
            if (result <= 0) break;

            unsigned char readBuffer[20];
            int bytesRead = readSocket(env, sock, readBuffer, sizeof readBuffer,
                                       fromAddr);
            if (bytesRead != (int)testStringLength ||
                strncmp((char*)readBuffer, (char*)testString,
                        testStringLength) != 0) {
                break;
            }

            // We use this packet's source address, if it's good:
            loopbackWorks = !badAddressForUs(fromAddr.sin_addr.s_addr);
#endif
        } while (0);

        if (sock >= 0) {
            socketLeaveGroup(env, sock, testAddr.s_addr);
            closeSocket(sock);
        }

        if (!loopbackWorks) do {
                // We couldn't find our address using multicast loopback,
                // so try instead to look it up directly - by first getting our
                // host name, and then resolving this host name
                char hostname[100];
                hostname[0] = '\0';
                int result = gethostname(hostname, sizeof hostname);
                if (result != 0 || hostname[0] == '\0') {
                    env.setResultErrMsg("initial gethostname() failed");
                    break;
                }

                // Try to resolve "hostname" to an IP address:
                NetAddressList addresses(hostname);
                NetAddressList::Iterator iter(addresses);
                NetAddress const* address;

                // Take the first address that's not bad:
                netAddressBits addr = 0;
                while ((address = iter.nextAddress()) != NULL) {
                    netAddressBits a = *(netAddressBits*)(address->data());
                    if (!badAddressForUs(a)) {
                        addr = a;
                        break;
                    }
                }

                // Assign the address that we found to "fromAddr" (as if the
                // 'loopback' method had worked), to simplify the code below:
                fromAddr.sin_addr.s_addr = addr;
            } while (0);

        // Make sure we have a good address:
        netAddressBits from = fromAddr.sin_addr.s_addr;
        if (badAddressForUs(from)) {
            char tmp[100];
            sprintf(tmp, "This computer has an invalid IP address: %s",
                    AddressString(from).val());
            env.setResultMsg(tmp);
            from = 0;
        }

        ourAddress = from;

        // Use our newly-discovered IP address, and the current time,
        // to initialize the random number generator's seed:
        struct timeval timeNow;
        gettimeofday(&timeNow, NULL);
        unsigned seed = ourAddress ^ timeNow.tv_sec ^ timeNow.tv_usec;
        our_srandom(seed);
    }
    return ourAddress;
}

netAddressBits chooseRandomIPv4SSMAddress(UsageEnvironment& env) {
    // First, a hack to ensure that our random number generator is seeded:
    (void)ourIPAddress(env);

    // Choose a random address in the range [232.0.1.0, 232.255.255.255)
    // i.e., [0xE8000100, 0xE8FFFFFF)
    netAddressBits const first = 0xE8000100, lastPlus1 = 0xE8FFFFFF;
    netAddressBits const range = lastPlus1 - first;

    return ntohl(first + ((netAddressBits)our_random()) % range);
}

char const* timestampString() {
    struct timeval tvNow;
    gettimeofday(&tvNow, NULL);

#if !defined(_WIN32_WCE)
    static char timeString[9];  // holds hh:mm:ss plus trailing '\0'

    time_t tvNow_t = tvNow.tv_sec;
    char const* ctimeResult = ctime(&tvNow_t);
    if (ctimeResult == NULL) {
        sprintf(timeString, "??:??:??");
    } else {
        char const* from = &ctimeResult[11];
        int i;
        for (i = 0; i < 8; ++i) {
            timeString[i] = from[i];
        }
        timeString[i] = '\0';
    }
#else
    // WinCE apparently doesn't have "ctime()", so instead, construct
    // a timestamp string just using the integer and fractional parts
    // of "tvNow":
    static char timeString[50];
    sprintf(timeString, "%lu.%06ld", tvNow.tv_sec, tvNow.tv_usec);
#endif

    return (char const*)&timeString;
}

#if (defined(__WIN32__) || defined(_WIN32)) && !defined(__MINGW32__)
// For Windoze, we need to implement our own gettimeofday()

// used to make sure that static variables in gettimeofday() aren't initialized
// simultaneously by multiple threads
static LONG initializeLock_gettimeofday = 0;

#if !defined(_WIN32_WCE)
#include <sys/timeb.h>
#endif

int gettimeofday(struct timeval* tp, int* /*tz*/) {
    static LARGE_INTEGER tickFrequency, epochOffset;

    static Boolean isInitialized = False;

    LARGE_INTEGER tickNow;

#if !defined(_WIN32_WCE)
    QueryPerformanceCounter(&tickNow);
#else
    tickNow.QuadPart = GetTickCount();
#endif

    if (!isInitialized) {
        if (1 == InterlockedIncrement(&initializeLock_gettimeofday)) {
#if !defined(_WIN32_WCE)
            // For our first call, use "ftime()", so that we get a time with a
            // proper epoch. For subsequent calls, use
            // "QueryPerformanceCount()", because it's more fine-grain.
            struct timeb tb;
            ftime(&tb);
            tp->tv_sec = tb.time;
            tp->tv_usec = 1000 * tb.millitm;

            // Also get our counter frequency:
            QueryPerformanceFrequency(&tickFrequency);
#else
            /* FILETIME of Jan 1 1970 00:00:00. */
            const LONGLONG epoch = 116444736000000000LL;
            FILETIME fileTime;
            LARGE_INTEGER time;
            GetSystemTimeAsFileTime(&fileTime);

            time.HighPart = fileTime.dwHighDateTime;
            time.LowPart = fileTime.dwLowDateTime;

            // convert to from 100ns time to unix timestamp in seconds,
            // 1000*1000*10
            tp->tv_sec = (long)((time.QuadPart - epoch) / 10000000L);

            /*
              GetSystemTimeAsFileTime has just a seconds resolution,
              thats why wince-version of gettimeofday is not 100% accurate, usec
              accuracy would be calculated like this:
              // convert 100 nanoseconds to usec
              tp->tv_usec= (long)((time.QuadPart - epoch)%10000000L) / 10L;
            */
            tp->tv_usec = 0;

            // resolution of GetTickCounter() is always milliseconds
            tickFrequency.QuadPart = 1000;
#endif
            // compute an offset to add to subsequent counter times, so we get a
            // proper epoch:
            epochOffset.QuadPart =
                    tp->tv_sec * tickFrequency.QuadPart +
                    (tp->tv_usec * tickFrequency.QuadPart) / 1000000L -
                    tickNow.QuadPart;

            // next caller can use ticks for time calculation
            isInitialized = True;
            return 0;
        } else {
            InterlockedDecrement(&initializeLock_gettimeofday);
            // wait until first caller has initialized static values
            while (!isInitialized) {
                Sleep(1);
            }
        }
    }

    // adjust our tick count so that we get a proper epoch:
    tickNow.QuadPart += epochOffset.QuadPart;

    tp->tv_sec = (long)(tickNow.QuadPart / tickFrequency.QuadPart);
    tp->tv_usec =
            (long)(((tickNow.QuadPart % tickFrequency.QuadPart) * 1000000L) /
                   tickFrequency.QuadPart);

    return 0;
}
#endif
#undef ANDROID_OLD_NDK
