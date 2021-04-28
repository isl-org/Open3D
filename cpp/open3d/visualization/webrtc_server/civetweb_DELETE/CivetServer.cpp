/* Copyright (c) 2013-2017 the Civetweb developers
 * Copyright (c) 2013 No Face Press, LLC
 *
 * License http://opensource.org/licenses/mit-license.php MIT License
 */

#include "CivetServer.h"

#include <assert.h>
#include <string.h>

#include <stdexcept>

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) (void)(x)
#endif

#ifndef MAX_PARAM_BODY_LENGTH
// Set a default limit for parameters in a form body: 2 MB
#define MAX_PARAM_BODY_LENGTH (1024 * 1024 * 2)
#endif

bool CivetHandler::handleGet(CivetServer *server, struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handlePost(CivetServer *server, struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handleHead(CivetServer *server, struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handlePut(CivetServer *server, struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handlePatch(CivetServer *server,
                               struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handleDelete(CivetServer *server,
                                struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetHandler::handleOptions(CivetServer *server,
                                 struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return false;
}

bool CivetWebSocketHandler::handleConnection(CivetServer *server,
                                             const struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return true;
}

void CivetWebSocketHandler::handleReadyState(CivetServer *server,
                                             struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return;
}

bool CivetWebSocketHandler::handleData(CivetServer *server,
                                       struct mg_connection *conn,
                                       int bits,
                                       char *data,
                                       size_t data_len) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    UNUSED_PARAMETER(bits);
    UNUSED_PARAMETER(data);
    UNUSED_PARAMETER(data_len);
    return true;
}

void CivetWebSocketHandler::handleClose(CivetServer *server,
                                        const struct mg_connection *conn) {
    UNUSED_PARAMETER(server);
    UNUSED_PARAMETER(conn);
    return;
}

int CivetServer::requestHandler(struct mg_connection *conn, void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return 0;

    mg_lock_context(me->context);
    me->connections[conn] = CivetConnection();
    mg_unlock_context(me->context);

    CivetHandler *handler = (CivetHandler *)cbdata;

    if (handler) {
        if (strcmp(request_info->request_method, "GET") == 0) {
            return handler->handleGet(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "POST") == 0) {
            return handler->handlePost(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "HEAD") == 0) {
            return handler->handleHead(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "PUT") == 0) {
            return handler->handlePut(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "DELETE") == 0) {
            return handler->handleDelete(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "OPTIONS") == 0) {
            return handler->handleOptions(me, conn) ? 1 : 0;
        } else if (strcmp(request_info->request_method, "PATCH") == 0) {
            return handler->handlePatch(me, conn) ? 1 : 0;
        }
    }

    return 0;  // No handler found
}

int CivetServer::authHandler(struct mg_connection *conn, void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return 0;

    mg_lock_context(me->context);
    me->connections[conn] = CivetConnection();
    mg_unlock_context(me->context);

    CivetAuthHandler *handler = (CivetAuthHandler *)cbdata;

    if (handler) {
        return handler->authorize(me, conn) ? 1 : 0;
    }

    return 0;  // No handler found
}

int CivetServer::webSocketConnectionHandler(const struct mg_connection *conn,
                                            void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return 0;

    CivetWebSocketHandler *handler = (CivetWebSocketHandler *)cbdata;

    if (handler) {
        return handler->handleConnection(me, conn) ? 0 : 1;
    }

    return 1;  // No handler found, close connection
}

void CivetServer::webSocketReadyHandler(struct mg_connection *conn,
                                        void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return;

    CivetWebSocketHandler *handler = (CivetWebSocketHandler *)cbdata;

    if (handler) {
        handler->handleReadyState(me, conn);
    }
}

int CivetServer::webSocketDataHandler(struct mg_connection *conn,
                                      int bits,
                                      char *data,
                                      size_t data_len,
                                      void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return 0;

    CivetWebSocketHandler *handler = (CivetWebSocketHandler *)cbdata;

    if (handler) {
        return handler->handleData(me, conn, bits, data, data_len) ? 1 : 0;
    }

    return 1;  // No handler found
}

void CivetServer::webSocketCloseHandler(const struct mg_connection *conn,
                                        void *cbdata) {
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    assert(request_info != NULL);
    CivetServer *me = (CivetServer *)(request_info->user_data);
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return;

    CivetWebSocketHandler *handler = (CivetWebSocketHandler *)cbdata;

    if (handler) {
        handler->handleClose(me, conn);
    }
}

CivetCallbacks::CivetCallbacks() { memset(this, 0, sizeof(*this)); }

CivetServer::CivetServer(const char **options,
                         const struct CivetCallbacks *_callbacks,
                         const void *UserContextIn)
    : context(0) {
    struct CivetCallbacks callbacks;

    UserContext = UserContextIn;

    if (_callbacks) {
        callbacks = *_callbacks;
        userCloseHandler = _callbacks->connection_close;
    } else {
        userCloseHandler = NULL;
    }
    callbacks.connection_close = closeHandler;
    context = mg_start(&callbacks, this, options);
    if (context == NULL)
        throw CivetException(
                "null context when constructing CivetServer. "
                "Possible problem binding to port.");
}

CivetServer::CivetServer(const std::vector<std::string> &options,
                         const struct CivetCallbacks *_callbacks,
                         const void *UserContextIn)
    : context(0) {
    struct CivetCallbacks callbacks;

    UserContext = UserContextIn;

    if (_callbacks) {
        callbacks = *_callbacks;
        userCloseHandler = _callbacks->connection_close;
    } else {
        userCloseHandler = NULL;
    }
    callbacks.connection_close = closeHandler;

    std::vector<const char *> pointers(options.size() + 1);
    for (size_t i = 0; i < options.size(); i++) {
        pointers[i] = (options[i].c_str());
    }
    pointers.back() = NULL;

    context = mg_start(&callbacks, this, &pointers[0]);
    if (context == NULL)
        throw CivetException(
                "null context when constructing CivetServer. "
                "Possible problem binding to port.");
}

CivetServer::~CivetServer() { close(); }

void CivetServer::closeHandler(const struct mg_connection *conn) {
    CivetServer *me = (CivetServer *)mg_get_user_data(mg_get_context(conn));
    assert(me != NULL);

    // Happens when a request hits the server before the context is saved
    if (me->context == NULL) return;

    if (me->userCloseHandler) {
        me->userCloseHandler(conn);
    }
    mg_lock_context(me->context);
    me->connections.erase(conn);
    mg_unlock_context(me->context);
}

void CivetServer::addHandler(const std::string &uri, CivetHandler *handler) {
    mg_set_request_handler(context, uri.c_str(), requestHandler, handler);
}

void CivetServer::addWebSocketHandler(const std::string &uri,
                                      CivetWebSocketHandler *handler) {
    mg_set_websocket_handler(context, uri.c_str(), webSocketConnectionHandler,
                             webSocketReadyHandler, webSocketDataHandler,
                             webSocketCloseHandler, handler);
}

void CivetServer::addAuthHandler(const std::string &uri,
                                 CivetAuthHandler *handler) {
    mg_set_auth_handler(context, uri.c_str(), authHandler, handler);
}

void CivetServer::removeHandler(const std::string &uri) {
    mg_set_request_handler(context, uri.c_str(), NULL, NULL);
}

void CivetServer::removeWebSocketHandler(const std::string &uri) {
    mg_set_websocket_handler(context, uri.c_str(), NULL, NULL, NULL, NULL,
                             NULL);
}

void CivetServer::removeAuthHandler(const std::string &uri) {
    mg_set_auth_handler(context, uri.c_str(), NULL, NULL);
}

void CivetServer::close() {
    if (context) {
        mg_stop(context);
        context = 0;
    }
}

int CivetServer::getCookie(struct mg_connection *conn,
                           const std::string &cookieName,
                           std::string &cookieValue) {
    // Maximum cookie length as per microsoft is 4096.
    // http://msdn.microsoft.com/en-us/library/ms178194.aspx
    char _cookieValue[4096];
    const char *cookie = mg_get_header(conn, "Cookie");
    int lRead = mg_get_cookie(cookie, cookieName.c_str(), _cookieValue,
                              sizeof(_cookieValue));
    cookieValue.clear();
    cookieValue.append(_cookieValue);
    return lRead;
}

const char *CivetServer::getHeader(struct mg_connection *conn,
                                   const std::string &headerName) {
    return mg_get_header(conn, headerName.c_str());
}

void CivetServer::urlDecode(const char *src,
                            std::string &dst,
                            bool is_form_url_encoded) {
    urlDecode(src, strlen(src), dst, is_form_url_encoded);
}

void CivetServer::urlDecode(const char *src,
                            size_t src_len,
                            std::string &dst,
                            bool is_form_url_encoded) {
    // assign enough buffer
    std::vector<char> buf(src_len + 1);
    int r = mg_url_decode(src, static_cast<int>(src_len), &buf[0],
                          static_cast<int>(buf.size()), is_form_url_encoded);
    if (r < 0) {
        // never reach here
        throw std::out_of_range("");
    }
    // dst can contain NUL characters
    dst.assign(buf.begin(), buf.begin() + r);
}

bool CivetServer::getParam(struct mg_connection *conn,
                           const char *name,
                           std::string &dst,
                           size_t occurrence) {
    const char *formParams = NULL;
    const char *queryString = NULL;
    const struct mg_request_info *ri = mg_get_request_info(conn);
    assert(ri != NULL);
    CivetServer *me = (CivetServer *)(ri->user_data);
    assert(me != NULL);
    mg_lock_context(me->context);
    CivetConnection &conobj = me->connections[conn];
    mg_unlock_context(me->context);

    mg_lock_connection(conn);
    if (conobj.postData.empty()) {
        // check if there is a request body
        for (;;) {
            char buf[2048];
            int r = mg_read(conn, buf, sizeof(buf));
            try {
                if (r == 0) {
                    conobj.postData.push_back('\0');
                    break;
                } else if ((r < 0) || ((conobj.postData.size() + r) >
                                       MAX_PARAM_BODY_LENGTH)) {
                    conobj.postData.assign(1, '\0');
                    break;
                }
                conobj.postData.insert(conobj.postData.end(), buf, buf + r);
            } catch (...) {
                conobj.postData.clear();
                break;
            }
        }
    }
    if (!conobj.postData.empty()) {
        // check if form parameter are already stored
        formParams = &conobj.postData[0];
    }

    if (ri->query_string != NULL) {
        // get requests do store html <form> field values in the http
        // query_string
        queryString = ri->query_string;
    }

    mg_unlock_connection(conn);

    bool get_param_success = false;
    if (formParams != NULL) {
        get_param_success =
                getParam(formParams, strlen(formParams), name, dst, occurrence);
    }
    if (!get_param_success && queryString != NULL) {
        get_param_success = getParam(queryString, strlen(queryString), name,
                                     dst, occurrence);
    }

    return get_param_success;
}

bool CivetServer::getParam(const char *data,
                           size_t data_len,
                           const char *name,
                           std::string &dst,
                           size_t occurrence) {
    char buf[256];
    int r = mg_get_var2(data, data_len, name, buf, sizeof(buf), occurrence);
    if (r >= 0) {
        // dst can contain NUL characters
        dst.assign(buf, r);
        return true;
    } else if (r == -2) {
        // more buffer
        std::vector<char> vbuf(sizeof(buf) * 2);
        for (;;) {
            r = mg_get_var2(data, data_len, name, &vbuf[0], vbuf.size(),
                            occurrence);
            if (r >= 0) {
                dst.assign(vbuf.begin(), vbuf.begin() + r);
                return true;
            } else if (r != -2) {
                break;
            }
            // more buffer
            vbuf.resize(vbuf.size() * 2);
        }
    }
    dst.clear();
    return false;
}

std::string CivetServer::getPostData(struct mg_connection *conn) {
    mg_lock_connection(conn);
    std::string postdata;
    char buf[2048];
    int r = mg_read(conn, buf, sizeof(buf));
    while (r > 0) {
        postdata.append(buf, r);
        r = mg_read(conn, buf, sizeof(buf));
    }
    mg_unlock_connection(conn);
    return postdata;
}

void CivetServer::urlEncode(const char *src, std::string &dst, bool append) {
    urlEncode(src, strlen(src), dst, append);
}

void CivetServer::urlEncode(const char *src,
                            size_t src_len,
                            std::string &dst,
                            bool append) {
    if (!append) dst.clear();

    for (; src_len > 0; src++, src_len--) {
        if (*src == '\0') {
            // src and dst can contain NUL characters without encoding
            dst.push_back(*src);
        } else {
            char buf[2] = {*src, '\0'};
            char dst_buf[4];
            if (mg_url_encode(buf, dst_buf, sizeof(dst_buf)) < 0) {
                // never reach here
                throw std::out_of_range("");
            }
            dst.append(dst_buf);
        }
    }
}

std::vector<int> CivetServer::getListeningPorts() {
    std::vector<struct mg_server_port> server_ports = getListeningPortsFull();

    std::vector<int> ports(server_ports.size());
    for (size_t i = 0; i < server_ports.size(); i++) {
        ports[i] = server_ports[i].port;
    }

    return ports;
}

std::vector<struct mg_server_port> CivetServer::getListeningPortsFull() {
    std::vector<struct mg_server_port> server_ports(8);
    for (;;) {
        int size = mg_get_server_ports(context,
                                       static_cast<int>(server_ports.size()),
                                       &server_ports[0]);
        if (size < static_cast<int>(server_ports.size())) {
            server_ports.resize(size < 0 ? 0 : size);
            break;
        }
        server_ports.resize(server_ports.size() * 2);
    }
    return server_ports;
}
