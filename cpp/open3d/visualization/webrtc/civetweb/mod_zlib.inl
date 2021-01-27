/* Experimental implementation for on-the-fly compression */
#if !defined(USE_ZLIB)
#error "This file must only be included, if USE_ZLIB is set"
#endif

#if !defined(MEM_LEVEL)
#define MEM_LEVEL (8)
#endif

static void *
zalloc(void *opaque, uInt items, uInt size)
{
	struct mg_connection *conn = (struct mg_connection *)opaque;
	void *ret = mg_calloc_ctx(items, size, conn->phys_ctx);

	return ret;
}


static void
zfree(void *opaque, void *address)
{
	struct mg_connection *conn = (struct mg_connection *)opaque;
	(void)conn; /* not required */

	mg_free(address);
}


static void
send_compressed_data(struct mg_connection *conn, struct mg_file *filep)
{

	int zret;
	z_stream zstream;
	int do_flush;
	unsigned bytes_avail;
	unsigned char in_buf[MG_BUF_LEN];
	unsigned char out_buf[MG_BUF_LEN];
	FILE *in_file = filep->access.fp;

	/* Prepare state buffer. User server context memory allocation. */
	memset(&zstream, 0, sizeof(zstream));
	zstream.zalloc = zalloc;
	zstream.zfree = zfree;
	zstream.opaque = (void *)conn;

	/* Initialize for GZIP compression (MAX_WBITS | 16) */
	zret = deflateInit2(&zstream,
	                    Z_BEST_COMPRESSION,
	                    Z_DEFLATED,
	                    MAX_WBITS | 16,
	                    MEM_LEVEL,
	                    Z_DEFAULT_STRATEGY);

	if (zret != Z_OK) {
		mg_cry_internal(conn,
		                "GZIP init failed (%i): %s",
		                zret,
		                (zstream.msg ? zstream.msg : "<no error message>"));
		deflateEnd(&zstream);
		return;
	}

	/* Read until end of file */
	do {
		zstream.avail_in = fread(in_buf, 1, MG_BUF_LEN, in_file);
		if (ferror(in_file)) {
			mg_cry_internal(conn, "fread failed: %s", strerror(ERRNO));
			(void)deflateEnd(&zstream);
			return;
		}

		do_flush = (feof(in_file) ? Z_FINISH : Z_NO_FLUSH);
		zstream.next_in = in_buf;

		/* run deflate() on input until output buffer not full, finish
		 * compression if all of source has been read in */
		do {
			zstream.avail_out = MG_BUF_LEN;
			zstream.next_out = out_buf;
			zret = deflate(&zstream, do_flush);

			if (zret == Z_STREAM_ERROR) {
				/* deflate error */
				zret = -97;
				break;
			}

			bytes_avail = MG_BUF_LEN - zstream.avail_out;
			if (bytes_avail) {
				if (mg_send_chunk(conn, (char *)out_buf, bytes_avail) < 0) {
					zret = -98;
					break;
				}
			}

		} while (zstream.avail_out == 0);

		if (zret < -90) {
			/* Forward write error */
			break;
		}

		if (zstream.avail_in != 0) {
			/* all input will be used, otherwise GZIP is incomplete */
			zret = -99;
			break;
		}

		/* done when last data in file processed */
	} while (do_flush != Z_FINISH);

	if (zret != Z_STREAM_END) {
		/* Error: We did not compress everything. */
		mg_cry_internal(conn,
		                "GZIP incomplete (%i): %s",
		                zret,
		                (zstream.msg ? zstream.msg : "<no error message>"));
	}

	deflateEnd(&zstream);

	/* Send "end of chunked data" marker */
	mg_write(conn, "0\r\n\r\n", 5);
}

#if defined(USE_WEBSOCKET) && defined(MG_EXPERIMENTAL_INTERFACES)
int
websocket_deflate_initialize(struct mg_connection *conn, int server)
{
	int zret =
	    deflateInit2(&conn->websocket_deflate_state,
	                 Z_BEST_COMPRESSION,
	                 Z_DEFLATED,
	                 server
	                     ? -1 * conn->websocket_deflate_server_max_windows_bits
	                     : -1 * conn->websocket_deflate_client_max_windows_bits,
	                 MEM_LEVEL,
	                 Z_DEFAULT_STRATEGY);
	if (zret != Z_OK) {
		mg_cry_internal(conn,
		                "Websocket deflate init failed (%i): %s",
		                zret,
		                (conn->websocket_deflate_state.msg
		                     ? conn->websocket_deflate_state.msg
		                     : "<no error message>"));
		deflateEnd(&conn->websocket_deflate_state);
		return zret;
	}

	zret = inflateInit2(
	    &conn->websocket_inflate_state,
	    server ? -1 * conn->websocket_deflate_client_max_windows_bits
	           : -1 * conn->websocket_deflate_server_max_windows_bits);
	if (zret != Z_OK) {
		mg_cry_internal(conn,
		                "Websocket inflate init failed (%i): %s",
		                zret,
		                (conn->websocket_inflate_state.msg
		                     ? conn->websocket_inflate_state.msg
		                     : "<no error message>"));
		inflateEnd(&conn->websocket_inflate_state);
		return zret;
	}
	if ((conn->websocket_deflate_server_no_context_takeover && server)
	    || (conn->websocket_deflate_client_no_context_takeover && !server))
		conn->websocket_deflate_flush = Z_FULL_FLUSH;
	else
		conn->websocket_deflate_flush = Z_SYNC_FLUSH;

	conn->websocket_deflate_initialized = 1;
	return Z_OK;
}

void
websocket_deflate_negotiate(struct mg_connection *conn)
{
	const char *extensions = mg_get_header(conn, "Sec-WebSocket-Extensions");
	int val;
	if (extensions && !strncmp(extensions, "permessage-deflate", 18)) {
		conn->accept_gzip = 1;
		conn->websocket_deflate_client_max_windows_bits = 15;
		conn->websocket_deflate_server_max_windows_bits = 15;
		conn->websocket_deflate_server_no_context_takeover = 0;
		conn->websocket_deflate_client_no_context_takeover = 0;
		extensions += 18;
		while (*extensions != '\0') {
			if (*extensions == ';' || *extensions == ' ')
				++extensions;
			else if (!strncmp(extensions, "server_no_context_takeover", 26)) {
				extensions += 26;
				conn->websocket_deflate_server_no_context_takeover = 1;
			} else if (!strncmp(extensions, "client_no_context_takeover", 26)) {
				extensions += 26;
				conn->websocket_deflate_client_no_context_takeover = 1;
			} else if (!strncmp(extensions, "server-max-window-bits", 22)) {
				extensions += 22;
				if (*extensions == '=') {
					++extensions;
					if (*extensions == '"')
						++extensions;
					val = 0;
					while (*extensions >= '0' && *extensions <= '9') {
						val = val * 10 + (*extensions - '0');
						++extensions;
					}
					if (val < 9 || val > 15) {
						// The permessage-deflate spec specifies that a
						// value of 8 is also allowed, but zlib doesn't accept
						// that.
						mg_cry_internal(conn,
						                "server-max-window-bits must be "
						                "between 9 and 15. Got %i",
						                val);
					} else
						conn->websocket_deflate_server_max_windows_bits = val;
					if (*extensions == '"')
						++extensions;
				}
			} else if (!strncmp(extensions, "client-max-window-bits", 22)) {
				extensions += 22;
				if (*extensions == '=') {
					++extensions;
					if (*extensions == '"')
						++extensions;
					val = 0;
					while (*extensions >= '0' && *extensions <= '9') {
						val = val * 10 + (*extensions - '0');
						++extensions;
					}
					if (val < 9 || val > 15)
						// The permessage-deflate spec specifies that a
						// value of 8 is also allowed, but zlib doesn't
						// accept that.
						mg_cry_internal(conn,
						                "client-max-window-bits must be "
						                "between 9 and 15. Got %i",
						                val);
					else
						conn->websocket_deflate_client_max_windows_bits = val;
					if (*extensions == '"')
						++extensions;
				}
			} else {
				mg_cry_internal(conn,
				                "Unknown parameter %s for permessage-deflate",
				                extensions);
				break;
			}
		}
	} else {
		conn->accept_gzip = 0;
	}
	conn->websocket_deflate_initialized = 0;
}

void
websocket_deflate_response(struct mg_connection *conn)
{
	if (conn->accept_gzip) {
		mg_printf(conn,
		          "Sec-WebSocket-Extensions: permessage-deflate; "
		          "server_max_window_bits=%i; "
		          "client_max_window_bits=%i"
		          "%s%s\r\n",
		          conn->websocket_deflate_server_max_windows_bits,
		          conn->websocket_deflate_client_max_windows_bits,
		          conn->websocket_deflate_client_no_context_takeover
		              ? "; client_no_context_takeover"
		              : "",
		          conn->websocket_deflate_server_no_context_takeover
		              ? "; server_no_context_takeover"
		              : "");
	};
}
#endif
