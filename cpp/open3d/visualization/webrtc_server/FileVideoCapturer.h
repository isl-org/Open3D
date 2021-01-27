/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** filecapturer.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <live555helper/mkvclient.h>

#include "open3d/visualization/webrtc_server/LiveVideoSource.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class FileVideoCapturer : public LiveVideoSource<MKVClient> {
public:
    FileVideoCapturer(const std::string& uri,
                      const std::map<std::string, std::string>& opts);
    virtual ~FileVideoCapturer();

    static FileVideoCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        return new FileVideoCapturer(url, opts);
    }
};
}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
