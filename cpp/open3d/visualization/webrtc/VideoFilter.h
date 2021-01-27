/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Videofilter.h
**
** -------------------------------------------------------------------------*/

#pragma once

#include <pc/video_track_source.h>

template <class T>
class VideoFilter : public webrtc::VideoTrackSource {
public:
    static rtc::scoped_refptr<VideoFilter> Create(
            rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> videoSource,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<T> source = absl::WrapUnique(new T(videoSource, opts));
        if (!source) {
            return nullptr;
        }
        return new rtc::RefCountedObject<VideoFilter>(std::move(source));
    }

protected:
    explicit VideoFilter(std::unique_ptr<T> source)
        : webrtc::VideoTrackSource(/*remote=*/false),
          m_source(std::move(source)) {}

    SourceState state() const override { return kLive; }
    bool GetStats(Stats* stats) override {
        bool result = false;
        T* source = m_source.get();
        if (source) {
            stats->input_height = source->height();
            stats->input_width = source->width();
            result = true;
        }
        return result;
    }

private:
    rtc::VideoSourceInterface<webrtc::VideoFrame>* source() override {
        return m_source.get();
    }
    std::unique_ptr<T> m_source;
};
