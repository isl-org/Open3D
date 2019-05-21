// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_SYNC_H
#define LIBREALSENSE_SYNC_H

#include "types.h"

namespace rsimpl
{
    class frame_archive
    {
        // Define a movable but explicitly noncopyable buffer type to hold our frame data
        struct frame 
        { 
            std::vector<byte> data;
            int timestamp;

            frame() : timestamp() {}
            frame(const frame & r) = delete;
            frame(frame && r) : frame() { *this = std::move(r); }

            frame & operator = (const frame & r) = delete;
            frame & operator = (frame && r) { data = move(r.data); timestamp = r.timestamp; return *this; }            
        };

        // This data will be left constant after creation, and accessed from all threads
        subdevice_mode_selection modes[RS_STREAM_NATIVE_COUNT];
        rs_stream key_stream;
        std::vector<rs_stream> other_streams;

        // This data will be read and written exclusively from the application thread
        frame frontbuffer[RS_STREAM_NATIVE_COUNT];

        // This data will be read and written by all threads, and synchronized with a mutex
        std::vector<frame> frames[RS_STREAM_NATIVE_COUNT];
        std::vector<frame> freelist;
        std::mutex mutex;
        std::condition_variable cv;

        // This data will be read and written exclusively from the frame callback thread
        frame backbuffer[RS_STREAM_NATIVE_COUNT];

        void get_next_frames();
        void dequeue_frame(rs_stream stream);
        void discard_frame(rs_stream stream);
        void cull_frames();
    public:
        frame_archive(const std::vector<subdevice_mode_selection> & selection, rs_stream key_stream);

        // Safe to call from any thread
        bool is_stream_enabled(rs_stream stream) const { return modes[stream].mode.pf.fourcc != 0; }
        const subdevice_mode_selection & get_mode(rs_stream stream) const { return modes[stream]; }

        // Application thread API
        void wait_for_frames();
        bool poll_for_frames();
        const byte * get_frame_data(rs_stream stream) const;
        int get_frame_timestamp(rs_stream stream) const;

        // Frame callback thread API
        byte * alloc_frame(rs_stream stream, int timestamp);
        void commit_frame(rs_stream stream);  
    };
}

#endif