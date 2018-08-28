// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_STREAM_H
#define LIBREALSENSE_STREAM_H

#include "types.h"

#include <memory> // For shared_ptr

namespace rsimpl
{
    struct stream_interface
    {
        rs_extrinsics                           get_extrinsics_to(const stream_interface & r) const;
        virtual rsimpl::pose                    get_pose() const = 0;
        virtual float                           get_depth_scale() const = 0;
        virtual int                             get_mode_count() const { return 0; }
        virtual void                            get_mode(int mode, int * w, int * h, rs_format * f, int * fps) const { throw std::logic_error("no modes"); }

        virtual bool                            is_enabled() const = 0;
        virtual rs_intrinsics                   get_intrinsics() const = 0;
        virtual rs_intrinsics                   get_rectified_intrinsics() const = 0;
        virtual rs_format                       get_format() const = 0;
        virtual int                             get_framerate() const = 0;

        virtual int                             get_frame_number() const = 0;
        virtual const byte *                    get_frame_data() const = 0;    
    };
    
    class frame_archive;

    struct native_stream final : public stream_interface
    {
        const device_config &                   config;
        const rs_stream                         stream;
        std::vector<subdevice_mode_selection>   modes;
        std::shared_ptr<frame_archive>          archive;

                                                native_stream(device_config & config, rs_stream stream);

        pose                                    get_pose() const override { return config.info.stream_poses[stream]; }
        float                                   get_depth_scale() const override { return config.depth_scale; }
        int                                     get_mode_count() const override { return (int)modes.size(); }
        void                                    get_mode(int mode, int * w, int * h, rs_format * f, int * fps) const override;

        bool                                    is_enabled() const override;
        subdevice_mode_selection                get_mode() const;
        rs_intrinsics                           get_intrinsics() const override;
        rs_intrinsics                           get_rectified_intrinsics() const override;
        rs_format                               get_format() const override { return get_mode().get_format(stream); }
        int                                     get_framerate() const override { return get_mode().get_framerate(stream); }

        int                                     get_frame_number() const override;
        const byte *                            get_frame_data() const override;
    };

    class point_stream final : public stream_interface
    {
        const stream_interface &                source;
        mutable std::vector<byte>               image;
        mutable int                             number;
    public:
                                                point_stream(const stream_interface & source) : source(source), number() {}

        pose                                    get_pose() const override { return {{{1,0,0},{0,1,0},{0,0,1}}, source.get_pose().position}; }
        float                                   get_depth_scale() const override { return source.get_depth_scale(); }

        bool                                    is_enabled() const override { return source.is_enabled(); }
        rs_intrinsics                           get_intrinsics() const override { return source.get_intrinsics(); }
        rs_intrinsics                           get_rectified_intrinsics() const override { return source.get_rectified_intrinsics(); }
        rs_format                               get_format() const override { return RS_FORMAT_XYZ32F; }
        int                                     get_framerate() const override { return source.get_framerate(); }

        int                                     get_frame_number() const override { return source.get_frame_number(); }
        const byte *                            get_frame_data() const override;
    };

    class rectified_stream final : public stream_interface
    {
        const stream_interface &                source;
        mutable std::vector<int>                table;
        mutable std::vector<byte>               image;
        mutable int                             number;
    public:
                                                rectified_stream(const stream_interface & source) : source(source), number() {}

        pose                                    get_pose() const override { return {{{1,0,0},{0,1,0},{0,0,1}}, source.get_pose().position}; }
        float                                   get_depth_scale() const override { return source.get_depth_scale(); }

        bool                                    is_enabled() const override { return source.is_enabled(); }
        rs_intrinsics                           get_intrinsics() const override { return source.get_rectified_intrinsics(); }
        rs_intrinsics                           get_rectified_intrinsics() const override { return source.get_rectified_intrinsics(); }
        rs_format                               get_format() const override { return source.get_format(); }
        int                                     get_framerate() const override { return source.get_framerate(); }

        int                                     get_frame_number() const override { return source.get_frame_number(); }
        const byte *                            get_frame_data() const override;
    };

    class aligned_stream final : public stream_interface
    {
        const stream_interface &                from, & to;
        mutable std::vector<byte>               image;
        mutable int                             number;
    public:
                                                aligned_stream(const stream_interface & from, const stream_interface & to) : from(from), to(to), number() {}

        pose                                    get_pose() const override { return to.get_pose(); }
        float                                   get_depth_scale() const override { return to.get_depth_scale(); }

        bool                                    is_enabled() const override { return from.is_enabled() && to.is_enabled(); }
        rs_intrinsics                           get_intrinsics() const override { return to.get_intrinsics(); }
        rs_intrinsics                           get_rectified_intrinsics() const override { return to.get_rectified_intrinsics(); }
        rs_format                               get_format() const override { return from.get_format(); }
        int                                     get_framerate() const override { return from.get_framerate(); }

        int                                     get_frame_number() const override { return from.get_frame_number(); }
        const byte *                            get_frame_data() const override;
    };
}

#endif