// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_TYPES_HPP
#define LIBREALSENSE_RS2_TYPES_HPP

#include "../rs.h"
#include "../h/rs_context.h"
#include "../h/rs_device.h"
#include "../h/rs_frame.h"
#include "../h/rs_processing.h"
#include "../h/rs_record_playback.h"
#include "../h/rs_sensor.h"
#include "../h/rs_pipeline.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <exception>
#include <iterator>
#include <sstream>
#include <chrono>

struct rs2_frame_callback
{
    virtual void                            on_frame(rs2_frame * f) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_frame_callback() {}
};

struct rs2_frame_processor_callback
{
    virtual void                            on_frame(rs2_frame * f, rs2_source * source) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_frame_processor_callback() {}
};

struct rs2_notifications_callback
{
    virtual void                            on_notification(rs2_notification* n) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_notifications_callback() {}
};

struct rs2_log_callback
{
    virtual void                            on_event(rs2_log_severity severity, const char * message) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_log_callback() {}
};

struct rs2_devices_changed_callback
{
    virtual void                            on_devices_changed(rs2_device_list* removed, rs2_device_list* added) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_devices_changed_callback() {}
};

struct rs2_playback_status_changed_callback
{
    virtual void                            on_playback_status_changed(rs2_playback_status status) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_playback_status_changed_callback() {}
};

struct rs2_update_progress_callback
{
    virtual void                            on_update_progress(const float update_progress) = 0;
    virtual void                            release() = 0;
    virtual                                 ~rs2_update_progress_callback() {}
};

namespace rs2
{
    class error : public std::runtime_error
    {
        std::string function, args;
        rs2_exception_type type;
    public:
        explicit error(rs2_error* err) : runtime_error(rs2_get_error_message(err))
        {
            function = (nullptr != rs2_get_failed_function(err)) ? rs2_get_failed_function(err) : std::string();
            args = (nullptr != rs2_get_failed_args(err)) ? rs2_get_failed_args(err) : std::string();
            type = rs2_get_librealsense_exception_type(err);
            rs2_free_error(err);
        }

        explicit error(const std::string& message) : runtime_error(message.c_str())
        {
            function = "";
            args = "";
            type = RS2_EXCEPTION_TYPE_UNKNOWN;
        }

        const std::string& get_failed_function() const
        {
            return function;
        }

        const std::string& get_failed_args() const
        {
            return args;
        }

        rs2_exception_type get_type() const { return type; }

        static void handle(rs2_error* e);
    };

    #define RS2_ERROR_CLASS(name, base) \
    class name : public base\
    {\
    public:\
        explicit name(rs2_error* e) noexcept : base(e) {}\
    }

    RS2_ERROR_CLASS(recoverable_error, error);
    RS2_ERROR_CLASS(unrecoverable_error, error);
    RS2_ERROR_CLASS(camera_disconnected_error, unrecoverable_error);
    RS2_ERROR_CLASS(backend_error, unrecoverable_error);
    RS2_ERROR_CLASS(device_in_recovery_mode_error, unrecoverable_error);
    RS2_ERROR_CLASS(invalid_value_error, recoverable_error);
    RS2_ERROR_CLASS(wrong_api_call_sequence_error, recoverable_error);
    RS2_ERROR_CLASS(not_implemented_error, recoverable_error);
    #undef RS2_ERROR_CLASS

    inline void error::handle(rs2_error* e)
    {
        if (e)
        {
            auto h = rs2_get_librealsense_exception_type(e);
            switch (h) {
            case RS2_EXCEPTION_TYPE_CAMERA_DISCONNECTED:
                throw camera_disconnected_error(e);
            case RS2_EXCEPTION_TYPE_BACKEND:
                throw backend_error(e);
            case RS2_EXCEPTION_TYPE_INVALID_VALUE:
                throw invalid_value_error(e);
            case RS2_EXCEPTION_TYPE_WRONG_API_CALL_SEQUENCE:
                throw wrong_api_call_sequence_error(e);
            case RS2_EXCEPTION_TYPE_NOT_IMPLEMENTED:
                throw not_implemented_error(e);
            case RS2_EXCEPTION_TYPE_DEVICE_IN_RECOVERY_MODE:
                throw device_in_recovery_mode_error(e);
            default:
                throw error(e);
            }
        }
    }

    class context;
    class device;
    class device_list;
    class syncer;
    class device_base;
    class roi_sensor;
    class frame;

    struct option_range
    {
        float min;
        float max;
        float def;
        float step;
    };

    struct region_of_interest
    {
        int min_x;
        int min_y;
        int max_x;
        int max_y;
    };
}

inline std::ostream & operator << (std::ostream & o, rs2_vector v) { return o << v.x << ", " << v.y << ", " << v.z; }
inline std::ostream & operator << (std::ostream & o, rs2_quaternion q) { return o << q.x << ", " << q.y << ", " << q.z << ", " << q.w; }

#endif // LIBREALSENSE_RS2_TYPES_HPP
