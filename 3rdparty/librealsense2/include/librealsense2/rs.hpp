// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_HPP
#define LIBREALSENSE_RS2_HPP

#include "rs.h"
#include "hpp/rs_types.hpp"
#include "hpp/rs_context.hpp"
#include "hpp/rs_device.hpp"
#include "hpp/rs_frame.hpp"
#include "hpp/rs_processing.hpp"
#include "hpp/rs_record_playback.hpp"
#include "hpp/rs_sensor.hpp"
#include "hpp/rs_pipeline.hpp"

namespace rs2
{
    inline void log_to_console(rs2_log_severity min_severity)
    {
        rs2_error* e = nullptr;
        rs2_log_to_console(min_severity, &e);
        error::handle(e);
    }

    inline void log_to_file(rs2_log_severity min_severity, const char * file_path = nullptr)
    {
        rs2_error* e = nullptr;
        rs2_log_to_file(min_severity, file_path, &e);
        error::handle(e);
    }

    inline void log(rs2_log_severity severity, const char* message)
    {
        rs2_error* e = nullptr;
        rs2_log(severity, message, &e);
        error::handle(e);
    }
}

inline std::ostream & operator << (std::ostream & o, rs2_stream stream) { return o << rs2_stream_to_string(stream); }
inline std::ostream & operator << (std::ostream & o, rs2_format format) { return o << rs2_format_to_string(format); }
inline std::ostream & operator << (std::ostream & o, rs2_distortion distortion) { return o << rs2_distortion_to_string(distortion); }
inline std::ostream & operator << (std::ostream & o, rs2_option option) { return o << rs2_option_to_string(option); } // This function is being deprecated. For existing options it will return option name, but for future API additions the user should call rs2_get_option_name instead.
inline std::ostream & operator << (std::ostream & o, rs2_log_severity severity) { return o << rs2_log_severity_to_string(severity); }
inline std::ostream & operator << (std::ostream & o, rs2_camera_info camera_info) { return o << rs2_camera_info_to_string(camera_info); }
inline std::ostream & operator << (std::ostream & o, rs2_frame_metadata_value metadata) { return o << rs2_frame_metadata_to_string(metadata); }
inline std::ostream & operator << (std::ostream & o, rs2_timestamp_domain domain) { return o << rs2_timestamp_domain_to_string(domain); }
inline std::ostream & operator << (std::ostream & o, rs2_notification_category notificaton) { return o << rs2_notification_category_to_string(notificaton); }
inline std::ostream & operator << (std::ostream & o, rs2_sr300_visual_preset preset) { return o << rs2_sr300_visual_preset_to_string(preset); }
inline std::ostream & operator << (std::ostream & o, rs2_exception_type exception_type) { return o << rs2_exception_type_to_string(exception_type); }
inline std::ostream & operator << (std::ostream & o, rs2_playback_status status) { return o << rs2_playback_status_to_string(status); }

#endif // LIBREALSENSE_RS2_HPP
