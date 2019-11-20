# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
target_sources(${LRS_TARGET}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/rs.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/rs.h"

        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_types.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_context.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_device.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_frame.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_option.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_processing.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_record_playback.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_sensor.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_internal.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_pipeline.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_config.h"

        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_types.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_context.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_device.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_export.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_frame.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_processing.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_record_playback.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_sensor.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_options.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_internal.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/hpp/rs_pipeline.hpp"

        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/rsutil.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/rs_advanced_mode.h"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/rs_advanced_mode.hpp"
        "${CMAKE_CURRENT_LIST_DIR}/librealsense2/h/rs_advanced_mode_command.h"
)
