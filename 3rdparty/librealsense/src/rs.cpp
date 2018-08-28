// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "context.h"
#include "device.h"

#include <climits>

////////////////////////
// API implementation //
////////////////////////

struct rs_error
{
    std::string message;
    const char * function;
    std::string args;
};

// This facility allows for translation of exceptions to rs_error structs at the API boundary
namespace rsimpl
{
    template<class T> void stream_args(std::ostream & out, const char * names, const T & last) { out << names << ':' << last; }
    template<class T, class... U> void stream_args(std::ostream & out, const char * names, const T & first, const U &... rest)
    {
        while(*names && *names != ',') out << *names++;
        out << ':' << first << ", ";
        while(*names && (*names == ',' || isspace(*names))) ++names;
        stream_args(out, names, rest...);
    }

    static void translate_exception(const char * name, std::string args, rs_error ** error)
    {
        try { throw; }
        catch (const std::exception & e) { if (error) *error = new rs_error {e.what(), name, move(args)}; } // todo - Handle case where THIS code throws
        catch (...) { if (error) *error = new rs_error {"unknown error", name, move(args)}; } // todo - Handle case where THIS code throws
    }
}
#define HANDLE_EXCEPTIONS_AND_RETURN(R, ...) catch(...) { std::ostringstream ss; rsimpl::stream_args(ss, #__VA_ARGS__, __VA_ARGS__); rsimpl::translate_exception(__FUNCTION__, ss.str(), error); return R; }
#define VALIDATE_NOT_NULL(ARG) if(!ARG) throw std::runtime_error("null pointer passed for argument \"" #ARG "\"");
#define VALIDATE_ENUM(ARG) if(!rsimpl::is_valid(ARG)) { std::ostringstream ss; ss << "bad enum value for argument \"" #ARG "\""; throw std::runtime_error(ss.str()); }
#define VALIDATE_RANGE(ARG, MIN, MAX) if(ARG < MIN || ARG > MAX) { std::ostringstream ss; ss << "out of range value for argument \"" #ARG "\""; throw std::runtime_error(ss.str()); }
#define VALIDATE_NATIVE_STREAM(ARG) VALIDATE_ENUM(ARG); if(ARG >= RS_STREAM_NATIVE_COUNT) { std::ostringstream ss; ss << "argument \"" #ARG "\" must be a native stream"; throw std::runtime_error(ss.str()); }

rs_context * rs_create_context(int api_version, rs_error ** error) try
{
    if (api_version != RS_API_VERSION) throw std::runtime_error("api version mismatch");
    return new rs_context();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, api_version)

void rs_delete_context(rs_context * context, rs_error ** error) try
{
    VALIDATE_NOT_NULL(context);
    delete context;
}
HANDLE_EXCEPTIONS_AND_RETURN(, context)

int rs_get_device_count(const rs_context * context, rs_error ** error) try
{
    VALIDATE_NOT_NULL(context);
    return (int)context->devices.size();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, context)

rs_device * rs_get_device(rs_context * context, int index, rs_error ** error) try
{
    VALIDATE_NOT_NULL(context);
    VALIDATE_RANGE(index, 0, (int)context->devices.size()-1);
    return context->devices[index].get();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, context, index)



const char * rs_get_device_name(const rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->get_name();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, device)

const char * rs_get_device_serial(const rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->get_serial();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, device)

const char * rs_get_device_firmware_version(const rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->get_firmware_version();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, device)


void rs_get_device_extrinsics(const rs_device * device, rs_stream from, rs_stream to, rs_extrinsics * extrin, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(from);
    VALIDATE_ENUM(to);
    VALIDATE_NOT_NULL(extrin);
    *extrin = device->get_stream_interface(from).get_extrinsics_to(device->get_stream_interface(to));
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, from, to, extrin)

int rs_device_supports_option(const rs_device * device, rs_option option, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(option);
    return device->supports_option(option);
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, option)

int rs_get_stream_mode_count(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_mode_count();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, stream)

void rs_get_stream_mode(const rs_device * device, rs_stream stream, int index, int * width, int * height, rs_format * format, int * framerate, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    VALIDATE_RANGE(index, 0, device->get_stream_interface(stream).get_mode_count()-1);
    device->get_stream_interface(stream).get_mode(index, width, height, format, framerate);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, stream, index, width, height, format, framerate)


void rs_enable_stream(rs_device * device, rs_stream stream, int width, int height, rs_format format, int framerate, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_NATIVE_STREAM(stream);
    VALIDATE_RANGE(width, 0, INT_MAX);
    VALIDATE_RANGE(height, 0, INT_MAX);
    VALIDATE_ENUM(format);
    VALIDATE_RANGE(framerate, 0, INT_MAX);
    device->enable_stream(stream, width, height, format, framerate);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, stream, width, height, format, framerate)

void rs_enable_stream_preset(rs_device * device, rs_stream stream, rs_preset preset, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_NATIVE_STREAM(stream);
    VALIDATE_ENUM(preset);
    device->enable_stream_preset(stream, preset);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, stream, preset)

void rs_disable_stream(rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_NATIVE_STREAM(stream);
    device->disable_stream(stream);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, stream)

int rs_is_stream_enabled(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).is_enabled();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, stream)

int rs_get_stream_width(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_intrinsics().width;
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, stream)

int rs_get_stream_height(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_intrinsics().height;
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, stream)

rs_format rs_get_stream_format(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_format();
}
HANDLE_EXCEPTIONS_AND_RETURN(RS_FORMAT_ANY, device, stream)

int rs_get_stream_framerate(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_framerate();
}
HANDLE_EXCEPTIONS_AND_RETURN(RS_FORMAT_ANY, device, stream)

void rs_get_stream_intrinsics(const rs_device * device, rs_stream stream, rs_intrinsics * intrin, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    VALIDATE_NOT_NULL(intrin);
    *intrin = device->get_stream_interface(stream).get_intrinsics();
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, stream, intrin)



void rs_start_device(rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    device->start();
}
HANDLE_EXCEPTIONS_AND_RETURN(, device)

void rs_stop_device(rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    device->stop();
}
HANDLE_EXCEPTIONS_AND_RETURN(, device)

int rs_is_device_streaming(const rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->is_capturing();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device)

float rs_get_device_depth_scale(const rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->get_depth_scale();
}
HANDLE_EXCEPTIONS_AND_RETURN(0.0f, device)



void rs_wait_for_frames(rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    device->wait_all_streams();
}
HANDLE_EXCEPTIONS_AND_RETURN(, device)

int rs_poll_for_frames(rs_device * device, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    return device->poll_all_streams();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device)

int rs_get_frame_timestamp(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_frame_number();
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, stream)

const void * rs_get_frame_data(const rs_device * device, rs_stream stream, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(stream);
    return device->get_stream_interface(stream).get_frame_data();
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, device, stream)



const char * rs_get_stream_name(rs_stream stream, rs_error ** error) try
{
    VALIDATE_ENUM(stream);
    return rsimpl::get_string(stream);
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, stream)

const char * rs_get_format_name(rs_format format, rs_error ** error) try
{
   VALIDATE_ENUM(format);
   return rsimpl::get_string(format);
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, format)

const char * rs_get_preset_name(rs_preset preset, rs_error ** error) try
{
    VALIDATE_ENUM(preset);
    return rsimpl::get_string(preset);
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, preset)

const char * rs_get_distortion_name(rs_distortion distortion, rs_error ** error) try
{
   VALIDATE_ENUM(distortion);
   return rsimpl::get_string(distortion);
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, distortion)

const char * rs_get_option_name(rs_option option, rs_error ** error) try
{
    VALIDATE_ENUM(option);
    return rsimpl::get_string(option);
}
HANDLE_EXCEPTIONS_AND_RETURN(nullptr, option)



void rs_get_device_option_range(rs_device * device, rs_option option, double * min, double * max, double * step, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(option);
    double x; // Prevent internal code from having to worry about whether nulls are passed in for min/max/step by giving it somewhere to write to
    device->get_option_range(option, min ? *min : x, max ? *max : x, step ? *step : x);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, option, min, max, step)

void rs_get_device_options(rs_device * device, const rs_option options[], int count, double values[], rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_RANGE(count, 0, INT_MAX);
    VALIDATE_NOT_NULL(options);
    for(int i=0; i<count; ++i) VALIDATE_ENUM(options[i]);
    VALIDATE_NOT_NULL(values);
    device->get_options(options, count, values);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, options, count, values)

void rs_set_device_options(rs_device * device, const rs_option options[], int count, const double values[], rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_RANGE(count, 0, INT_MAX);
    VALIDATE_NOT_NULL(options);
    for(int i=0; i<count; ++i) VALIDATE_ENUM(options[i]);
    VALIDATE_NOT_NULL(values);
    device->set_options(options, count, values);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, options, count, values)

double rs_get_device_option(rs_device * device, rs_option option, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(option);
    double value;
    device->get_options(&option, 1, &value);
    return value;
}
HANDLE_EXCEPTIONS_AND_RETURN(0, device, option)

void rs_set_device_option(rs_device * device, rs_option option, double value, rs_error ** error) try
{
    VALIDATE_NOT_NULL(device);
    VALIDATE_ENUM(option);
    device->set_options(&option, 1, &value);
}
HANDLE_EXCEPTIONS_AND_RETURN(, device, option, value)



void rs_free_error(rs_error * error) { if (error) delete error; }
const char * rs_get_failed_function(const rs_error * error) { return error ? error->function : nullptr; }
const char * rs_get_failed_args(const rs_error * error) { return error ? error->args.c_str() : nullptr; }
const char * rs_get_error_message(const rs_error * error) { return error ? error->message.c_str() : nullptr; }



const char * rs_stream_to_string(rs_stream stream) { return rsimpl::get_string(stream); }
const char * rs_format_to_string(rs_format format) { return rsimpl::get_string(format); }
const char * rs_preset_to_string(rs_preset preset) { return rsimpl::get_string(preset); }
const char * rs_distortion_to_string(rs_distortion distortion) { return rsimpl::get_string(distortion); }
const char * rs_option_to_string(rs_option option) { return rsimpl::get_string(option); }



void rs_log_to_console(rs_log_severity min_severity, rs_error ** error) try
{
    rsimpl::log_to_console(min_severity);
}
HANDLE_EXCEPTIONS_AND_RETURN(, min_severity)

void rs_log_to_file(rs_log_severity min_severity, const char * file_path, rs_error ** error) try
{
    rsimpl::log_to_file(min_severity, file_path);
}
HANDLE_EXCEPTIONS_AND_RETURN(, min_severity, file_path)