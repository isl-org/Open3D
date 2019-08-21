// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <dlfcn.h>
#include <k4a/k4a.h>
#include <k4arecord/record.h>
#include <iostream>

#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"
#include "Open3D/IO/Sensor/AzureKinect/PluginMacros.h"

static void* GetLibHandle() {
    static void* handle = nullptr;
    static const std::string lib_name = "libk4arecord.so.1.1.1";

    if (!handle) {
        handle = dlopen(lib_name.c_str(), RTLD_LAZY);
        std::cout << "Loaded " << lib_name << std::endl;
        if (!handle) {
            const char* msg = dlerror();
            throw std::runtime_error("Cannot load " + std::string(msg));
        }
    }

    // handle != nullptr guaranteed here
    return handle;
}

#define DEFINE_BRIDGED_FUNC_WITH_COUNT(f_name, return_type, num_args, ...) \
    return_type f_name(CALL_EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__)) { \
        typedef return_type (*f_type)(                                     \
                CALL_EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__));         \
        static f_type f = nullptr;                                         \
                                                                           \
        if (!f) {                                                          \
            f = (f_type)dlsym(GetLibHandle(), #f_name);                    \
            if (!f) {                                                      \
                const char* msg = dlerror();                               \
                throw std::runtime_error(std::string("Cannot load ") +     \
                                         #f_name + ": " + msg);            \
            }                                                              \
        }                                                                  \
        return f(CALL_EXTRACT_PARAMS(num_args, __VA_ARGS__));              \
    }

#define DEFINE_BRIDGED_FUNC(f_name, return_type, ...)   \
    DEFINE_BRIDGED_FUNC_WITH_COUNT(f_name, return_type, \
                                   COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

namespace open3d {
namespace io {
namespace k4a_plugin {

DEFINE_BRIDGED_FUNC(k4a_record_create,
                    k4a_result_t,
                    const char*,
                    path,
                    k4a_device_t,
                    device,
                    const k4a_device_configuration_t,
                    device_config,
                    k4a_record_t*,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4a_record_write_header,
                    k4a_result_t,
                    k4a_record_t,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4a_record_write_capture,
                    k4a_result_t,
                    k4a_record_t,
                    recording_handle,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_record_flush,
                    k4a_result_t,
                    k4a_record_t,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4a_record_close, void, k4a_record_t, recording_handle)

}  // namespace k4a_plugin
}  // namespace io
}  // namespace open3d
