// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_PIPELINE_HPP
#define LIBREALSENSE_RS2_PIPELINE_HPP

#include "rs_types.hpp"
#include "rs_frame.hpp"
#include "rs_context.hpp"

namespace rs2
{
    /**
    * The pipeline profile includes a device and a selection of active streams, with specific profiles.
    * The profile is a selection of the above under filters and conditions defined by the pipeline.
    * Streams may belong to more than one sensor of the device.
    */
    class pipeline_profile
    {
    public:

        pipeline_profile() : _pipeline_profile(nullptr) {}

        /**
        * Return the selected streams profiles, which are enabled in this profile.
        *
        * \return   Vector of stream profiles
        */
        std::vector<stream_profile> get_streams() const
        {
            std::vector<stream_profile> results;

            rs2_error* e = nullptr;
            std::shared_ptr<rs2_stream_profile_list> list(
                rs2_pipeline_profile_get_streams(_pipeline_profile.get(), &e),
                rs2_delete_stream_profiles_list);
            error::handle(e);

            auto size = rs2_get_stream_profiles_count(list.get(), &e);
            error::handle(e);

            for (auto i = 0; i < size; i++)
            {
                stream_profile profile(rs2_get_stream_profile(list.get(), i, &e));
                error::handle(e);
                results.push_back(profile);
            }

            return results;
        }

        /**
        * Return the stream profile that is enabled for the specified stream in this profile.
        *
        * \param[in] stream_type     Stream type of the desired profile
        * \param[in] stream_index    Stream index of the desired profile. -1 for any matching.
        * \return   The first matching stream profile
        */

        stream_profile get_stream(rs2_stream stream_type, int stream_index = -1) const
        {
            for (auto&& s : get_streams())
            {
                if (s.stream_type() == stream_type &&  (stream_index == -1 || s.stream_index() == stream_index))
                {
                    return s;
                }
            }
            throw std::runtime_error("Profile does not contain the requested stream");
        }

        /**
        * Retrieve the device used by the pipeline.
        * The device class provides the application access to control camera additional settings -
        * get device information, sensor options information, options value query and set, sensor specific extensions.
        * Since the pipeline controls the device streams configuration, activation state and frames reading, calling
        * the device API functions, which execute those operations, results in unexpected behavior.
        * The pipeline streaming device is selected during pipeline \c start(). Devices of profiles, which are not returned by
        * pipeline \c start() or \c get_active_profile(), are not guaranteed to be used by the pipeline.
        *
        * \return rs2::device The pipeline selected device
        */
        device get_device() const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_device> dev(
                rs2_pipeline_profile_get_device(_pipeline_profile.get(), &e),
                rs2_delete_device);

            error::handle(e);

            return device(dev);
        }

        /**
        * Conversion to boolean value to test for the object's validity
        *
        * \return true iff the profile is valid
        */
        operator bool() const
        {
            return _pipeline_profile != nullptr;
        }

        explicit operator std::shared_ptr<rs2_pipeline_profile>() { return _pipeline_profile; }
        pipeline_profile(std::shared_ptr<rs2_pipeline_profile> profile) :
            _pipeline_profile(profile){}
    private:

        std::shared_ptr<rs2_pipeline_profile> _pipeline_profile;
        friend class config;
        friend class pipeline;
    };

    class pipeline;

    /**
    * The config allows pipeline users to request filters for the pipeline streams and device selection and configuration.
    * This is an optional step in pipeline creation, as the pipeline resolves its streaming device internally.
    * Config provides its users a way to set the filters and test if there is no conflict with the pipeline requirements
    * from the device. It also allows the user to find a matching device for the config filters and the pipeline, in order to
    * select a device explicitly, and modify its controls before streaming starts.
    */
    class config
    {
    public:
        config()
        {
            rs2_error* e = nullptr;
            _config = std::shared_ptr<rs2_config>(
                rs2_create_config(&e),
                rs2_delete_config);
            error::handle(e);
        }

        /**
        * Enable a device stream explicitly, with selected stream parameters.
        * The method allows the application to request a stream with specific configuration. If no stream is explicitly enabled,
        * the pipeline configures the device and its streams according to the attached computer vision modules and processing
        * blocks requirements, or default configuration for the first available device.
        * The application can configure any of the input stream parameters according to its requirement, or set to 0 for don't
        * care value.
        * The config accumulates the application calls for enable configuration methods, until the configuration is applied.
        * Multiple enable stream calls for the same stream override each other, and the last call is maintained.
        * Upon calling \c resolve(), the config checks for conflicts between the application configuration requests and the
        * attached computer vision modules and processing blocks requirements, and fails if conflicts are found.
        * Before \c resolve() is called, no conflict check is done.
        *
        * \param[in] stream_type    Stream type to be enabled
        * \param[in] stream_index   Stream index, used for multiple streams of the same type. -1 indicates any.
        * \param[in] width          Stream image width - for images streams. 0 indicates any.
        * \param[in] height         Stream image height - for images streams. 0 indicates any.
        * \param[in] format         Stream data format - pixel format for images streams, of data type for other streams. RS2_FORMAT_ANY indicates any.
        * \param[in] framerate      Stream frames per second. 0 indicates any.
        */
        void enable_stream(rs2_stream stream_type, int stream_index, int width, int height, rs2_format format = RS2_FORMAT_ANY, int framerate = 0)
        {
            rs2_error* e = nullptr;
            rs2_config_enable_stream(_config.get(), stream_type, stream_index, width, height, format, framerate, &e);
            error::handle(e);
        }

        /**
        * Stream type and possibly also stream index. Other parameters are resolved internally.
        *
        * \param[in] stream_type    Stream type to be enabled
        * \param[in] stream_index   Stream index, used for multiple streams of the same type. -1 indicates any.
        */
        void enable_stream(rs2_stream stream_type, int stream_index = -1)
        {
            enable_stream(stream_type, stream_index, 0, 0, RS2_FORMAT_ANY, 0);
        }

        /**
        * Stream type and resolution, and possibly format and frame rate. Other parameters are resolved internally.
        *
        * \param[in] stream_type    Stream type to be enabled
        * \param[in] width          Stream image width - for images streams. 0 indicates any.
        * \param[in] height         Stream image height - for images streams. 0 indicates any.
        * \param[in] format         Stream data format - pixel format for images streams, of data type for other streams. RS2_FORMAT_ANY indicates any.
        * \param[in] framerate      Stream frames per second. 0 indicates any.
        */
        void enable_stream(rs2_stream stream_type, int width, int height, rs2_format format = RS2_FORMAT_ANY, int framerate = 0)
        {
            enable_stream(stream_type, -1, width, height, format, framerate);
        }

        /**
        * Stream type and format, and possibly frame rate. Other parameters are resolved internally.
        *
        * \param[in] stream_type    Stream type to be enabled
        * \param[in] format         Stream data format - pixel format for images streams, of data type for other streams. RS2_FORMAT_ANY indicates any.
        * \param[in] framerate      Stream frames per second. 0 indicates any.
        */
        void enable_stream(rs2_stream stream_type, rs2_format format, int framerate = 0)
        {
            enable_stream(stream_type, -1, 0, 0, format, framerate);
        }

        /**
        * Stream type, index, and format, and possibly framerate. Other parameters are resolved internally.
        *
        * \param[in] stream_type    Stream type to be enabled
        * \param[in] stream_index   Stream index, used for multiple streams of the same type. -1 indicates any.
        * \param[in] format         Stream data format - pixel format for images streams, of data type for other streams. RS2_FORMAT_ANY indicates any.
        * \param[in] framerate      Stream frames per second. 0 indicates any.
        */
        void enable_stream(rs2_stream stream_type, int stream_index, rs2_format format, int framerate = 0)
        {
            enable_stream(stream_type, stream_index, 0, 0, format, framerate);
        }

        /**
        * Enable all device streams explicitly.
        * The conditions and behavior of this method are similar to those of \c enable_stream().
        * This filter enables all raw streams of the selected device. The device is either selected explicitly by the
        * application, or by the pipeline requirements or default. The list of streams is device dependent.
        */
        void enable_all_streams()
        {
            rs2_error* e = nullptr;
            rs2_config_enable_all_stream(_config.get(), &e);
            error::handle(e);
        }

        /**
        * Select a specific device explicitly by its serial number, to be used by the pipeline.
        * The conditions and behavior of this method are similar to those of \c enable_stream().
        * This method is required if the application needs to set device or sensor settings prior to pipeline streaming,
        * to enforce the pipeline to use the configured device.
        *
        * \param[in] serial device serial number, as returned by RS2_CAMERA_INFO_SERIAL_NUMBER
        */
        void enable_device(const std::string& serial)
        {
            rs2_error* e = nullptr;
            rs2_config_enable_device(_config.get(), serial.c_str(), &e);
            error::handle(e);
        }

        /**
        * Select a recorded device from a file, to be used by the pipeline through playback.
        * The device available streams are as recorded to the file, and \c resolve() considers only this device and
        * configuration as available.
        * This request cannot be used if \c enable_record_to_file() is called for the current config, and vice versa.
        *
        * \param[in] file_name  The playback file of the device
        */
        void enable_device_from_file(const std::string& file_name, bool repeat_playback = true)
        {
            rs2_error* e = nullptr;
            rs2_config_enable_device_from_file_repeat_option(_config.get(), file_name.c_str(), repeat_playback, &e);
            error::handle(e);
        }

        /**
        * Requires that the resolved device would be recorded to file.
        * This request cannot be used if \c enable_device_from_file() is called for the current config, and vice versa.
        * as available.
        *
        * \param[in] file_name  The desired file for the output record
        */
        void enable_record_to_file(const std::string& file_name)
        {
            rs2_error* e = nullptr;
            rs2_config_enable_record_to_file(_config.get(), file_name.c_str(), &e);
            error::handle(e);
        }

        /**
        * Disable a device stream explicitly, to remove any requests on this stream profile.
        * The stream can still be enabled due to pipeline computer vision module request. This call removes any filter on the
        * stream configuration.
        *
        * \param[in] stream    Stream type, for which the filters are cleared
        */
        void disable_stream(rs2_stream stream, int index = -1)
        {
            rs2_error* e = nullptr;
            rs2_config_disable_indexed_stream(_config.get(), stream, index, &e);
            error::handle(e);
        }

        /**
        * Disable all device stream explicitly, to remove any requests on the streams profiles.
        * The streams can still be enabled due to pipeline computer vision module request. This call removes any filter on the
        * streams configuration.
        */
        void disable_all_streams()
        {
            rs2_error* e = nullptr;
            rs2_config_disable_all_streams(_config.get(), &e);
            error::handle(e);
        }

        /**
        * Resolve the configuration filters, to find a matching device and streams profiles.
        * The method resolves the user configuration filters for the device and streams, and combines them with the requirements
        * of the computer vision modules and processing blocks attached to the pipeline. If there are no conflicts of requests,
        * it looks for an available device, which can satisfy all requests, and selects the first matching streams configuration.
        * In the absence of any request, the rs2::config selects the first available device and the first color and depth
        * streams configuration.
        * The pipeline profile selection during \c start() follows the same method. Thus, the selected profile is the same, if no
        * change occurs to the available devices.
        * Resolving the pipeline configuration provides the application access to the pipeline selected device for advanced
        * control.
        * The returned configuration is not applied to the device, so the application doesn't own the device sensors. However,
        * the application can call \c enable_device(), to enforce the device returned by this method is selected by pipeline \c
        * start(), and configure the device and sensors options or extensions before streaming starts.
        *
        * \param[in] p  The pipeline for which the selected filters are applied
        * \return       A matching device and streams profile, which satisfies the filters and pipeline requests.
        */
        pipeline_profile resolve(std::shared_ptr<rs2_pipeline> p) const
        {
            rs2_error* e = nullptr;
            auto profile = std::shared_ptr<rs2_pipeline_profile>(
                rs2_config_resolve(_config.get(), p.get(), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(profile);
        }

        /**
        * Check if the config can resolve the configuration filters, to find a matching device and streams profiles.
        * The resolution conditions are as described in \c resolve().
        *
        * \param[in] p  The pipeline for which the selected filters are applied
        * \return       True if a valid profile selection exists, false if no selection can be found under the config filters and the available devices.
        */
        bool can_resolve(std::shared_ptr<rs2_pipeline> p) const
        {
            rs2_error* e = nullptr;
            int res = rs2_config_can_resolve(_config.get(), p.get(), &e);
            error::handle(e);
            return res != 0;
        }

        std::shared_ptr<rs2_config> get() const
        {
            return _config;
        }
        explicit operator std::shared_ptr<rs2_config>() const
        {
            return _config;
        }

        config(std::shared_ptr<rs2_config> cfg) : _config(cfg) {}
    private:
        std::shared_ptr<rs2_config> _config;
    };

    /**
    * The pipeline simplifies the user interaction with the device and computer vision processing modules.
    * The class abstracts the camera configuration and streaming, and the vision modules triggering and threading.
    * It lets the application focus on the computer vision output of the modules, or the device output data.
    * The pipeline can manage computer vision modules, which are implemented as a processing blocks.
    * The pipeline is the consumer of the processing block interface, while the application consumes the
    * computer vision interface.
    */
    class pipeline
    {
    public:

        /**
        * Create a pipeline for processing data from a single device.
        * The caller can provide a context created by the application, usually for playback or testing purposes.
        *
        * \param[in] ctx   The context allocated by the application. Using the platform context by default.
        */
        pipeline(context ctx = context())
        {
            rs2_error* e = nullptr;
            _pipeline = std::shared_ptr<rs2_pipeline>(
                rs2_create_pipeline(ctx._context.get(), &e),
                rs2_delete_pipeline);
            error::handle(e);
        }

        /**
        * Start the pipeline streaming with its default configuration.
        * The pipeline streaming loop captures samples from the device, and delivers them to the attached computer vision
        * modules and processing blocks, according to each module requirements and threading model.
        * During the loop execution, the application can access the camera streams by calling \c wait_for_frames() or
        * \c poll_for_frames().
        * The streaming loop runs until the pipeline is stopped.
        * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
        *
        * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
        */
        pipeline_profile start()
        {
            rs2_error* e = nullptr;
            auto p = std::shared_ptr<rs2_pipeline_profile>(
                rs2_pipeline_start(_pipeline.get(), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(p);
        }

        /**
        * Start the pipeline streaming according to the configuration.
        * The pipeline streaming loop captures samples from the device, and delivers them to the attached computer vision modules
        * and processing blocks, according to each module requirements and threading model.
        * During the loop execution, the application can access the camera streams by calling \c wait_for_frames() or
        * \c poll_for_frames().
        * The streaming loop runs until the pipeline is stopped.
        * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.

        * The pipeline selects and activates the device upon start, according to configuration or a default configuration.
        * When the rs2::config is provided to the method, the pipeline tries to activate the config \c resolve() result.
        * If the application requests are conflicting with pipeline computer vision modules or no matching device is available on
        * the platform, the method fails.
        * Available configurations and devices may change between config \c resolve() call and pipeline start, in case devices
        * are connected or disconnected, or another application acquires ownership of a device.
        *
        * \param[in] config   A rs2::config with requested filters on the pipeline configuration. By default no filters are applied.
        * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
        */
        pipeline_profile start(const config& config)
        {
            rs2_error* e = nullptr;
            auto p = std::shared_ptr<rs2_pipeline_profile>(
                rs2_pipeline_start_with_config(_pipeline.get(), config.get().get(), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(p);
        }

        /**
        * Start the pipeline streaming with its default configuration.
        * The pipeline captures samples from the device, and delivers them to the provided frame callback.
        * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
        * When starting the pipeline with a callback both \c wait_for_frames() and \c poll_for_frames() will throw exception.
        *
        * \param[in] callback   Stream callback, can be any callable object accepting rs2::frame
        * \return               The actual pipeline device and streams profile, which was successfully configured to the streaming device.
        */
        template<class S>
        pipeline_profile start(S callback)
        {
            rs2_error* e = nullptr;
            auto p = std::shared_ptr<rs2_pipeline_profile>(
                rs2_pipeline_start_with_callback_cpp(_pipeline.get(), new frame_callback<S>(callback), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(p);
        }

        /**
        * Start the pipeline streaming according to the configuraion.
        * The pipeline captures samples from the device, and delivers them to the provided frame callback.
        * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
        * When starting the pipeline with a callback both \c wait_for_frames() and \c poll_for_frames() will throw exception.
        * The pipeline selects and activates the device upon start, according to configuration or a default configuration.
        * When the rs2::config is provided to the method, the pipeline tries to activate the config \c resolve() result.
        * If the application requests are conflicting with pipeline computer vision modules or no matching device is available on
        * the platform, the method fails.
        * Available configurations and devices may change between config \c resolve() call and pipeline start, in case devices
        * are connected or disconnected, or another application acquires ownership of a device.
        *
        * \param[in] config     A rs2::config with requested filters on the pipeline configuration. By default no filters are applied.
        * \param[in] callback   Stream callback, can be any callable object accepting rs2::frame
        * \return               The actual pipeline device and streams profile, which was successfully configured to the streaming device.
        */
        template<class S>
        pipeline_profile start(const config& config, S callback)
        {
            rs2_error* e = nullptr;
            auto p = std::shared_ptr<rs2_pipeline_profile>(
                rs2_pipeline_start_with_config_and_callback_cpp(_pipeline.get(), config.get().get(), new frame_callback<S>(callback), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(p);
        }

        /**
        * Stop the pipeline streaming.
        * The pipeline stops delivering samples to the attached computer vision modules and processing blocks, stops the device
        * streaming and releases the device resources used by the pipeline. It is the application's responsibility to release any
        * frame reference it owns.
        * The method takes effect only after \c start() was called, otherwise an exception is raised.
        */
        void stop()
        {
            rs2_error* e = nullptr;
            rs2_pipeline_stop(_pipeline.get(), &e);
            error::handle(e);
        }

        /**
        * Wait until a new set of frames becomes available.
        * The frames set includes time-synchronized frames of each enabled stream in the pipeline.
        * In case of different frame rates of the streams, the frames set include a matching frame of the slow stream,
        * which may have been included in previous frames set.
        * The method blocks the calling thread, and fetches the latest unread frames set.
        * Device frames, which were produced while the function wasn't called, are dropped. To avoid frame drops, this method
        * should be called as fast as the device frame rate.
        * The application can maintain the frames handles to defer processing. However, if the application maintains too long
        * history, the device may lack memory resources to produce new frames, and the following call to this method shall fail
        * to retrieve new frames, until resources become available.
        *
        * \param[in] timeout_ms   Max time in milliseconds to wait until an exception will be thrown
        * \return                 Set of time synchronized frames, one from each active stream
        */
        frameset wait_for_frames(unsigned int timeout_ms = RS2_DEFAULT_TIMEOUT) const
        {
            rs2_error* e = nullptr;
            frame f(rs2_pipeline_wait_for_frames(_pipeline.get(), timeout_ms, &e));
            error::handle(e);

            return frameset(f);
        }

        /**
        * Check if a new set of frames is available and retrieve the latest undelivered set.
        * The frames set includes time-synchronized frames of each enabled stream in the pipeline.
        * The method returns without blocking the calling thread, with status of new frames available or not.
        * If available, it fetches the latest frames set.
        * Device frames, which were produced while the function wasn't called, are dropped.
        * To avoid frame drops, this method should be called as fast as the device frame rate.
        * The application can maintain the frames handles to defer processing. However, if the application maintains too long
        * history, the device may lack memory resources to produce new frames, and the following calls to this method shall
        * return no new frames, until resources become available.
        *
        * \param[out] f     Frames set handle
        * \return           True if new set of time synchronized frames was stored to f, false if no new frames set is available
        */
        bool poll_for_frames(frameset* f) const
        {
            if (!f)
            {
                throw std::invalid_argument("null frameset");
            }
            rs2_error* e = nullptr;
            rs2_frame* frame_ref = nullptr;
            auto res = rs2_pipeline_poll_for_frames(_pipeline.get(), &frame_ref, &e);
            error::handle(e);

            if (res) *f = frameset(frame(frame_ref));
            return res > 0;
        }

        bool try_wait_for_frames(frameset* f, unsigned int timeout_ms = RS2_DEFAULT_TIMEOUT) const
        {
            if (!f)
            {
                throw std::invalid_argument("null frameset");
            }
            rs2_error* e = nullptr;
            rs2_frame* frame_ref = nullptr;
            auto res = rs2_pipeline_try_wait_for_frames(_pipeline.get(), &frame_ref, timeout_ms, &e);
            error::handle(e);
            if (res) *f = frameset(frame(frame_ref));
            return res > 0;
        }

        /**
        * Return the active device and streams profiles, used by the pipeline.
        * The pipeline streams profiles are selected during \c start(). The method returns a valid result only when the pipeline is active -
        * between calls to \c start() and \c stop().
        * After \c stop() is called, the pipeline doesn't own the device, thus, the pipeline selected device may change in
        * subsequent activations.
        *
        * \return  The actual pipeline device and streams profile, which was successfully configured to the streaming device on start.
        */
        pipeline_profile get_active_profile() const
        {
            rs2_error* e = nullptr;
            auto p = std::shared_ptr<rs2_pipeline_profile>(
                rs2_pipeline_get_active_profile(_pipeline.get(), &e),
                rs2_delete_pipeline_profile);

            error::handle(e);
            return pipeline_profile(p);
        }

        operator std::shared_ptr<rs2_pipeline>() const
        {
            return _pipeline;
        }
        explicit pipeline(std::shared_ptr<rs2_pipeline> ptr) : _pipeline(ptr) {}

    private:
        std::shared_ptr<rs2_pipeline> _pipeline;
        friend class config;
    };
}
#endif // LIBREALSENSE_RS2_PROCESSING_HPP
