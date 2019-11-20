// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_PROCESSING_HPP
#define LIBREALSENSE_RS2_PROCESSING_HPP

#include "rs_types.hpp"
#include "rs_frame.hpp"
#include "rs_options.hpp"

namespace rs2
{
    /**
    * The source used to generate frames, which is usually done by the low level driver for each sensor. frame_source is one of the parameters
    * of processing_block's callback function, which can be used to re-generate the frame and via frame_ready invoke another callback function
    * to notify application frame is ready. Please refer to "video_processing_thread" code snippet in rs-measure.cpp for a detailed usage example.
    */
    class frame_source
    {
    public:
        /**
        * Allocate a new video frame with given params
        *
        * \param[in] profile     Stream profile going to allocate.
        * \param[in] original    Original frame, if new_bpp, new_width, new_height or new_stride is zero, newly created frame will base on original frame's metadata to allocate new frame. If frame_type is RS2_EXTENSION_DEPTH_FRAME, the original of the returned frame will be set to it.
        * \param[in] new_bpp     Frame bit per pixel to create.
        * \param[in] new_width   Frame width to create.
        * \param[in] new_height  Frame height to create.
        * \param[in] new_stride  Frame stride to crate.
        * \param[in] frame_type  Which frame type are going to create.
        * \return   The allocated frame
        */
        frame allocate_video_frame(const stream_profile& profile,
            const frame& original,
            int new_bpp = 0,
            int new_width = 0,
            int new_height = 0,
            int new_stride = 0,
            rs2_extension frame_type = RS2_EXTENSION_VIDEO_FRAME) const
        {
            rs2_error* e = nullptr;
            auto result = rs2_allocate_synthetic_video_frame(_source, profile.get(),
                original.get(), new_bpp, new_width, new_height, new_stride, frame_type, &e);
            error::handle(e);
            return result;
        }

        frame allocate_points(const stream_profile& profile,
            const frame& original) const
        {
            rs2_error* e = nullptr;
            auto result = rs2_allocate_points(_source, profile.get(), original.get(), &e);
            error::handle(e);
            return result;
        }

        /**
        * Allocate composite frame with given params
        *
        * \param[in] frames      Frame vecotor used to create composite frame, the size of composite frame will be the same as vector size.
        * \return    The allocated composite frame
        */
        frame allocate_composite_frame(std::vector<frame> frames) const
        {
            rs2_error* e = nullptr;

            std::vector<rs2_frame*> refs(frames.size(), (rs2_frame*)nullptr);
            for (size_t i = 0; i < frames.size(); i++)
                std::swap(refs[i], frames[i].frame_ref);

            auto result = rs2_allocate_composite_frame(_source, refs.data(), (int)refs.size(), &e);
            error::handle(e);
            return result;
        }
        /**
        * Invoke the callback funtion informing the frame is ready.
        *
        * \param[in] frames      frame to send to callback function.
        */
        void frame_ready(frame result) const
        {
            rs2_error* e = nullptr;
            rs2_synthetic_frame_ready(_source, result.get(), &e);
            error::handle(e);
            result.frame_ref = nullptr;
        }

        rs2_source* _source;
    private:
        template<class T>
        friend class frame_processor_callback;

        frame_source(rs2_source* source) : _source(source) {}
        frame_source(const frame_source&) = delete;

    };

    template<class T>
    class frame_processor_callback : public rs2_frame_processor_callback
    {
        T on_frame_function;
    public:
        explicit frame_processor_callback(T on_frame) : on_frame_function(on_frame) {}

        void on_frame(rs2_frame* f, rs2_source * source) override
        {
            frame_source src(source);
            frame frm(f);
            on_frame_function(std::move(frm), src);
        }

        void release() override { delete this; }
    };

    class frame_queue
    {
    public:
        /**
        * create frame queue. frame queues are the simplest x-platform synchronization primitive provided by librealsense
        * to help developers who are not using async APIs
        * param[in] capacity size of the frame queue
        * param[in] keep_frames  if set to true, the queue automatically calls keep() on every frame enqueued into it.
        */
        explicit frame_queue(unsigned int capacity, bool keep_frames = false) : _capacity(capacity), _keep(keep_frames)
        {
            rs2_error* e = nullptr;
            _queue = std::shared_ptr<rs2_frame_queue>(
                rs2_create_frame_queue(capacity, &e),
                rs2_delete_frame_queue);
            error::handle(e);
        }

        frame_queue() : frame_queue(1) {}

        /**
        * enqueue new frame into the queue
        * \param[in] f - frame handle to enqueue (this operation passed ownership to the queue)
        */
        void enqueue(frame f) const
        {
            if (_keep) f.keep();
            rs2_enqueue_frame(f.frame_ref, _queue.get()); // noexcept
            f.frame_ref = nullptr; // frame has been essentially moved from
        }

        /**
        * wait until new frame becomes available in the queue and dequeue it
        * \return frame handle to be released using rs2_release_frame
        */
        frame wait_for_frame(unsigned int timeout_ms = 5000) const
        {
            rs2_error* e = nullptr;
            auto frame_ref = rs2_wait_for_frame(_queue.get(), timeout_ms, &e);
            error::handle(e);
            return{ frame_ref };
        }

        /**
        * poll if a new frame is available and dequeue if it is
        * \param[out] f - frame handle
        * \return true if new frame was stored to f
        */
        template<typename T>
        typename std::enable_if<std::is_base_of<rs2::frame, T>::value, bool>::type poll_for_frame(T* output) const
        {
            rs2_error* e = nullptr;
            rs2_frame* frame_ref = nullptr;
            auto res = rs2_poll_for_frame(_queue.get(), &frame_ref, &e);
            error::handle(e);
            frame f{ frame_ref };
            if (res) *output = f;
            return res > 0;
        }

        template<typename T>
        typename std::enable_if<std::is_base_of<rs2::frame, T>::value, bool>::type try_wait_for_frame(T* output, unsigned int timeout_ms = 5000) const
        {
            rs2_error* e = nullptr;
            rs2_frame* frame_ref = nullptr;
            auto res = rs2_try_wait_for_frame(_queue.get(), timeout_ms, &frame_ref, &e);
            error::handle(e);
            frame f{ frame_ref };
            if (res) *output = f;
            return res > 0;
        }
        /**
        * Does the same thing as enqueue function.
        */
        void operator()(frame f) const
        {
            enqueue(std::move(f));
        }
        /**
        * Return the capacity of the queue
        * \return capacity size
        */
        size_t capacity() const { return _capacity; }

        /**
        * Return whether or not the queue calls keep on enqueued frames
        * \return keeping frames
        */
        bool keep_frames() const { return _keep; }

    private:
        std::shared_ptr<rs2_frame_queue> _queue;
        size_t _capacity;
        bool _keep;
    };

    /**
    * Define the processing block flow, inherit this class to generate your own processing_block. Please refer to the viewer class in examples.hpp for a detailed usage example.
    */
    class processing_block : public options
    {
    public:
        using options::supports;

        /**
        * Start the processing block with callback function on_frame to inform the application the frame is processed.
        *
        * \param[in] on_frame      callback function for notifying the frame to be processed is ready.
        */
        template<class S>
        void start(S on_frame)
        {
            rs2_error* e = nullptr;
            rs2_start_processing(get(), new frame_callback<S>(on_frame), &e);
            error::handle(e);
        }
        /**
        * Does the same thing as "start" function
        *
        * \param[in] on_frame      address of callback function for noticing the frame to be processed is ready.
        * \return address of callback function.
        */
        template<class S>
        S& operator>>(S& on_frame)
        {
            start(on_frame);
            return on_frame;
        }
        /**
        * Ask processing block to process the frame
        *
        * \param[in] on_frame      frame to be processed.
        */
        void invoke(frame f) const
        {
            rs2_frame* ptr = nullptr;
            std::swap(f.frame_ref, ptr);

            rs2_error* e = nullptr;
            rs2_process_frame(get(), ptr, &e);
            error::handle(e);
        }
        /**
        * constructor with already created low level processing block assigned.
        *
        * \param[in] block - low level rs2_processing_block created before.
        */
        processing_block(std::shared_ptr<rs2_processing_block> block)
            : options((rs2_options*)block.get()), _block(block)
        {
        }

        /**
        * constructor with callback function on_frame in rs2_frame_processor_callback structure assigned.
        *
        * \param[in] processing_function - function pointer of on_frame function in rs2_frame_processor_callback structure, which will be called back by invoke function .
        */
        template<class S>
        processing_block(S processing_function)
        {
            rs2_error* e = nullptr;
            _block = std::shared_ptr<rs2_processing_block>(
                rs2_create_processing_block(new frame_processor_callback<S>(processing_function), &e),
                rs2_delete_processing_block);
            options::operator=(_block);
            error::handle(e);
        }

        operator rs2_options*() const { return (rs2_options*)get(); }
        rs2_processing_block* get() const { return _block.get(); }

        /**
        * Check if a specific camera info field is supported.
        * \param[in] info    the parameter to check for support
        * \return            true if the parameter both exists and well-defined for the specific processing_block
        */
        bool supports(rs2_camera_info info) const
        {
            rs2_error* e = nullptr;
            auto is_supported = rs2_supports_processing_block_info(_block.get(), info, &e);
            error::handle(e);
            return is_supported > 0;
        }

        /**
        * Retrieve camera specific information, like versions of various internal components.
        * \param[in] info     camera info type to retrieve
        * \return             the requested camera info string, in a format specific to the processing_block model
        */
        const char* get_info(rs2_camera_info info) const
        {
            rs2_error* e = nullptr;
            auto result = rs2_get_processing_block_info(_block.get(), info, &e);
            error::handle(e);
            return result;
        }
    protected:
        void register_simple_option(rs2_option option_id, option_range range) {
            rs2_error * e = nullptr;
            rs2_processing_block_register_simple_option(_block.get(), option_id,
                    range.min, range.max, range.step, range.def, &e);
            error::handle(e);
        }
        std::shared_ptr<rs2_processing_block> _block;
    };

    /**
    * Define the filter workflow, inherit this class to generate your own filter. Refer to the viewer class in examples.hpp for a more detailed example.
    */
    class filter : public processing_block, public filter_interface
    {
    public:
        /**
        * Ask processing block to process the frame and poll the processed frame from internal queue
        *
        * \param[in] on_frame      frame to be processed.
        * return processed frame
        */
        rs2::frame process(rs2::frame frame) const override
        {
            invoke(frame);
            rs2::frame f;
            if (!_queue.poll_for_frame(&f))
                throw std::runtime_error("Error occured during execution of the processing block! See the log for more info");
            return f;
        }

        /**
        * constructor with already created low level processing block assigned.
        *
        * \param[in] block - low level rs2_processing_block created before.
        */
        filter(std::shared_ptr<rs2_processing_block> block, int queue_size = 1)
            : processing_block(block),
            _queue(queue_size)
        {
            start(_queue);
        }

        /**
        * constructor with callback function on_frame in rs2_frame_processor_callback structure assigned.
        *
        * \param[in] processing_function - function pointer of on_frame function in rs2_frame_processor_callback structure, which will be called back by invoke function .
        */
        template<class S>
        filter(S processing_function, int queue_size = 1) :
            processing_block(processing_function),
            _queue(queue_size)
        {
            start(_queue);
        }


        frame_queue get_queue() { return _queue; }
        rs2_processing_block* get() const { return _block.get(); }

        template<class T>
        bool is() const
        {
            T extension(*this);
            return extension;
        }

        template<class T>
        T as() const
        {
            T extension(*this);
            return extension;
        }

        operator bool() const { return _block.get() != nullptr; }
    protected:
        frame_queue _queue;
    };

    /**
    * Generates 3D point clouds based on a depth frame. Can also map textures from a color frame.
    */
    class pointcloud : public filter
    {
    public:
        /**
        * create pointcloud instance
        */
        pointcloud() : filter(init(), 1) {}

        pointcloud(rs2_stream stream, int index = 0) : filter(init(), 1)
        {
            set_option(RS2_OPTION_STREAM_FILTER, float(stream));
            set_option(RS2_OPTION_STREAM_INDEX_FILTER, float(index));
        }
        /**
        * Generate the pointcloud and texture mappings of depth map.
        *
        * \param[in] depth - the depth frame to generate point cloud and texture.
        * \return points instance.
        */
        points calculate(frame depth)
        {
            auto res = process(depth);
            if (res.as<points>())
                return res;

            if (auto set = res.as <frameset>())
            {
                for (auto f : set)
                {
                    if(f.as<points>())
                        return f;
                }
            }
            throw std::runtime_error("Error occured during execution of the processing block! See the log for more info");
        }
        /**
        * Map the point cloud to the given color frame.
        *
        * \param[in] mapped - the frame to be mapped to as texture.
        */
        void map_to(frame mapped)
        {
            set_option(RS2_OPTION_STREAM_FILTER, float(mapped.get_profile().stream_type()));
            set_option(RS2_OPTION_STREAM_FORMAT_FILTER, float(mapped.get_profile().format()));
            set_option(RS2_OPTION_STREAM_INDEX_FILTER, float(mapped.get_profile().stream_index()));
            process(mapped);
        }

    protected:
        pointcloud(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}

    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;

            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_pointcloud(&e),
                rs2_delete_processing_block);

            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(pb);
            return block;
        }
    };

    class yuy_decoder : public filter
    {
    public:
        /**
        * Creates YUY decoder processing block. This block accepts raw YUY frames and outputs frames of other formats.
        * YUY is a common video format used by a variety of web-cams. It benefits from packing pixels into 2 bytes per pixel
        * without signficant quality drop. YUY representation can be converted back to more usable RGB form,
        * but this requires somewhat costly conversion.
        * The SDK will automatically try to use SSE2 and AVX instructions and CUDA where available to get
        * best performance. Other implementations (using GLSL, OpenCL, Neon and NCS) should follow.
        */
        yuy_decoder() : filter(init(), 1) { }

    protected:
        yuy_decoder(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}

    private:
        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_yuy_decoder(&e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };
  
  class threshold_filter : public filter
    {
    public:
        /**
        * Creates depth thresholding filter
        * By controlling min and max options on the block, one could filter out depth values
        * that are either too large or too small, as a software post-processing step
        */
        threshold_filter(float min_dist = 0.15f, float max_dist = 4.f) 
            : filter(init(), 1) 
        { 
            set_option(RS2_OPTION_MIN_DISTANCE, min_dist);
            set_option(RS2_OPTION_MAX_DISTANCE, max_dist);
        }

        threshold_filter(filter f) : filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_THRESHOLD_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }

    protected:
        threshold_filter(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}
        
    private:
        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_threshold(&e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };

    class units_transform : public filter
    {
    public:
        /**
        * Creates depth units to meters processing block.
        */
        units_transform() : filter(init(), 1) {}

    protected:
        units_transform(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}
        
    private:
        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_units_transform(&e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };

    class asynchronous_syncer : public processing_block
    {
    public:
        /**
        * Real asynchronous syncer within syncer class
        */
        asynchronous_syncer() : processing_block(init()) {}

    private:
        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_sync_processing_block(&e),
                rs2_delete_processing_block);

            error::handle(e);
            return block;
        }
    };

    class syncer
    {
    public:
        /**
        * Sync instance to align frames from different streams
        */
        syncer(int queue_size = 1)
            :_results(queue_size)
        {
            _sync.start(_results);
        }

        /**
        * Wait until coherent set of frames becomes available
        * \param[in] timeout_ms   Max time in milliseconds to wait until an exception will be thrown
        * \return Set of coherent frames
        */
        frameset wait_for_frames(unsigned int timeout_ms = 5000) const
        {
            return frameset(_results.wait_for_frame(timeout_ms));
        }

        /**
        * Check if a coherent set of frames is available
        * \param[out] fs      New coherent frame-set
        * \return true if new frame-set was stored to result
        */
        bool poll_for_frames(frameset* fs) const
        {
            frame result;
            if (_results.poll_for_frame(&result))
            {
                *fs = frameset(result);
                return true;
            }
            return false;
        }

        /**
        * Wait until coherent set of frames becomes available
        * \param[in] timeout_ms     Max time in milliseconds to wait until an available frame
        * \param[out] fs            New coherent frame-set
        * \return true if new frame-set was stored to result
        */
        bool try_wait_for_frames(frameset* fs, unsigned int timeout_ms = 5000) const
        {
            frame result;
            if (_results.try_wait_for_frame(&result, timeout_ms))
            {
                *fs = frameset(result);
                return true;
            }
            return false;
        }

        void operator()(frame f) const
        {
            _sync.invoke(std::move(f));
        }
    private:
        asynchronous_syncer _sync;
        frame_queue _results;
    };

    /**
    Auxiliary processing block that performs image alignment using depth data and camera calibration
    */
    class align : public filter
    {
    public:
        /**
        Create align filter
        Alignment is performed between a depth image and another image.
        To perform alignment of a depth image to the other, set the align_to parameter with the other stream type.
        To perform alignment of a non depth image to a depth image, set the align_to parameter to RS2_STREAM_DEPTH.
        Camera calibration and frame's stream type are determined on the fly, according to the first valid frameset passed to process().

        * \param[in] align_to      The stream type to which alignment should be made.
        */
        align(rs2_stream align_to) : filter(init(align_to), 1) {}

        /**
        * Run the alignment process on the given frames to get an aligned set of frames
        *
        * \param[in] frames      A set of frames, where at least one of which is a depth frame
        * \return Input frames aligned to one another
        */
        frameset process(frameset frames)
        {
            return filter::process(frames);
        }

    protected:
        align(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}

    private:
        friend class context;
        std::shared_ptr<rs2_processing_block> init(rs2_stream align_to)
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_align(align_to, &e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };

    class colorizer : public filter
    {
    public:
        /**
        * Create colorizer filter
        * Colorizer generate color image based on input depth frame
        */
        colorizer() : filter(init(), 1) { }
        /**
        * Create colorizer processing block
        * Colorizer generate color image base on input depth frame
        * \param[in] color_scheme - select one of the available color schemes:
        *                           0 - Jet
        *                           1 - Classic
        *                           2 - WhiteToBlack
        *                           3 - BlackToWhite
        *                           4 - Bio
        *                           5 - Cold
        *                           6 - Warm
        *                           7 - Quantized
        *                           8 - Pattern
        *                           9 - Hue
        */
        colorizer(float color_scheme) : filter(init(), 1)
        {
            set_option(RS2_OPTION_COLOR_SCHEME, float(color_scheme));
        }
        /**
        * Start to generate color image base on depth frame
        * \param[in] depth - depth frame to be processed to generate the color image
        * \return video_frame - generated color image
        */
        video_frame colorize(frame depth) const
        {
            return process(depth);
        }

    protected:
        colorizer(std::shared_ptr<rs2_processing_block> block) : filter(block, 1) {}

    private:
        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_colorizer(&e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(pb);

            return block;
        }
    };

    class decimation_filter : public filter
    {
    public:
        /**
        * Create decimation filter
        * Decimation filter performs downsampling by using the median with specific kernel size
        */
        decimation_filter() : filter(init(), 1) {}
        /**
        * Create decimation filter
        * Decimation filter performs downsampling by using the median with specific kernel size
        * \param[in] magnitude - number of filter iterations.
        */
        decimation_filter(float magnitude) : filter(init(), 1)
        {
            set_option(RS2_OPTION_FILTER_MAGNITUDE, magnitude);
        }

        decimation_filter(filter f) : filter(f)
        {
             rs2_error* e = nullptr;
             if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_DECIMATION_FILTER, &e) && !e)
             {
                 _block.reset();
             }
             error::handle(e);
        }
       
    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_decimation_filter_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(this);

            return block;
        }
    };

    class temporal_filter : public filter
    {
    public:
        /**
        * Create temporal filter with default settings
        * Temporal filter smooths the image by calculating multiple frames with alpha and delta settings
        * alpha defines the weight of current frame, and delta defines the threshold for edge classification and preserving.
        * For more information, check the temporal-filter.cpp
        */
        temporal_filter() : filter(init(), 1) {}
        /**
        * Create temporal filter with user settings
        * Temporal filter smooths the image by calculating multiple frames with alpha and delta settings
        * \param[in] smooth_alpha - defines the weight of current frame.
        * \param[in] smooth_delta - delta defines the threshold for edge classification and preserving.
        * \param[in] persistence_control - A set of predefined rules (masks) that govern when missing pixels will be replaced with the last valid value so that the data will remain persistent over time:
        * 0 - Disabled - Persistency filter is not activated and no hole filling occurs.
        * 1 - Valid in 8/8 - Persistency activated if the pixel was valid in 8 out of the last 8 frames
        * 2 - Valid in 2/last 3 - Activated if the pixel was valid in two out of the last 3 frames
        * 3 - Valid in 2/last 4 - Activated if the pixel was valid in two out of the last 4 frames
        * 4 - Valid in 2/8 - Activated if the pixel was valid in two out of the last 8 frames
        * 5 - Valid in 1/last 2 - Activated if the pixel was valid in one of the last two frames
        * 6 - Valid in 1/last 5 - Activated if the pixel was valid in one out of the last 5 frames
        * 7 - Valid in 1/last 8 - Activated if the pixel was valid in one out of the last 8 frames
        * 8 - Persist Indefinitely - Persistency will be imposed regardless of the stored history(most aggressive filtering)
        * For more information, check temporal-filter.cpp
        */
        temporal_filter(float smooth_alpha, float smooth_delta, int persistence_control) : filter(init(), 1)
        {
            set_option(RS2_OPTION_HOLES_FILL, float(persistence_control));
            set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, float(smooth_alpha));
            set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, float(smooth_delta));
        }

        temporal_filter(filter f) :filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_TEMPORAL_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }
    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_temporal_filter_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(pb);

            return block;
        }
    };

    class spatial_filter : public filter
    {
    public:
        /**
        * Create spatial filter
        * Spatial filter smooths the image by calculating frame with alpha and delta settings
        * alpha defines the weight of the current pixel for smoothing, and is bounded within [25..100]%,
        * delta defines the depth gradient below which the smoothing will occur as number of depth levels
        * For more information, check the spatial-filter.cpp
        */
        spatial_filter() : filter(init(), 1) { }

        /**
        * Create spatial filter
        * Spatial filter smooths the image by calculating frame with alpha and delta settings
        * \param[in] smooth_alpha - defines the weight of the current pixel for smoothing is bounded within [25..100]%
        * \param[in] smooth_delta - defines the depth gradient below which the smoothing will occur as number of depth levels
        * \param[in] magnitude - number of filter iterations.
        * \param[in] hole_fill - an in-place heuristic symmetric hole-filling mode applied horizontally during the filter passes.
        *                           Intended to rectify minor artefacts with minimal performance impact
        */
        spatial_filter(float smooth_alpha, float smooth_delta, float magnitude, float hole_fill) : filter(init(), 1)
        {
            set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, float(smooth_alpha));
            set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, float(smooth_delta));
            set_option(RS2_OPTION_FILTER_MAGNITUDE, magnitude);
            set_option(RS2_OPTION_HOLES_FILL, hole_fill);
        }

        spatial_filter(filter f) :filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_SPATIAL_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }
    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_spatial_filter_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(pb);

            return block;
        }
    };

    class disparity_transform : public filter
    {
    public:
        /**
        * Create disparity transform filter
        * Converts from depth representation to disparity representation and vice-versa in depth frames
        */
        disparity_transform(bool transform_to_disparity = true) : filter(init(transform_to_disparity), 1) { }

        disparity_transform(filter f) :filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_DISPARITY_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }
    private:
        friend class context;
        std::shared_ptr<rs2_processing_block> init(bool transform_to_disparity)
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_disparity_transform_block(uint8_t(transform_to_disparity), &e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(pb);

            return block;
        }
    };
    
    class zero_order_invalidation : public filter
    {
    public:
        /**
        * Create zero order fix filter
        * The filter fixes the zero order artifact
        */
        zero_order_invalidation() : filter(init())
        {}

        zero_order_invalidation(filter f) :filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_ZERO_ORDER_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }

    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_zero_order_invalidation_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };

    class hole_filling_filter : public filter
    {
    public:
        /**
        * Create hole filling filter
        * The processing performed depends on the selected hole filling mode.
        */
        hole_filling_filter() : filter(init(), 1) {}

        /**
        * Create hole filling filter
        * The processing performed depends on the selected hole filling mode.
        * \param[in] mode - select the hole fill mode:
        * 0 - fill_from_left - Use the value from the left neighbor pixel to fill the hole
        * 1 - farest_from_around - Use the value from the neighboring pixel which is furthest away from the sensor
        * 2 - nearest_from_around - - Use the value from the neighboring pixel closest to the sensor
        */
        hole_filling_filter(int mode) : filter(init(), 1)
        {
            set_option(RS2_OPTION_HOLES_FILL, float(mode));
        }

        hole_filling_filter(filter f) :filter(f)
        {
            rs2_error* e = nullptr;
            if (!rs2_is_processing_block_extendable_to(f.get(), RS2_EXTENSION_HOLE_FILLING_FILTER, &e) && !e)
            {
                _block.reset();
            }
            error::handle(e);
        }
    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_hole_filling_filter_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            // Redirect options API to the processing block
            //options::operator=(_block);

            return block;
        }
    };

    class rates_printer : public filter
    {
    public:
        /**
        * Create hole filling processing block
        * the processing perform the hole filling base on different hole filling mode.
        */
        rates_printer() : filter(init(), 1) {}

    private:
        friend class context;

        std::shared_ptr<rs2_processing_block> init()
        {
            rs2_error* e = nullptr;
            auto block = std::shared_ptr<rs2_processing_block>(
                rs2_create_rates_printer_block(&e),
                rs2_delete_processing_block);
            error::handle(e);

            return block;
        }
    };
}
#endif // LIBREALSENSE_RS2_PROCESSING_HPP
