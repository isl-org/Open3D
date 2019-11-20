// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_SENSOR_HPP
#define LIBREALSENSE_RS2_SENSOR_HPP

#include "rs_types.hpp"
#include "rs_frame.hpp"
#include "rs_processing.hpp"
#include "rs_options.hpp"
namespace rs2
{

    class notification
    {
    public:
        notification(rs2_notification* nt)
        {
            rs2_error* e = nullptr;
            _description = rs2_get_notification_description(nt, &e);
            error::handle(e);
            _timestamp = rs2_get_notification_timestamp(nt, &e);
            error::handle(e);
            _severity = rs2_get_notification_severity(nt, &e);
            error::handle(e);
            _category = rs2_get_notification_category(nt, &e);
            error::handle(e);
            _serialized_data = rs2_get_notification_serialized_data(nt, &e);
            error::handle(e);
        }

        notification() = default;

        /**
        * retrieve the notification category
        * \return            the notification category
        */
        rs2_notification_category get_category() const
        {
            return _category;
        }
        /**
        * retrieve the notification description
        * \return            the notification description
        */
        std::string get_description() const
        {
            return _description;
        }

        /**
        * retrieve the notification arrival timestamp
        * \return            the arrival timestamp
        */
        double get_timestamp() const
        {
            return _timestamp;
        }

        /**
        * retrieve the notification severity
        * \return            the severity
        */
        rs2_log_severity get_severity() const
        {
            return _severity;
        }

        /**
        * retrieve the notification's serialized data
        * \return            the serialized data (in JSON format)
        */
        std::string get_serialized_data() const
        {
            return _serialized_data;
        }

    private:
        std::string _description;
        double _timestamp = -1;
        rs2_log_severity _severity = RS2_LOG_SEVERITY_COUNT;
        rs2_notification_category _category = RS2_NOTIFICATION_CATEGORY_COUNT;
        std::string _serialized_data;
    };

    template<class T>
    class notifications_callback : public rs2_notifications_callback
    {
        T on_notification_function;
    public:
        explicit notifications_callback(T on_notification) : on_notification_function(on_notification) {}

        void on_notification(rs2_notification* _notification) override
        {
            on_notification_function(notification{ _notification });
        }

        void release() override { delete this; }
    };



    class sensor : public options
    {
    public:

        using options::supports;
        /**
        * open sensor for exclusive access, by committing to a configuration
        * \param[in] profile    configuration committed by the sensor
        */
        void open(const stream_profile& profile) const
        {
            rs2_error* e = nullptr;
            rs2_open(_sensor.get(),
                profile.get(),
                &e);
            error::handle(e);
        }

        /**
        * check if specific camera info is supported
        * \param[in] info    the parameter to check for support
        * \return                true if the parameter both exist and well-defined for the specific sensor
        */
        bool supports(rs2_camera_info info) const
        {
            rs2_error* e = nullptr;
            auto is_supported = rs2_supports_sensor_info(_sensor.get(), info, &e);
            error::handle(e);
            return is_supported > 0;
        }

        /**
        * retrieve camera specific information, like versions of various internal components
        * \param[in] info     camera info type to retrieve
        * \return             the requested camera info string, in a format specific to the sensor model
        */
        const char* get_info(rs2_camera_info info) const
        {
            rs2_error* e = nullptr;
            auto result = rs2_get_sensor_info(_sensor.get(), info, &e);
            error::handle(e);
            return result;
        }

        /**
        * open sensor for exclusive access, by committing to composite configuration, specifying one or more stream profiles
        * this method should be used for interdependent  streams, such as depth and infrared, that have to be configured together
        * \param[in] profiles   vector of configurations to be commited by the sensor
        */
        void open(const std::vector<stream_profile>& profiles) const
        {
            rs2_error* e = nullptr;

            std::vector<const rs2_stream_profile*> profs;
            profs.reserve(profiles.size());
            for (auto& p : profiles)
            {
                profs.push_back(p.get());
            }

            rs2_open_multiple(_sensor.get(),
                profs.data(),
                static_cast<int>(profiles.size()),
                &e);
            error::handle(e);
        }

        /**
        * close sensor for exclusive access
        * this method should be used for releasing sensor resource
        */
        void close() const
        {
            rs2_error* e = nullptr;
            rs2_close(_sensor.get(), &e);
            error::handle(e);
        }

        /**
        * Start passing frames into user provided callback
        * \param[in] callback   Stream callback, can be any callable object accepting rs2::frame
        */
        template<class T>
        void start(T callback) const
        {
            rs2_error* e = nullptr;
            rs2_start_cpp(_sensor.get(), new frame_callback<T>(std::move(callback)), &e);
            error::handle(e);
        }

        /**
        * stop streaming
        */
        void stop() const
        {
            rs2_error* e = nullptr;
            rs2_stop(_sensor.get(), &e);
            error::handle(e);
        }

        /**
        * register notifications callback
        * \param[in] callback   notifications callback
        */
        template<class T>
        void set_notifications_callback(T callback) const
        {
            rs2_error* e = nullptr;
            rs2_set_notifications_callback_cpp(_sensor.get(),
                new notifications_callback<T>(std::move(callback)), &e);
            error::handle(e);
        }


        /**
        * Retrieves the list of stream profiles supported by the sensor.
        * \return   list of stream profiles that given sensor can provide
        */
        std::vector<stream_profile> get_stream_profiles() const
        {
            std::vector<stream_profile> results{};

            rs2_error* e = nullptr;
            std::shared_ptr<rs2_stream_profile_list> list(
                rs2_get_stream_profiles(_sensor.get(), &e),
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
        * get the recommended list of filters by the sensor
        * \return   list of filters that recommended by sensor
        */
        std::vector<filter> get_recommended_filters() const
        {
            std::vector<filter> results{};

            rs2_error* e = nullptr;
            std::shared_ptr<rs2_processing_block_list> list(
                rs2_get_recommended_processing_blocks(_sensor.get(), &e),
                rs2_delete_recommended_processing_blocks);
            error::handle(e);

            auto size =  rs2_get_recommended_processing_blocks_count(list.get(), &e);
            error::handle(e);

            for (auto i = 0; i < size; i++)
            {
                auto f = std::shared_ptr<rs2_processing_block>(
                    rs2_get_processing_block(list.get(), i, &e),
                    rs2_delete_processing_block);
                error::handle(e);
                results.push_back(f);
            }

            return results;
        }

        sensor& operator=(const std::shared_ptr<rs2_sensor> other)
        {
            options::operator=(other);
            _sensor.reset();
            _sensor = other;
            return *this;
        }

        sensor& operator=(const sensor& other)
        {
            *this = nullptr;
             options::operator=(other._sensor);
            _sensor = other._sensor;
            return *this;
        }
        sensor() : _sensor(nullptr) {}

        operator bool() const
        {
            return _sensor != nullptr;
        }

        const std::shared_ptr<rs2_sensor>& get() const
        {
            return _sensor;
        }

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

        explicit sensor(std::shared_ptr<rs2_sensor> dev)
            :options((rs2_options*)dev.get()), _sensor(dev)
        {
        }
        explicit operator std::shared_ptr<rs2_sensor>() { return _sensor; }

    protected:
        friend context;
        friend device_list;
        friend device;
        friend device_base;
        friend roi_sensor;

        std::shared_ptr<rs2_sensor> _sensor;


    };

    inline std::shared_ptr<sensor> sensor_from_frame(frame f)
    {
        std::shared_ptr<rs2_sensor> psens(f.get_sensor(), rs2_delete_sensor);
        return std::make_shared<sensor>(psens);
    }

    inline bool operator==(const sensor& lhs, const sensor& rhs)
    {
        if (!(lhs.supports(RS2_CAMERA_INFO_NAME) && lhs.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)
            && rhs.supports(RS2_CAMERA_INFO_NAME) && rhs.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)))
            return false;

        return std::string(lhs.get_info(RS2_CAMERA_INFO_NAME)) == rhs.get_info(RS2_CAMERA_INFO_NAME)
            && std::string(lhs.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) == rhs.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    }

    class roi_sensor : public sensor
    {
    public:
        roi_sensor(sensor s)
            : sensor(s.get())
        {
            rs2_error* e = nullptr;
            if(rs2_is_sensor_extendable_to(_sensor.get(), RS2_EXTENSION_ROI, &e) == 0 && !e)
            {
                _sensor.reset();
            }
            error::handle(e);
        }

        void set_region_of_interest(const region_of_interest& roi)
        {
            rs2_error* e = nullptr;
            rs2_set_region_of_interest(_sensor.get(), roi.min_x, roi.min_y, roi.max_x, roi.max_y, &e);
            error::handle(e);
        }

        region_of_interest get_region_of_interest() const
        {
            region_of_interest roi {};
            rs2_error* e = nullptr;
            rs2_get_region_of_interest(_sensor.get(), &roi.min_x, &roi.min_y, &roi.max_x, &roi.max_y, &e);
            error::handle(e);
            return roi;
        }

        operator bool() const { return _sensor.get() != nullptr; }
    };

    class depth_sensor : public sensor
    {
    public:
        depth_sensor(sensor s)
            : sensor(s.get())
        {
            rs2_error* e = nullptr;
            if (rs2_is_sensor_extendable_to(_sensor.get(), RS2_EXTENSION_DEPTH_SENSOR, &e) == 0 && !e)
            {
                _sensor.reset();
            }
            error::handle(e);
        }

        /** Retrieves mapping between the units of the depth image and meters
        * \return depth in meters corresponding to a depth value of 1
        */
        float get_depth_scale() const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_depth_scale(_sensor.get(), &e);
            error::handle(e);
            return res;
        }

        operator bool() const { return _sensor.get() != nullptr; }
        explicit depth_sensor(std::shared_ptr<rs2_sensor> dev) : depth_sensor(sensor(dev)) {}
    };

    class depth_stereo_sensor : public depth_sensor
    {
    public:
        depth_stereo_sensor(sensor s): depth_sensor(s)
        {
            rs2_error* e = nullptr;
            if (_sensor && rs2_is_sensor_extendable_to(_sensor.get(), RS2_EXTENSION_DEPTH_STEREO_SENSOR, &e) == 0 && !e)
            {
                _sensor.reset();
            }
            error::handle(e);
        }

        /**
        * Retrieve the stereoscopic baseline value from the sensor.
        */
        float get_stereo_baseline() const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_stereo_baseline(_sensor.get(), &e);
            error::handle(e);
            return res;
        }

        operator bool() const { return _sensor.get() != nullptr; }
    };


    class pose_sensor : public sensor
    {
    public:
        pose_sensor(sensor s)
            : sensor(s.get())
        {
            rs2_error* e = nullptr;
            if (rs2_is_sensor_extendable_to(_sensor.get(), RS2_EXTENSION_POSE_SENSOR, &e) == 0 && !e)
            {
                _sensor.reset();
            }
            error::handle(e);
        }

        /**
         * Load relocalization map onto device. Only one relocalization map can be imported at a time;
         * any previously existing map will be overwritten.
         * The imported map exists simultaneously with the map created during the most tracking session after start(),
         * and they are merged after the imported map is relocalized.
         * This operation must be done before start().
         * \param[in] lmap_buf map data as a binary blob
         * \return true if success
         */
        bool import_localization_map(const std::vector<uint8_t>& lmap_buf) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_import_localization_map(_sensor.get(), lmap_buf.data(), uint32_t(lmap_buf.size()), &e);
            error::handle(e);
            return !!res;
        }

        /**
         * Get relocalization map that is currently on device, created and updated during most recent tracking session.
         * Can be called before or after stop().
         * \return map data as a binary blob
         */
        std::vector<uint8_t> export_localization_map() const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<const rs2_raw_data_buffer> loc_map(
                    rs2_export_localization_map(_sensor.get(), &e),
                    rs2_delete_raw_data);
            error::handle(e);

            auto start = rs2_get_raw_data(loc_map.get(), &e);
            error::handle(e);

            std::vector<uint8_t> results;
            if (start)
            {
                auto size = rs2_get_raw_data_size(loc_map.get(), &e);
                error::handle(e);

                results = std::vector<uint8_t>(start, start + size);
            }
            return results;
        }

        /**
         * Creates a named virtual landmark in the current map, known as static node.
         * The static node's pose is provided relative to the origin of current coordinate system of device poses.
         * This function fails if the current tracker confidence is below 3 (high confidence).
         * \param[in] guid unique name of the static node (limited to 127 chars).
         * If a static node with the same name already exists in the current map or the imported map, the static node is overwritten.
         * \param[in] pos position of the static node in the 3D space.
         * \param[in] orient_quat orientation of the static node in the 3D space, represented by a unit quaternion.
         * \return true if success.
         */
        bool set_static_node(const std::string& guid, const rs2_vector& pos, const rs2_quaternion& orient) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_set_static_node(_sensor.get(), guid.c_str(), pos, orient, &e);
            error::handle(e);
            return !!res;
        }


        /**
         * Gets the current pose of a static node that was created in the current map or in an imported map.
         * Static nodes of imported maps are available after relocalizing the imported map.
         * The static node's pose is returned relative to the current origin of coordinates of device poses.
         * Thus, poses of static nodes of an imported map are consistent with current device poses after relocalization.
         * This function fails if the current tracker confidence is below 3 (high confidence).
         * \param[in] guid unique name of the static node (limited to 127 chars).
         * \param[out] pos position of the static node in the 3D space.
         * \param[out] orient_quat orientation of the static node in the 3D space, represented by a unit quaternion.
         * \return true if success.
         */
        bool get_static_node(const std::string& guid, rs2_vector& pos, rs2_quaternion& orient) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_static_node(_sensor.get(), guid.c_str(), &pos, &orient, &e);
            error::handle(e);
            return !!res;
        }

        operator bool() const { return _sensor.get() != nullptr; }
        explicit pose_sensor(std::shared_ptr<rs2_sensor> dev) : pose_sensor(sensor(dev)) {}
    };

    class wheel_odometer : public sensor
    {
    public:
        wheel_odometer(sensor s)
            : sensor(s.get())
        {
            rs2_error* e = nullptr;
            if (rs2_is_sensor_extendable_to(_sensor.get(), RS2_EXTENSION_WHEEL_ODOMETER, &e) == 0 && !e)
            {
                _sensor.reset();
            }
            error::handle(e);
        }

        /** Load Wheel odometer settings from host to device
        * \param[in] odometry_config_buf   odometer configuration/calibration blob serialized from jsom file
        * \return true on success
        */
        bool load_wheel_odometery_config(const std::vector<uint8_t>& odometry_config_buf) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_load_wheel_odometry_config(_sensor.get(), odometry_config_buf.data(), uint32_t(odometry_config_buf.size()), &e);
            error::handle(e);
            return !!res;
        }

        /** Send wheel odometry data for each individual sensor (wheel)
        * \param[in] wo_sensor_id       - Zero-based index of (wheel) sensor with the same type within device
        * \param[in] frame_num          - Monotonocally increasing frame number, managed per sensor.
        * \param[in] translational_velocity   - Translational velocity in meter/sec
        * \return true on success
        */
        bool send_wheel_odometry(uint8_t wo_sensor_id, uint32_t frame_num, const rs2_vector& translational_velocity)
        {
            rs2_error* e = nullptr;
            auto res = rs2_send_wheel_odometry(_sensor.get(), wo_sensor_id, frame_num, translational_velocity, &e);
            error::handle(e);
            return !!res;
        }

        operator bool() const { return _sensor.get() != nullptr; }
        explicit wheel_odometer(std::shared_ptr<rs2_sensor> dev) : wheel_odometer(sensor(dev)) {}
    };
}
#endif // LIBREALSENSE_RS2_SENSOR_HPP
