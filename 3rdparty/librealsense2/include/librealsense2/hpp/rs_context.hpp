// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_CONTEXT_HPP
#define LIBREALSENSE_RS2_CONTEXT_HPP

#include "rs_types.hpp"
#include "rs_record_playback.hpp"
#include "rs_processing.hpp"

namespace rs2
{
    class event_information
    {
    public:
        event_information(device_list removed, device_list added)
            :_removed(removed), _added(added) {}

        /**
        * check if a specific device was disconnected
        * \return            true if device disconnected, false if device connected
        */
        bool was_removed(const rs2::device& dev) const
        {
            rs2_error* e = nullptr;

            if (!dev)
                return false;

            auto res = rs2_device_list_contains(_removed.get_list(), dev.get().get(), &e);
            error::handle(e);

            return res > 0;
        }

        /**
        * check if a specific device was added
        * \return            true if device added, false otherwise
        */
        bool was_added(const rs2::device& dev) const
        {
            rs2_error* e = nullptr;

            if (!dev)
                return false;

            auto res = rs2_device_list_contains(_added.get_list(), dev.get().get(), &e);
            error::handle(e);

            return res > 0;
        }

        /**
        * returns a list of all newly connected devices
        * \return            the list of all new connected devices
        */
        device_list get_new_devices()  const
        {
            return _added;
        }

    private:
        device_list _removed;
        device_list _added;
    };

    template<class T>
    class devices_changed_callback : public rs2_devices_changed_callback
    {
        T _callback;

    public:
        explicit devices_changed_callback(T callback) : _callback(callback) {}

        void on_devices_changed(rs2_device_list* removed, rs2_device_list* added) override
        {
            std::shared_ptr<rs2_device_list> old(removed, rs2_delete_device_list);
            std::shared_ptr<rs2_device_list> news(added, rs2_delete_device_list);


            event_information info({ device_list(old), device_list(news) });
            _callback(info);
        }

        void release() override { delete this; }
    };

    class pipeline;
    class device_hub;
    class software_device;

    /**
    * default librealsense context class
    * includes realsense API version as provided by RS2_API_VERSION macro
    */
    class context
    {
    public:
        context()
        {
            rs2_error* e = nullptr;
            _context = std::shared_ptr<rs2_context>(
                rs2_create_context(RS2_API_VERSION, &e),
                rs2_delete_context);
            error::handle(e);
        }

        /**
        * create a static snapshot of all connected devices at the time of the call
        * \return            the list of devices connected devices at the time of the call
        */
        device_list query_devices() const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_device_list> list(
                rs2_query_devices(_context.get(), &e),
                rs2_delete_device_list);
            error::handle(e);

            return device_list(list);
        }

        /**
        * create a static snapshot of all connected devices at the time of the call
        * \return            the list of devices connected devices at the time of the call
        */
        device_list query_devices(int mask) const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_device_list> list(
                rs2_query_devices_ex(_context.get(), mask, &e),
                rs2_delete_device_list);
            error::handle(e);

            return device_list(list);
        }

        /**
         * @brief Generate a flat list of all available sensors from all RealSense devices
         * @return List of sensors
         */
        std::vector<sensor> query_all_sensors() const
        {
            std::vector<sensor> results;
            for (auto&& dev : query_devices())
            {
                auto sensors = dev.query_sensors();
                std::copy(sensors.begin(), sensors.end(), std::back_inserter(results));
            }
            return results;
        }


        device get_sensor_parent(const sensor& s) const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_device> dev(
                rs2_create_device_from_sensor(s._sensor.get(), &e),
                rs2_delete_device);
            error::handle(e);
            return device{ dev };
        }

        /**
        * register devices changed callback
        * \param[in] callback   devices changed callback
        */
        template<class T>
        void set_devices_changed_callback(T callback)
        {
            rs2_error* e = nullptr;
            rs2_set_devices_changed_callback_cpp(_context.get(),
                new devices_changed_callback<T>(std::move(callback)), &e);
            error::handle(e);
        }

        /**
         * Creates a device from a RealSense file
         *
         * On successful load, the device will be appended to the context and a devices_changed event triggered
         * @param file  Path to a RealSense File
         * @return A playback device matching the given file
         */
        playback load_device(const std::string& file)
        {
            rs2_error* e = nullptr;
            auto device = std::shared_ptr<rs2_device>(
                rs2_context_add_device(_context.get(), file.c_str(), &e),
                rs2_delete_device);
            rs2::error::handle(e);

            return playback { device };
        }

        void unload_device(const std::string& file)
        {
            rs2_error* e = nullptr;
            rs2_context_remove_device(_context.get(), file.c_str(), &e);
            rs2::error::handle(e);
        }

        void unload_tracking_module()
        {
            rs2_error* e = nullptr;
            rs2_context_unload_tracking_module(_context.get(), &e);
            rs2::error::handle(e);
        }

        context(std::shared_ptr<rs2_context> ctx)
            : _context(ctx)
        {}
        explicit operator std::shared_ptr<rs2_context>() { return _context; };
    protected:
        friend class rs2::pipeline;
        friend class rs2::device_hub;
        friend class rs2::software_device;

        std::shared_ptr<rs2_context> _context;
    };

    /**
    * device_hub class - encapsulate the handling of connect and disconnect events
    */
    class device_hub
    {
    public:
        explicit device_hub(context ctx)
        {
            rs2_error* e = nullptr;
            _device_hub = std::shared_ptr<rs2_device_hub>(
                rs2_create_device_hub(ctx._context.get(), &e),
                rs2_delete_device_hub);
            error::handle(e);
        }

        /**
        * If any device is connected return it, otherwise wait until next RealSense device connects.
        * Calling this method multiple times will cycle through connected devices
        */
        device wait_for_device() const
        {
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_device> dev(
                rs2_device_hub_wait_for_device(_device_hub.get(), &e),
                rs2_delete_device);

            error::handle(e);

            return device(dev);

        }

        /**
        * Checks if device is still connected
        */
        bool is_connected(const device& dev) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_device_hub_is_device_connected(_device_hub.get(), dev._dev.get(), &e);
            error::handle(e);

            return res > 0 ? true : false;

        }

        explicit operator std::shared_ptr<rs2_device_hub>() { return _device_hub; }
        explicit device_hub(std::shared_ptr<rs2_device_hub> hub) : _device_hub(std::move(hub)) {}
    private:
        std::shared_ptr<rs2_device_hub> _device_hub;
    };

}
#endif // LIBREALSENSE_RS2_CONTEXT_HPP
