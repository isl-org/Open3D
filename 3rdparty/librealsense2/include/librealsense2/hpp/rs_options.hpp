// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_OPTIONS_HPP
#define LIBREALSENSE_RS2_OPTIONS_HPP

#include "rs_types.hpp"

namespace rs2
{
    class options
    {
    public:
        /**
        * check if particular option is supported
        * \param[in] option     option id to be checked
        * \return true if option is supported
        */
        bool supports(rs2_option option) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_supports_option(_options, option, &e);
            error::handle(e);
            return res > 0;
        }

        /**
        * get option description
        * \param[in] option     option id to be checked
        * \return human-readable option description
        */
        const char* get_option_description(rs2_option option) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_option_description(_options, option, &e);
            error::handle(e);
            return res;
        }

        /**
        * get option name
        * \param[in] option     option id to be checked
        * \return human-readable option name
        */
        const char* get_option_name(rs2_option option) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_option_name(_options, option, &e);
            error::handle(e);
            return res;
        }

        /**
        * get option value description (in case specific option value hold special meaning)
        * \param[in] option     option id to be checked
        * \param[in] val      value of the option
        * \return human-readable description of a specific value of an option or null if no special meaning
        */
        const char* get_option_value_description(rs2_option option, float val) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_option_value_description(_options, option, val, &e);
            error::handle(e);
            return res;
        }

        /**
        * read option's value
        * \param[in] option   option id to be queried
        * \return value of the option
        */
        float get_option(rs2_option option) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_get_option(_options, option, &e);
            error::handle(e);
            return res;
        }

        /**
        * retrieve the available range of values of a supported option
        * \return option  range containing minimum and maximum values, step and default value
        */
        option_range get_option_range(rs2_option option) const
        {
            option_range result;
            rs2_error* e = nullptr;
            rs2_get_option_range(_options, option,
                &result.min, &result.max, &result.step, &result.def, &e);
            error::handle(e);
            return result;
        }

        /**
        * write new value to the option
        * \param[in] option     option id to be queried
        * \param[in] value      new value for the option
        */
        void set_option(rs2_option option, float value) const
        {
            rs2_error* e = nullptr;
            rs2_set_option(_options, option, value, &e);
            error::handle(e);
        }

        /**
        * check if particular option is read-only
        * \param[in] option     option id to be checked
        * \return true if option is read-only
        */
        bool is_option_read_only(rs2_option option) const
        {
            rs2_error* e = nullptr;
            auto res = rs2_is_option_read_only(_options, option, &e);
            error::handle(e);
            return res > 0;
        }

        std::vector<rs2_option> get_supported_options()
        {
            std::vector<rs2_option> res;
            rs2_error* e = nullptr;
            std::shared_ptr<rs2_options_list> options_list(
                rs2_get_options_list(_options, &e),
                rs2_delete_options_list);
        

            for (auto opt = 0; opt < rs2_get_options_list_size(options_list.get(), &e);opt++)
            {
                res.push_back(rs2_get_option_from_list(options_list.get(), opt, &e));
            }
            return res;
        };

        options& operator=(const options& other)
        {
            _options = other._options;
            return *this;
        }
        // if operator= is ok, this should be ok too
        options(const options& other) : _options(other._options) {}

        virtual ~options() = default;
    protected:
        explicit options(rs2_options* o = nullptr) : _options(o) 
        {
        }

        template<class T>
        options& operator=(const T& dev)
        {
            _options = (rs2_options*)(dev.get());
            return *this;
        }

    private:
        rs2_options* _options;
    };
}
#endif // LIBREALSENSE_RS2_OIPTIONS_HPP
