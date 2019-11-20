/* License: Apache 2.0. See LICENSE file in root directory.
Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_option.h
* \brief
* Exposes sensor options functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_OPTION_H
#define LIBREALSENSE_RS2_OPTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rs_types.h"

    /** \brief Defines general configuration controls.
    These can generally be mapped to camera UVC controls, and can be set / queried at any time unless stated otherwise.
    */
    typedef enum rs2_option
    {
        RS2_OPTION_BACKLIGHT_COMPENSATION, /**< Enable / disable color backlight compensation*/
        RS2_OPTION_BRIGHTNESS, /**< Color image brightness*/
        RS2_OPTION_CONTRAST, /**< Color image contrast*/
        RS2_OPTION_EXPOSURE, /**< Controls exposure time of color camera. Setting any value will disable auto exposure*/
        RS2_OPTION_GAIN, /**< Color image gain*/
        RS2_OPTION_GAMMA, /**< Color image gamma setting*/
        RS2_OPTION_HUE, /**< Color image hue*/
        RS2_OPTION_SATURATION, /**< Color image saturation setting*/
        RS2_OPTION_SHARPNESS, /**< Color image sharpness setting*/
        RS2_OPTION_WHITE_BALANCE, /**< Controls white balance of color image. Setting any value will disable auto white balance*/
        RS2_OPTION_ENABLE_AUTO_EXPOSURE, /**< Enable / disable color image auto-exposure*/
        RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, /**< Enable / disable color image auto-white-balance*/
        RS2_OPTION_VISUAL_PRESET, /**< Provide access to several recommend sets of option presets for the depth camera */
        RS2_OPTION_LASER_POWER, /**< Power of the F200 / SR300 projector, with 0 meaning projector off*/
        RS2_OPTION_ACCURACY, /**< Set the number of patterns projected per frame. The higher the accuracy value the more patterns projected. Increasing the number of patterns help to achieve better accuracy. Note that this control is affecting the Depth FPS */
        RS2_OPTION_MOTION_RANGE, /**< Motion vs. Range trade-off, with lower values allowing for better motion sensitivity and higher values allowing for better depth range*/
        RS2_OPTION_FILTER_OPTION, /**< Set the filter to apply to each depth frame. Each one of the filter is optimized per the application requirements*/
        RS2_OPTION_CONFIDENCE_THRESHOLD, /**< The confidence level threshold used by the Depth algorithm pipe to set whether a pixel will get a valid range or will be marked with invalid range*/
        RS2_OPTION_EMITTER_ENABLED, /**< Laser Emitter enabled */
        RS2_OPTION_FRAMES_QUEUE_SIZE, /**< Number of frames the user is allowed to keep per stream. Trying to hold-on to more frames will cause frame-drops.*/
        RS2_OPTION_TOTAL_FRAME_DROPS, /**< Total number of detected frame drops from all streams */
        RS2_OPTION_AUTO_EXPOSURE_MODE, /**< Auto-Exposure modes: Static, Anti-Flicker and Hybrid */
        RS2_OPTION_POWER_LINE_FREQUENCY, /**< Power Line Frequency control for anti-flickering Off/50Hz/60Hz/Auto */
        RS2_OPTION_ASIC_TEMPERATURE, /**< Current Asic Temperature */
        RS2_OPTION_ERROR_POLLING_ENABLED, /**< disable error handling */
        RS2_OPTION_PROJECTOR_TEMPERATURE, /**< Current Projector Temperature */
        RS2_OPTION_OUTPUT_TRIGGER_ENABLED, /**< Enable / disable trigger to be outputed from the camera to any external device on every depth frame */
        RS2_OPTION_MOTION_MODULE_TEMPERATURE, /**< Current Motion-Module Temperature */
        RS2_OPTION_DEPTH_UNITS, /**< Number of meters represented by a single depth unit */
        RS2_OPTION_ENABLE_MOTION_CORRECTION, /**< Enable/Disable automatic correction of the motion data */
        RS2_OPTION_AUTO_EXPOSURE_PRIORITY, /**< Allows sensor to dynamically ajust the frame rate depending on lighting conditions */
        RS2_OPTION_COLOR_SCHEME, /**< Color scheme for data visualization */
        RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, /**< Perform histogram equalization post-processing on the depth data */
        RS2_OPTION_MIN_DISTANCE, /**< Minimal distance to the target */
        RS2_OPTION_MAX_DISTANCE, /**< Maximum distance to the target */
        RS2_OPTION_TEXTURE_SOURCE, /**< Texture mapping stream unique ID */
        RS2_OPTION_FILTER_MAGNITUDE, /**< The 2D-filter effect. The specific interpretation is given within the context of the filter */
        RS2_OPTION_FILTER_SMOOTH_ALPHA, /**< 2D-filter parameter controls the weight/radius for smoothing.*/
        RS2_OPTION_FILTER_SMOOTH_DELTA, /**< 2D-filter range/validity threshold*/
        RS2_OPTION_HOLES_FILL, /**< Enhance depth data post-processing with holes filling where appropriate*/
        RS2_OPTION_STEREO_BASELINE, /**< The distance in mm between the first and the second imagers in stereo-based depth cameras*/
        RS2_OPTION_AUTO_EXPOSURE_CONVERGE_STEP, /**< Allows dynamically ajust the converge step value of the target exposure in Auto-Exposure algorithm*/
        RS2_OPTION_INTER_CAM_SYNC_MODE, /**< Impose Inter-camera HW synchronization mode. Applicable for D400/Rolling Shutter SKUs */
        RS2_OPTION_STREAM_FILTER, /**< Select a stream to process */
        RS2_OPTION_STREAM_FORMAT_FILTER, /**< Select a stream format to process */
        RS2_OPTION_STREAM_INDEX_FILTER, /**< Select a stream index to process */
        RS2_OPTION_EMITTER_ON_OFF, /**< When supported, this option make the camera to switch the emitter state every frame. 0 for disabled, 1 for enabled */
        RS2_OPTION_ZERO_ORDER_POINT_X, /**< Zero order point x*/
        RS2_OPTION_ZERO_ORDER_POINT_Y, /**< Zero order point y*/
        RS2_OPTION_LLD_TEMPERATURE, /**< LLD temperature*/
        RS2_OPTION_MC_TEMPERATURE, /**< MC temperature*/
        RS2_OPTION_MA_TEMPERATURE, /**< MA temperature*/
        RS2_OPTION_HARDWARE_PRESET, /**< Hardware stream configuration */
        RS2_OPTION_GLOBAL_TIME_ENABLED, /**< disable global time  */
        RS2_OPTION_APD_TEMPERATURE, /**< APD temperature*/
        RS2_OPTION_ENABLE_MAPPING, /**< Enable an internal map */
        RS2_OPTION_ENABLE_RELOCALIZATION, /**< Enable appearance based relocalization */
        RS2_OPTION_ENABLE_POSE_JUMPING, /**< Enable position jumping */
        RS2_OPTION_ENABLE_DYNAMIC_CALIBRATION, /**< Enable dynamic calibration */
        RS2_OPTION_DEPTH_OFFSET, /**< Offset from sensor to depth origin in millimetrers*/
        RS2_OPTION_COUNT /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
    } rs2_option;

    // This function is being deprecated. For existing options it will return option name, but for future API additions the user should call rs2_get_option_name instead.
    const char* rs2_option_to_string(rs2_option option);

    /** \brief For SR300 devices: provides optimized settings (presets) for specific types of usage. */
    typedef enum rs2_sr300_visual_preset
    {
        RS2_SR300_VISUAL_PRESET_SHORT_RANGE, /**< Preset for short range */
        RS2_SR300_VISUAL_PRESET_LONG_RANGE, /**< Preset for long range */
        RS2_SR300_VISUAL_PRESET_BACKGROUND_SEGMENTATION, /**< Preset for background segmentation */
        RS2_SR300_VISUAL_PRESET_GESTURE_RECOGNITION, /**< Preset for gesture recognition */
        RS2_SR300_VISUAL_PRESET_OBJECT_SCANNING, /**< Preset for object scanning */
        RS2_SR300_VISUAL_PRESET_FACE_ANALYTICS, /**< Preset for face analytics */
        RS2_SR300_VISUAL_PRESET_FACE_LOGIN, /**< Preset for face login */
        RS2_SR300_VISUAL_PRESET_GR_CURSOR, /**< Preset for GR cursor */
        RS2_SR300_VISUAL_PRESET_DEFAULT, /**< Camera default settings */
        RS2_SR300_VISUAL_PRESET_MID_RANGE, /**< Preset for mid-range */
        RS2_SR300_VISUAL_PRESET_IR_ONLY, /**< Preset for IR only */
        RS2_SR300_VISUAL_PRESET_COUNT                           /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
    } rs2_sr300_visual_preset;
    const char* rs2_sr300_visual_preset_to_string(rs2_sr300_visual_preset preset);

    /** \brief For RS400 devices: provides optimized settings (presets) for specific types of usage. */
    typedef enum rs2_rs400_visual_preset
    {
        RS2_RS400_VISUAL_PRESET_CUSTOM,
        RS2_RS400_VISUAL_PRESET_DEFAULT,
        RS2_RS400_VISUAL_PRESET_HAND,
        RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY,
        RS2_RS400_VISUAL_PRESET_HIGH_DENSITY,
        RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY,
        RS2_RS400_VISUAL_PRESET_REMOVE_IR_PATTERN,
        RS2_RS400_VISUAL_PRESET_COUNT
    } rs2_rs400_visual_preset;
    const char* rs2_rs400_visual_preset_to_string(rs2_rs400_visual_preset preset);

    /**
    * check if an option is read-only
    * \param[in] sensor   the RealSense sensor
    * \param[in] option   option id to be checked
    * \param[out] error   if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return true if option is read-only
    */
    int rs2_is_option_read_only(const rs2_options* options, rs2_option option, rs2_error** error);

    /**
    * read option value from the sensor
    * \param[in] sensor   the RealSense sensor
    * \param[in] option   option id to be queried
    * \param[out] error   if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return value of the option
    */
    float rs2_get_option(const rs2_options* options, rs2_option option, rs2_error** error);

    /**
    * write new value to sensor option
    * \param[in] sensor     the RealSense sensor
    * \param[in] option     option id to be queried
    * \param[in] value      new value for the option
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_set_option(const rs2_options* options, rs2_option option, float value, rs2_error** error);

   /**
   * get the list of supported options of options container
   * \param[in] options    the options container
   * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
   */
    rs2_options_list* rs2_get_options_list(const rs2_options* options, rs2_error** error);

    /**
   * get the size of options list
   * \param[in] options    the option list
   * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
   */
    int rs2_get_options_list_size(const rs2_options_list* options, rs2_error** error);

    /**
    * get option name
    * \param[in] options     options object
    * \param[in] option      option id to be checked
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return human-readable option name
    */
    const char* rs2_get_option_name(const rs2_options* options, rs2_option option, rs2_error** error);

    /**
   * get the specific option from options list
   * \param[in] i    the index of the option
   * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
   */
    rs2_option rs2_get_option_from_list(const rs2_options_list* options, int i, rs2_error** error);

    /**
    * Deletes options list
    * \param[in] list list to delete
    */
    void rs2_delete_options_list(rs2_options_list* list);

    /**
    * check if particular option is supported by a subdevice
    * \param[in] sensor     the RealSense sensor
    * \param[in] option     option id to be checked
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return true if option is supported
    */
    int rs2_supports_option(const rs2_options* options, rs2_option option, rs2_error** error);

    /**
    * retrieve the available range of values of a supported option
    * \param[in] sensor  the RealSense device
    * \param[in] option  the option whose range should be queried
    * \param[out] min    the minimum value which will be accepted for this option
    * \param[out] max    the maximum value which will be accepted for this option
    * \param[out] step   the granularity of options which accept discrete values, or zero if the option accepts continuous values
    * \param[out] def    the default value of the option
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_get_option_range(const rs2_options* sensor, rs2_option option, float* min, float* max, float* step, float* def, rs2_error** error);

    /**
    * get option description
    * \param[in] sensor     the RealSense sensor
    * \param[in] option     option id to be checked
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return human-readable option description
    */
    const char* rs2_get_option_description(const rs2_options* options, rs2_option option, rs2_error ** error);

    /**
    * get option value description (in case specific option value hold special meaning)
    * \param[in] device     the RealSense device
    * \param[in] option     option id to be checked
    * \param[in] value      value of the option
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return human-readable description of a specific value of an option or null if no special meaning
    */
    const char* rs2_get_option_value_description(const rs2_options* options, rs2_option option, float value, rs2_error ** error);

#ifdef __cplusplus
}
#endif
#endif
