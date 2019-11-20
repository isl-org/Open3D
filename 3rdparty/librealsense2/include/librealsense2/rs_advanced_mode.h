/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

#ifndef RS400_ADVANCED_MODE_H
#define RS400_ADVANCED_MODE_H

#define RS400_ADVANCED_MODE_HPP
#include "h/rs_advanced_mode_command.h"
#undef RS400_ADVANCED_MODE_HPP

#include "h/rs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Enable/Disable Advanced-Mode */
void rs2_toggle_advanced_mode(rs2_device* dev, int enable, rs2_error** error);

/* Check if Advanced-Mode is enabled */
void rs2_is_enabled(rs2_device* dev, int* enabled, rs2_error** error);

/* Sets new values for Depth Control Group, returns 0 if success */
void rs2_set_depth_control(rs2_device* dev, const STDepthControlGroup* group, rs2_error** error);

/* Gets new values for Depth Control Group, returns 0 if success */
void rs2_get_depth_control(rs2_device* dev, STDepthControlGroup* group, int mode, rs2_error** error);

/* Sets new values for RSM Group, returns 0 if success */
void rs2_set_rsm(rs2_device* dev, const STRsm* group, rs2_error** error);

/* Gets new values for RSM Group, returns 0 if success */
void rs2_get_rsm(rs2_device* dev, STRsm* group, int mode, rs2_error** error);

/* Sets new values for STRauSupportVectorControl, returns 0 if success */
void rs2_set_rau_support_vector_control(rs2_device* dev, const STRauSupportVectorControl* group, rs2_error** error);

/* Gets new values for STRauSupportVectorControl, returns 0 if success */
void rs2_get_rau_support_vector_control(rs2_device* dev, STRauSupportVectorControl* group, int mode, rs2_error** error);

/* Sets new values for STColorControl, returns 0 if success */
void rs2_set_color_control(rs2_device* dev, const STColorControl* group, rs2_error** error);

/* Gets new values for STColorControl, returns 0 if success */
void rs2_get_color_control(rs2_device* dev, STColorControl* group, int mode, rs2_error** error);

/* Sets new values for STRauColorThresholdsControl, returns 0 if success */
void rs2_set_rau_thresholds_control(rs2_device* dev, const STRauColorThresholdsControl* group, rs2_error** error);

/* Gets new values for STRauColorThresholdsControl, returns 0 if success */
void rs2_get_rau_thresholds_control(rs2_device* dev, STRauColorThresholdsControl* group, int mode, rs2_error** error);

/* Sets new values for STSloColorThresholdsControl, returns 0 if success */
void rs2_set_slo_color_thresholds_control(rs2_device* dev, const STSloColorThresholdsControl* group, rs2_error** error);

/* Gets new values for STRauColorThresholdsControl, returns 0 if success */
void rs2_get_slo_color_thresholds_control(rs2_device* dev, STSloColorThresholdsControl* group, int mode, rs2_error** error);

/* Sets new values for STSloPenaltyControl, returns 0 if success */
void rs2_set_slo_penalty_control(rs2_device* dev, const STSloPenaltyControl* group, rs2_error** error);

/* Gets new values for STSloPenaltyControl, returns 0 if success */
void rs2_get_slo_penalty_control(rs2_device* dev, STSloPenaltyControl* group, int mode, rs2_error** error);

/* Sets new values for STHdad, returns 0 if success */
void rs2_set_hdad(rs2_device* dev, const STHdad* group, rs2_error** error);

/* Gets new values for STHdad, returns 0 if success */
void rs2_get_hdad(rs2_device* dev, STHdad* group, int mode, rs2_error** error);

/* Sets new values for STColorCorrection, returns 0 if success */
void rs2_set_color_correction(rs2_device* dev, const  STColorCorrection* group, rs2_error** error);

/* Gets new values for STColorCorrection, returns 0 if success */
void rs2_get_color_correction(rs2_device* dev, STColorCorrection* group, int mode, rs2_error** error);

/* Sets new values for STDepthTableControl, returns 0 if success */
void rs2_set_depth_table(rs2_device* dev, const  STDepthTableControl* group, rs2_error** error);

/* Gets new values for STDepthTableControl, returns 0 if success */
void rs2_get_depth_table(rs2_device* dev, STDepthTableControl* group, int mode, rs2_error** error);

void rs2_set_ae_control(rs2_device* dev, const  STAEControl* group, rs2_error** error);

void rs2_get_ae_control(rs2_device* dev, STAEControl* group, int mode, rs2_error** error);

void rs2_set_census(rs2_device* dev, const  STCensusRadius* group, rs2_error** error);

void rs2_get_census(rs2_device* dev, STCensusRadius* group, int mode, rs2_error** error);

void rs2_set_amp_factor(rs2_device* dev, const  STAFactor* group, rs2_error** error);

/* Gets new values for STAFactor, returns 0 if success */
void rs2_get_amp_factor(rs2_device* dev, STAFactor* group, int mode, rs2_error** error);

/* Load JSON and apply advanced-mode controls, returns 0 if success */
void rs2_load_json(rs2_device* dev, const void* json_content, unsigned content_size, rs2_error** error);

/* Serialize JSON content, returns 0 if success */
rs2_raw_data_buffer* rs2_serialize_json(rs2_device* dev, rs2_error** error);

#ifdef __cplusplus
}
#endif
#endif
