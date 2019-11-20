/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_processing_gl.h
* \brief
* Exposes RealSense processing-block functionality for GPU for C compilers
*/

#ifndef LIBREALSENSE_RS2_PROCESSING_GL_H
#define LIBREALSENSE_RS2_PROCESSING_GL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "librealsense2/rs.h"

/*
* GL-specific extension types.
* Used similar to regular RS extensions
*/
typedef enum rs2_gl_extension
{
    RS2_GL_EXTENSION_VIDEO_FRAME,
    RS2_GL_EXTENSION_COUNT
} rs2_gl_extension;
const char* rs2_gl_extension_to_string(rs2_extension type);

/*
* In addition to "options", GL processing blocks
* expose new type of multidementional options each holding a 4x4 matrix
*/
typedef enum rs2_gl_matrix_type
{
    RS2_GL_MATRIX_TRANSFORMATION,
    RS2_GL_MATRIX_PROJECTION,
    RS2_GL_MATRIX_CAMERA,
    RS2_GL_MATRIX_COUNT
} rs2_gl_matrix_type;
const char* rs2_gl_matrix_type_to_string(rs2_gl_matrix_type type);

/* Forward-declare GLFW types */
typedef struct GLFWwindow GLFWwindow;
typedef struct GLFWmonitor GLFWmonitor;
typedef int(*glfwInitFun)(void);
typedef void(*glfwWindowHintFun)(int, int);
typedef GLFWwindow*(*glfwCreateWindowFun)(int, int, const char*, GLFWmonitor*, GLFWwindow*);
typedef void(*glfwDestroyWindowFun)(GLFWwindow*);
typedef void(*glfwMakeContextCurrentFun)(GLFWwindow*);
typedef GLFWwindow*(*glfwGetCurrentContextFun)(void);
typedef void(*glfwSwapIntervalFun)(int);
typedef void(*GLFWglproc)(void);
typedef GLFWglproc(*glfwGetProcAddressFun)(const char*);

/* To avoid GLFW version mismatch (mainly can affect Windows), the application passes pointers
* to its version of GLFW, via glfw_bindings struct
*/
struct glfw_binding
{
    glfwInitFun glfwInit;
    glfwWindowHintFun glfwWindowHint;
    glfwCreateWindowFun glfwCreateWindow;
    glfwDestroyWindowFun glfwDestroyWindow ;
    glfwMakeContextCurrentFun glfwMakeContextCurrent;
    glfwGetCurrentContextFun glfwGetCurrentContext;
    glfwSwapIntervalFun glfwSwapInterval;
    glfwGetProcAddressFun glfwGetProcAddress;
};

/**
* Creates a processing block that can efficiently convert YUY image format to RGB variants
* This is specifically useful for rendering the RGB frame to the screen (since the output is ready for rendering on the GPU)
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_gl_create_yuy_decoder(int api_version, rs2_error** error);

/**
* Sets new value to one of processing blocks matrices
* \param[in] block      Processing block object
* \param[in] type       Matrix type
* \param[in] m4x4       Pointer to 16 floating point values encoding 4x4 matrix
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_set_matrix(rs2_processing_block* block, rs2_gl_matrix_type type, float* m4x4, rs2_error** error);

/**
* Query if frame is extendable to one of the GL-specific extensions
* \param[in] f          Frame pointer
* \param[in] extension_type Extension type
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \returns          1 if true, 0 otherwise
*/
int rs2_gl_is_frame_extendable_to(const rs2_frame* f, rs2_gl_extension extension_type, rs2_error** error);

/**
* Assuming frame is extendable to RS2_GL_EXTENSION_VIDEO_FRAME,
* this method will fetch one of frames texture IDs
* Each GPU frame can hold one or more OpenGL textures
* \param[in] f          Frame pointer
* \param[in] id         Index of texture within the frame
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \returns              OpenGL texture
*/
unsigned int rs2_gl_frame_get_texture_id(const rs2_frame* f, unsigned int id, rs2_error** error);

/**
 * Camera renderer is a rendering block (meaning it has to be called within the main OpenGL rendering context)
 * that will render the camera model of the frame provided to it
 * \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
rs2_processing_block* rs2_gl_create_camera_renderer(int api_version, rs2_error** error);

/**
 * Pointcloud renderer will render texture pointcloud as either points
 * or connected polygons
 * \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
rs2_processing_block* rs2_gl_create_pointcloud_renderer(int api_version, rs2_error** error);

/**
* Creates Point-Cloud processing block. This block accepts depth frames and outputs Points frames
* In addition, given non-depth frame, the block will align texture coordinate to the non-depth stream
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_gl_create_pointcloud(int api_version, rs2_error** error);

/**
* Creates Upload processing block
* This object can explicitly copy frames from the CPU to the GPU
* This allows pre-emptively upload frame to the GPU on a background thread
* To be used directly in future GPU processing
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_gl_create_uploader(int api_version, rs2_error** error);

/**
* Creates Colorizer processing block
* This block has similar capabilities as the regular librealsense colorizer
* It is capable of applying depth to RGB conversion using various color schemes
* and optional histogram equalization
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_gl_create_colorizer(int api_version, rs2_error** error);

/**
* Creates Align processing block
* This block has similar capabilities as the regular librealsense align
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[in] align_to    stream type to be used as the target of frameset alignment
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_gl_create_align(int api_version, rs2_stream to, rs2_error** error);

/** 
 * Initialize rendering pipeline. This function must be called before executing
 * any of the rendering blocks.
 * Rendering blocks do not handle threading, and assume all calls (including init / shutdown)
 * Until initialized, rendering blocks will do nothing (function as bypass filters)
 * are serialized and coming from a single rendering thread
 * \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
 * \param[in] use_glsl  On modern GPUs you can get slightly better performance using GLSL
 *                      However, this assumes the current rendering context is v3+
 *                      Setting use_glsl to false will use legacy OpenGL calls
 *                      This in turn assumes the rendering context is either version < 3, or is a compatibility context
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_init_rendering(int api_version, int use_glsl, rs2_error** error);

/**
* In order to use GL rendering with GLFW application
* the user need to initialize rendering by passing GLFW binding information
* C++ wrapper will automatically generate and pass this data
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[in] bindings  Pointers to GLFW methods that will be used by the library
* \param[in] use_glsl  Use GLSL shaders for rendering
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_init_rendering_glfw(int api_version, glfw_binding bindings, int use_glsl, rs2_error** error);

/**
 * Initialize processing pipeline. This function allows GL processing blocks to 
 * run on the GPU. Until initialized, all GL processing blocks will fall back
 * to their CPU versions. 
 * When initializing using this method, texture sharing is not available.
 * \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_gl_init_processing(int api_version, int use_glsl, rs2_error** error);

/**
* In order to share GL processing results with GLFW rendering application
* the user need to initialize rendering by passing GLFW binding information
* C++ wrapper will automatically generate and pass this data
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[in] share_with  Pointer to GLFW window object. This window will be able to use texture IDs provided inside GPU-frames generated by the library
* \param[in] bindings  Pointers to GLFW methods that will be used by the library
* \param[in] use_glsl  Use GLSL shaders for processing
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_init_processing_glfw(int api_version, GLFWwindow* share_with, 
                                 glfw_binding bindings, int use_glsl, rs2_error** error);

/**
* Shutdown librealsense rendering. This will disable all rendering blocks
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_shutdown_rendering(int api_version, rs2_error** error);

/**
* Shutdown librealsense processing. This will switch all GL processing blocks into CPU only mode
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_gl_shutdown_processing(int api_version, rs2_error** error);

#ifdef __cplusplus
}
#endif
#endif
