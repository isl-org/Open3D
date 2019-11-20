# librealsense2 API

# File-System Structure

Under `librealsense2` folder you will find two subfolers:
* [h](./librealsense2/h) - Contains headers for the C language
* [hpp](./librealsense2/hpp) - Contains headers for the C++ language, depends on C headers

In addition, you can include [<librealsense2/rs.h>](./librealsense2/rs.h) and [<librealsense2/rs.hpp>](./librealsense2/rs.hpp) to get most of SDK functionality in C and C++ respectively. 

[<librealsense2/rs_advanced_mode.h>](./librealsense2/rs_advanced_mode.h) and [<librealsense2/rs_advanced_mode.hpp>](./librealsense2/rs_advanced_mode.hpp) can be included to get the extra [Advanced Mode](../doc/rs400_advanced_mode.md) functionality (in C and C++ respectively).

[<librealsense2/rsutil.h>](./librealsense2/rsutil.h) contains mathematical helper functions for projection of 2D points into 3D space and back. 

# Files and Classes

> For full up-to-date class documentation see [Doxygen class-list](http://intelrealsense.github.io/librealsense/doxygen/annotated.html)

## [rs_types.hpp](librealsense2/hpp/rs_types.hpp):

This file contains common functionality like the exception types. See [Error Handling](..doc/error_handling.md)

|Class|Description|
|-----|-----------|
|[error](librealsense2/hpp/rs_types.hpp#L76)| Common base-class for all exceptions thrown from the SDK |
|[recoverable_error](librealsense2/hpp/rs_types.hpp#L113)| Base class for all recorverable (in software) errors |
|[unrecoverable_error](librealsense2/hpp/rs_types.hpp#L114)| Base class for non-recoverable errors |
|[camera_disconnected_error](librealsense2/hpp/rs_types.hpp#L115)| Camera Disconnected error |
|[backend_error](librealsense2/hpp/rs_types.hpp#L116)| Error from the underlying OS-specific driver |
|[device_in_recovery_mode_error](librealsense2/hpp/rs_types.hpp#L117)| Device requires firmware update |
|[invalid_value_error](librealsense2/hpp/rs_types.hpp#L118)| Invalid input was passed to a function |
|[wrong_api_call_sequence_error](librealsense2/hpp/rs_types.hpp#L119)| Methods were called in wrong order |
|[not_implemented_error](librealsense2/hpp/rs_types.hpp#L120)| Functionality is not available error |

## [rs_context.hpp](librealsense2/hpp/rs_context.hpp):
Context can be used to enumerate and iterate over connected device, and also get notifications of device events. 

|Class|Description|
|-----|-----------|
|[context](librealsense2/hpp/rs_context.hpp#L81)| Context serve as a factory for all SDK devices |
|[device_hub](librealsense2/hpp/rs_context.hpp#L185)| Helper class that simplifies handling device change events |
|[event_information](librealsense2/hpp/rs_context.hpp#L16)| Information about device change |

## [rs_device.hpp](librealsense2/hpp/rs_device.hpp):
This file defines the concept of RealSense device. 

|Class|Description|
|-----|-----------|
|[device](librealsense2/hpp/rs_device.hpp#L109)| Encapsulates Intel RealSense device |
|[device_list](librealsense2/hpp/rs_device.hpp#L186)| Wrapper around list of devices |
|[debug_protocol](librealsense2/hpp/rs_device.hpp#L151)| Debug-extension that allows to send data directly to the firmware |

## [rs_processing.hpp](librealsense2/hpp/rs_processing.hpp):
The SDK offers many types of post-processing in form of the Processing Blocks. These primitives can be pipelined using queues and configured using options. 

|Class|Description|
|-----|-----------|
|[syncer](librealsense2/hpp/rs_processing.hpp#L260)| Processing block that accepts frames one-by-one and outputs coherent framesets|
|[frame_queue](librealsense2/hpp/rs_processing.hpp#L136)| Basic primitive abstracting a concurrent queue of frames|
|[processing_block](librealsense2/hpp/rs_processing.hpp#L105)| Base class for frame processing functions |
|[pointcloud](librealsense2/hpp/rs_processing.hpp#L196)| Processing block that accepts depth and texture frames and outputs 3D points with texture coordinates |
|[asynchronous_syncer](librealsense2/hpp/rs_processing.hpp#L232)| Non-blocking version of the syncher processing block |
|[align](librealsense2/hpp/rs_processing.hpp#L316)| Processing block that accepts frames from different viewports and generates new synthetic frames from a single given viewport |
|[colorizer](librealsense2/hpp/rs_processing.hpp#L356)| Processing block that accepts depth frames in Z16 format and outputs depth frames in RGB format, applying some depth coloring |
|[decimation_filter](librealsense2/hpp/rs_processing.hpp#L391)| Processing block that intelligently reduces the resolution of a depth frame |
|[temporal_filter](librealsense2/hpp/rs_processing.hpp#L428)| Processing block that filters depth data by looking into previous frames |
|[spatial_filter](librealsense2/hpp/rs_processing.hpp#L465)| Processing block that applies edge-preserving smoothing of depth data|

## [rs_sensor.hpp](librealsense2/hpp/rs_sensor.hpp):
RealSense devices contain sensors. Sensors are units of streaming, each sensor can be controlled individually. 

|Class|Description|
|-----|-----------|
|[sensor](librealsense2/hpp/rs_sensor.hpp#L392)| Base class for all supported sensors|
|[roi_sensor](librealsense2/hpp/rs_sensor.hpp#L446)| Sensor that has the ability to focus on a given ROI|
|[notification](librealsense2/hpp/rs_sensor.hpp#L15)| Asynchronious message that can be passed to the application |
|[depth_sensor](librealsense2/hpp/rs_sensor.hpp#L479)| Sensor that can provide depth frames |

## [rs_frame.hpp](librealsense2/hpp/rs_frame.hpp):
The output of sensors are frames. There are different kinds of frames, not all video frames. 

|Class|Description|
|-----|-----------|
|[frame](librealsense2/hpp/rs_frame.hpp#L157)| Base class for all frames |
|[points](librealsense2/hpp/rs_frame.hpp#L423)| Set of 3D points with texture coordinates |
|[video_frame](librealsense2/hpp/rs_frame.hpp#L348)| 2D image (with width, height and bpp) |
|[depth_frame](librealsense2/hpp/rs_frame.hpp#L480)| Depth frame |
|[frameset](librealsense2/hpp/rs_frame.hpp#L502)| A set of frames |
|[stream_profile](librealsense2/hpp/rs_frame.hpp#L24)| Base class for sensor configuration |
|[video_stream_profile](librealsense2/hpp/rs_frame.hpp#L113)| Video stream configuration |

## [rs_pipeline.hpp](librealsense2/hpp/rs_pipeline.hpp):
Pipeline is a high-level primitive combining several sensors and processing steps into one simple API. 

|Class|Description|
|-----|-----------|
|[pipeline](librealsense2/hpp/rs_pipeline.hpp#L345)| High level data-processor combining several sensors, queues and processing blocks |
|[pipeline_profile](librealsense2/hpp/rs_pipeline.hpp#L22)| Selected pipeline configuration |
|[config](librealsense2/hpp/rs_pipeline.hpp#L128)| Desired pipeline configuration |

## [rs_record_playback.hpp](librealsense2/hpp/rs_record_playback.hpp):
This file adds playback and record capability using ROS-bag files. See [src/media](src/media)

|Class|Description|
|-----|-----------|
|[playback](librealsense2/hpp/rs_record_playback.hpp#L30)| Device that mimiks live device from given input file |
|[recorder](librealsense2/hpp/rs_record_playback.hpp#L206)| Device that records live device into an output file |

## [rs_internal.hpp](librealsense2\hpp\rs_internal.hpp):
This file is not intented to be included by SDK users, but rather is used in SDK unit-tests. It allows to record everything that will happen in a specific test into a file and later use that file for dependency injection. 

|Class|Description|
|-----|-----------|
|[recording_context](librealsense2\hpp\rs_internal.hpp#L19)| Context that records all backend activity to file |
|[mock_context](librealsense2\hpp\rs_internal.hpp#L41)| Context that replays all activity from file |
