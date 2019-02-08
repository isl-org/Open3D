/*
  This file contains docstrings for the Python bindings.
  Do not edit! These were automatically extracted by mkdoc.py
 */

#define __EXPAND(x) x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define __VA_SIZE(...) __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b) a##b
#define __CAT2(a, b) __CAT1(a, b)
#define __DOC1(n1) __doc_##n1
#define __DOC2(n1, n2) __doc_##n1##_##n2
#define __DOC3(n1, n2, n3) __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4) __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5) __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6) \
    __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7) \
    __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...) \
    __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static const char *__doc_open3d_ClipIntensityImage =
        R"doc(Function to clipping pixel intensities min is lower bound max is upper
bound)doc";

static const char *__doc_open3d_ConvertDepthToFloatImage = R"doc()doc";

static const char *__doc_open3d_CreateDepthBoundaryMask =
        R"doc(Function to create a depthmap boundary mask from depth image)doc";

static const char *__doc_open3d_CreateDepthToCameraDistanceMultiplierFloatImage =
        R"doc(Factory function to create a float image composed of multipliers that
convert depth values into camera distances (ImageFactory.cpp) The
multiplier function M(u,v) is defined as: M(u, v) = sqrt(1 + ((u - cx)
/ fx) ^ 2 + ((v - cy) / fy) ^ 2) This function is used as a convenient
function for performance optimization in volumetric integration (see
Core/Integration/TSDFVolume.h).)doc";

static const char *__doc_open3d_CreateFloatImageFromImage =
        R"doc(Return a gray scaled float type image.)doc";

static const char *__doc_open3d_CreateImageFromFloatImage =
        R"doc(Function to change data types of image crafted for specific usage such
as single channel float image -> 8-bit RGB or 16-bit depth image)doc";

static const char *__doc_open3d_CreateImagePyramid =
        R"doc(Function to create image pyramid - overwrite)doc";

static const char *__doc_open3d_DilateImage =
        R"doc(Function to dilate 8bit mask map)doc";

static const char *__doc_open3d_DownsampleImage =
        R"doc(Function to 2x image downsample using simple 2x2 averaging)doc";

static const char *__doc_open3d_FilterHorizontalImage = R"doc()doc";

static const char *__doc_open3d_FilterImage =
        R"doc(Function to filter image with pre-defined filtering type)doc";

static const char *__doc_open3d_FilterImage_2 =
        R"doc(Function to filter image with arbitrary dx, dy separable filters)doc";

static const char *__doc_open3d_FilterImagePyramid =
        R"doc(Function to filter image pyramid)doc";

static const char *__doc_open3d_FlipImage = R"doc()doc";

static const char *__doc_open3d_Image = R"doc()doc";

static const char *__doc_open3d_Image_AllocateDataBuffer = R"doc()doc";

static const char *__doc_open3d_Image_BytesPerLine = R"doc()doc";

static const char *__doc_open3d_Image_Clear = R"doc()doc";

static const char *__doc_open3d_Image_ColorToIntensityConversionType =
        R"doc()doc";

static const char *__doc_open3d_Image_ColorToIntensityConversionType_Equal =
        R"doc()doc";

static const char *__doc_open3d_Image_ColorToIntensityConversionType_Weighted =
        R"doc()doc";

static const char *__doc_open3d_Image_FilterType = R"doc()doc";

static const char *__doc_open3d_Image_FilterType_Gaussian3 = R"doc()doc";

static const char *__doc_open3d_Image_FilterType_Gaussian5 = R"doc()doc";

static const char *__doc_open3d_Image_FilterType_Gaussian7 = R"doc(Test)doc";

static const char *__doc_open3d_Image_FilterType_Sobel3Dx = R"doc()doc";

static const char *__doc_open3d_Image_FilterType_Sobel3Dy = R"doc()doc";

static const char *__doc_open3d_Image_FloatValueAt =
        R"doc(Function to access the bilinear interpolated float value of a (single-
channel) float image)doc";

static const char *__doc_open3d_Image_GetMaxBound = R"doc()doc";

static const char *__doc_open3d_Image_GetMinBound = R"doc()doc";

static const char *__doc_open3d_Image_HasData = R"doc()doc";

static const char *__doc_open3d_Image_Image = R"doc()doc";

static const char *__doc_open3d_Image_IsEmpty = R"doc()doc";

static const char *__doc_open3d_Image_PrepareImage = R"doc()doc";

static const char *__doc_open3d_Image_TestImageBoundary = R"doc()doc";

static const char *__doc_open3d_Image_bytes_per_channel = R"doc()doc";

static const char *__doc_open3d_Image_data = R"doc()doc";

static const char *__doc_open3d_Image_height = R"doc()doc";

static const char *__doc_open3d_Image_num_of_channels = R"doc()doc";

static const char *__doc_open3d_Image_width = R"doc()doc";

static const char *__doc_open3d_LinearTransformImage =
        R"doc(Function to linearly transform pixel intensities image_new = scale *
image + offset)doc";

static const char *__doc_open3d_PinholeCameraIntrinsic = R"doc()doc";

static const char *__doc_open3d_PointerAt =
        R"doc(Function to access the raw data of a single-channel Image)doc";

static const char *__doc_open3d_PointerAt_2 =
        R"doc(Function to access the raw data of a multi-channel Image)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
