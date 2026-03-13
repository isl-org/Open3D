if (BUILD_TENSORFLOW_OPS AND WIN32)
    message(FATAL_ERROR "Building TensorFlow ops on Windows is currently not supported.")
    # see https://github.com/tensorflow/custom-op/issues/24
endif()

if (BUILD_TENSORFLOW_OPS)
    add_subdirectory(tensorflow)
endif()

if (BUILD_PYTORCH_OPS)
    add_subdirectory(pytorch)
endif()

target_sources(Open3D PRIVATE
    Model.cpp
)

add_subdirectory(contrib)
