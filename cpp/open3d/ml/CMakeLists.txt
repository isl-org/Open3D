if (BUILD_TENSORFLOW_OPS AND WIN32)
    message(FATAL_ERROR "Building TensorFlow ops on Windows is currently not supported.")
    # see https://github.com/tensorflow/custom-op/issues/24
endif()

if (BUILD_PADDLE_OPS AND (WIN32 OR APPLE))
    message(FATAL_ERROR "Building Paddle ops on Windows or MacOS is currently not supported.")
endif()

if (BUILD_TENSORFLOW_OPS)
    add_subdirectory(tensorflow)
endif()

if (BUILD_PYTORCH_OPS)
    add_subdirectory(pytorch)
endif()

if (BUILD_PADDLE_OPS)
    add_subdirectory(paddle)
endif()


add_subdirectory(contrib)
