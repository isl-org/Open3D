# Find TensorFlow include dir and library
#
# The following variables will be set:
# - Tensorflow_FOUND
# - Tensorflow_INCLUDE_DIR
# - Tensorflow_LIB_DIR
# - Tensorflow_FRAMEWORK_LIB
# - Tensorflow_DEFINITIONS

if(NOT Tensorflow_FOUND)
    # Searching for tensorflow requires the python executable
    find_package(PythonExecutable REQUIRED)

    message(STATUS "Getting TensorFlow properties ...")

    # Get Tensorflow_VERSION
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import tensorflow as tf; print(tf.__version__, end='')"
            OUTPUT_VARIABLE Tensorflow_VERSION)

    # Get Tensorflow_INCLUDE_DIR
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import tensorflow as tf; print(tf.sysconfig.get_include(), end='')"
        OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR)

    # Get Tensorflow_LIB_DIR
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')"
        OUTPUT_VARIABLE Tensorflow_LIB_DIR)

    # Get Tensorflow_FRAMEWORK_LIB
    find_library(
        Tensorflow_FRAMEWORK_LIB
        NAMES tensorflow_framework libtensorflow_framework.so.2
        PATHS "${Tensorflow_LIB_DIR}"
        NO_DEFAULT_PATH
    )

    # Get Tensorflow_DEFINITIONS
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import tensorflow as tf; defs = [ x[2:] for x in tf.sysconfig.get_compile_flags() if x.startswith('-D')]; print(';'.join(defs), end='')"
        OUTPUT_VARIABLE Tensorflow_DEFINITIONS
    )

    # Get TensorFlow CXX11_ABI: 0/1
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import tensorflow; print(tensorflow.__cxx11_abi_flag__, end='')"
        OUTPUT_VARIABLE Tensorflow_CXX11_ABI
    )
endif()

message(STATUS "TensorFlow       version: ${Tensorflow_VERSION}")
message(STATUS "             include dir: ${Tensorflow_INCLUDE_DIR}")
message(STATUS "             library dir: ${Tensorflow_LIB_DIR}")
message(STATUS "           framework lib: ${Tensorflow_FRAMEWORK_LIB}")
message(STATUS "             definitions: ${Tensorflow_DEFINITIONS}")
message(STATUS "           use cxx11 abi: ${Tensorflow_CXX11_ABI}")

# Check if the c++11 ABI is compatible
if((Tensorflow_CXX11_ABI AND (NOT GLIBCXX_USE_CXX11_ABI)) OR
   (NOT Tensorflow_CXX11_ABI AND GLIBCXX_USE_CXX11_ABI))
    message(FATAL_ERROR "TensorFlow and Open3D ABI mismatch: ${Tensorflow_CXX11_ABI} != ${GLIBCXX_USE_CXX11_ABI}")
else()
    message(STATUS "TensorFlow matches Open3D ABI: ${Tensorflow_CXX11_ABI} == ${GLIBCXX_USE_CXX11_ABI}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Tensorflow DEFAULT_MSG Tensorflow_INCLUDE_DIR Tensorflow_LIB_DIR
    Tensorflow_FRAMEWORK_LIB Tensorflow_DEFINITIONS)
