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
    if (NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python 3 not found in top level file")
    endif()

    message(STATUS "Getting TensorFlow properties ...")

    set(Tensorflow_FETCH_PROPERTIES
        "import tensorflow as tf"
        "print(tf.__version__, end=';')"
        "print(tf.sysconfig.get_include(), end=';')"
        "print(tf.sysconfig.get_lib(), end=';')"
        "defs = [ x[2:] for x in tf.sysconfig.get_compile_flags() if x.startswith('-D')]; print('::'.join(defs), end=';')"
        "print(tf.__cxx11_abi_flag__, end=';')"
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c" "${Tensorflow_FETCH_PROPERTIES}"
        OUTPUT_VARIABLE Tensorflow_PROPERTIES
    )

    list(GET Tensorflow_PROPERTIES 0 Tensorflow_VERSION)
    list(GET Tensorflow_PROPERTIES 1 Tensorflow_INCLUDE_DIR)
    list(GET Tensorflow_PROPERTIES 2 Tensorflow_LIB_DIR)
    list(GET Tensorflow_PROPERTIES 3 Tensorflow_DEFINITIONS)
    list(GET Tensorflow_PROPERTIES 4 Tensorflow_CXX11_ABI)

    unset(Tensorflow_FETCH_PROPERTIES)
    unset(Tensorflow_PROPERTIES)

    # Decode definitions into a proper list
    string(REGEX REPLACE "::" ";" Tensorflow_DEFINITIONS ${Tensorflow_DEFINITIONS})

    # Get Tensorflow_FRAMEWORK_LIB
    find_library(
        Tensorflow_FRAMEWORK_LIB
        NAMES tensorflow_framework libtensorflow_framework.so.2
        PATHS "${Tensorflow_LIB_DIR}"
        NO_DEFAULT_PATH
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
