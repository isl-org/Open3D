# Find Tensorflow include dir and library
#
# The following variables will be set
#   Tensorflow_FOUND
#   Tensorflow_INCLUDE_DIR
#   Tensorflow_LIB_DIR
#   Tensorflow_FRAMEWORK_LIB
#   Tensorflow_DEFINITIONS
#
if (NOT Tensorflow_FOUND)
    # searching for tensorflow requires the python executable
    find_package( PythonExecutable REQUIRED )

    message( STATUS "Getting Tensorflow properties ..." )

    execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" 
                     OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR )
    execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" 
                     OUTPUT_VARIABLE Tensorflow_LIB_DIR )
    find_library( Tensorflow_FRAMEWORK_LIB 
        NAMES tensorflow_framework libtensorflow_framework.so.2 
        PATHS "${Tensorflow_LIB_DIR}" 
        NO_DEFAULT_PATH )
    execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; defs = [ x[2:] for x in tf.sysconfig.get_compile_flags() if x.startswith('-D')]; print(';'.join(defs), end='')" 
                     OUTPUT_VARIABLE Tensorflow_DEFINITIONS )
endif()

message( STATUS "Tensorflow   include dir: ${Tensorflow_INCLUDE_DIR}" )
message( STATUS "             library dir: ${Tensorflow_LIB_DIR}" )
message( STATUS "           framework lib: ${Tensorflow_FRAMEWORK_LIB}" )
message( STATUS "             definitions: ${Tensorflow_DEFINITIONS}" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tensorflow DEFAULT_MSG Tensorflow_INCLUDE_DIR Tensorflow_LIB_DIR Tensorflow_FRAMEWORK_LIB Tensorflow_DEFINITIONS )

