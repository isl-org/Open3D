# build
file(GLOB_RECURSE LIB_SOURCE_FILES "*.cpp")

# create object library
add_library(GUI OBJECT ${LIB_SOURCE_FILES})
set_property(TARGET GUI PROPERTY CXX_STANDARD 14)
#target_include_directories(GUI SYSTEM INTERFACE ${3RDPARTY_INSTALL_DIR}/include/filament})
#message("[debug] SDL2_INCLUDE_DIRS: ${SDL2_INCLUDE_DIRS}")
#target_include_directories(GUI SYSTEM INTERFACE ${SDL2_INCLUDE_DIRS})
ShowAndAbortOnWarning(GUI)

# Enforce 3rd party dependencies
add_dependencies(GUI build_all_3rd_party_libs)
