add_executable(ColoredPointRegistration ColoredPointRegistration.cpp)
target_link_libraries(ColoredPointRegistration Visualization IO Core)
set_target_properties(ColoredPointRegistration PROPERTIES
                FOLDER "Tutorial"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Tutorial/Advanced")

add_executable(GlobalRegistration GlobalRegistration.cpp)
target_link_libraries(GlobalRegistration Visualization IO Core)
set_target_properties(GlobalRegistration PROPERTIES
                FOLDER "Tutorial"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Tutorial/Advanced")

add_executable(MultiwayRegistration MultiwayRegistration.cpp)
target_link_libraries(MultiwayRegistration Visualization IO Core)
set_target_properties(MultiwayRegistration PROPERTIES
                FOLDER "Tutorial"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Tutorial/Advanced")

