macro(EXPERIMENT EXPERIMENT_NAME)
    add_executable(${EXPERIMENT_NAME} "${EXPERIMENT_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXPERIMENT_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXPERIMENT_NAME} PROPERTIES
            FOLDER "examples/Cpp/Experimental"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
endmacro(EXPERIMENT)

EXPERIMENT(EvaluateFeatureMatch      ${CMAKE_PROJECT_NAME})
EXPERIMENT(EvaluatePCDMatch          ${CMAKE_PROJECT_NAME})
EXPERIMENT(IntegrateRGBD             ${CMAKE_PROJECT_NAME})
EXPERIMENT(OdometryRGBD              ${CMAKE_PROJECT_NAME})
EXPERIMENT(TrimMeshBasedOnPointCloud ${CMAKE_PROJECT_NAME})
EXPERIMENT(ViewDistances             ${CMAKE_PROJECT_NAME})
EXPERIMENT(ViewPCDMatch              ${CMAKE_PROJECT_NAME})
