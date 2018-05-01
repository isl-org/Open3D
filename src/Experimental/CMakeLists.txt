macro(EXPERIMENT EXPERIMENT_NAME)
    add_executable(${EXPERIMENT_NAME} "${EXPERIMENT_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXPERIMENT_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXPERIMENT_NAME} PROPERTIES
            FOLDER "EXPERIMENTAL"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")

endmacro(EXPERIMENT)

EXPERIMENT(EvaluateFeatureMatch      Core IO Visualization)
EXPERIMENT(EvaluatePCDMatch          Core IO Visualization)
EXPERIMENT(IntegrateRGBD             Core IO Visualization)
EXPERIMENT(OdometryRGBD              Core IO Visualization)
EXPERIMENT(TrimMeshBasedOnPointCloud Core IO Visualization)
EXPERIMENT(ViewDistances             Core IO Visualization)
EXPERIMENT(ViewPCDMatch              Core IO Visualization)
