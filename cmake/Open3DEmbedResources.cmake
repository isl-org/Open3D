# open3d_embed_resources(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <mat1> [<mat2>...]
# )

function(open3d_embed_resources target)
add_custom_command(
    OUTPUT
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/colorMap_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/colorMap_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultGradient_png.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultGradient_png.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLit_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLit_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitSSR_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitSSR_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitTransparency_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitTransparency_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultTexture_png.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultTexture_png.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlit_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlit_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlitTransparency_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlitTransparency_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_value_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_value_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/img_blit_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/img_blit_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/infiniteGroundPlane_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/infiniteGroundPlane_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/normals_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/normals_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pointcloud_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pointcloud_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Bold_ttf.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Bold_ttf.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_BoldItalic_ttf.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_BoldItalic_ttf.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Medium_ttf.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Medium_ttf.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_MediumItalic_ttf.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_MediumItalic_ttf.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/RobotoMono_Medium_ttf.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/RobotoMono_Medium_ttf.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_ibl_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_ibl_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_skybox_ktx.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_skybox_ktx.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/ui_blit_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/ui_blit_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitBackground_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitBackground_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitGradient_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitGradient_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitLine_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitLine_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitPolygonOffset_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitPolygonOffset_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitSolidColor_filamat.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitSolidColor_filamat.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
    DEPENDS
        EmbedResources
        materials
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources &&
        ${CMAKE_COMMAND} -E rm -f ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp &&
        ${CMAKE_COMMAND} -E rm -f ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/brightday_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/brightday_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/colorMap.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/crossroads_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/crossroads_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultGradient.png ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/default_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLitSSR.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLitTransparency.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/default_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultTexture.png ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultUnlit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultUnlitTransparency.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/depth.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/depth_value.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/hall_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/hall_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/img_blit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/infiniteGroundPlane.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/konzerthaus_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/konzerthaus_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/nightlights_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/nightlights_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/normals.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park2_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park2_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pillars_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pillars_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pointcloud.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-Bold.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-BoldItalic.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-Medium.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-MediumItalic.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/RobotoMono-Medium.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/streetlamp_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/streetlamp_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/ui_blit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitBackground.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitGradient.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitLine.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitPolygonOffset.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitSolidColor.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMAND
        EmbedResources -complete ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources 
    COMMENT
        "Generating Resource.cpp"
    VERBATIM
    )

add_custom_target(${target} ALL
    COMMAND
        echo "Embedding resources into the binary"
    DEPENDS 
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/brightday_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/colorMap_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/colorMap_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/crossroads_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/default_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultGradient_png.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultGradient_png.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLit_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLit_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitSSR_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitSSR_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitTransparency_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultLitTransparency_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultTexture_png.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultTexture_png.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlit_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlit_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlitTransparency_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/defaultUnlitTransparency_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_value_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/depth_value_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/hall_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/img_blit_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/img_blit_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/infiniteGroundPlane_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/infiniteGroundPlane_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/konzerthaus_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/nightlights_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/normals_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/normals_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/park2_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pillars_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pointcloud_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/pointcloud_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Bold_ttf.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Bold_ttf.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_BoldItalic_ttf.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_BoldItalic_ttf.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Medium_ttf.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_Medium_ttf.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_MediumItalic_ttf.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/Roboto_MediumItalic_ttf.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/RobotoMono_Medium_ttf.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/RobotoMono_Medium_ttf.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_ibl_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_ibl_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_skybox_ktx.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/streetlamp_skybox_ktx.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/ui_blit_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/ui_blit_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitBackground_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitBackground_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitGradient_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitGradient_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitLine_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitLine_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitPolygonOffset_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitPolygonOffset_filamat.h
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitSolidColor_filamat.cpp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/CompiledResources/unlitSolidColor_filamat.h
)

# set_target_properties(${target} PROPERTIES COMPILED_RESOURCES "${COMPILED_RESOURCES}")

endfunction()
