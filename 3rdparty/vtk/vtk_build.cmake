include(ExternalProject)

if(WIN32)
    set(VTK_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(VTK_LIB_SUFFIX "")
endif()

set(VTK_VERSION 9.1)

set(VTK_LIBRARIES
    vtkFiltersGeneral-${VTK_VERSION}${VTK_LIB_SUFFIX}
    # vtkCommonComputationalGeometry-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonExecutionModel-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonDataModel-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMath-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMisc-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonSystem-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonTransforms-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkkissfft-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkpugixml-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtksys-${VTK_VERSION}${VTK_LIB_SUFFIX}
)

foreach(item IN LISTS VTK_LIBRARIES)
    list(APPEND VTK_BUILD_BYPRODUCTS <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${item}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()


if(BUILD_VTK_FROM_SOURCE)

    ExternalProject_Add(
        ext_vtk
        PREFIX vtk
        URL https://www.vtk.org/files/release/${VTK_VERSION}/VTK-${VTK_VERSION}.0.tar.gz
        URL_HASH SHA256=8fed42f4f8f1eb8083107b68eaa9ad71da07110161a3116ad807f43e5ca5ce96
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vtk"
        # do not update
        UPDATE_COMMAND ""
        CMAKE_ARGS
            ${ExternalProject_CMAKE_ARGS_hidden}
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DVTK_GROUP_ENABLE_Imaging=NO
            -DVTK_GROUP_ENABLE_MPI=NO
            -DVTK_GROUP_ENABLE_Qt=NO
            -DVTK_GROUP_ENABLE_Rendering=NO
            -DVTK_GROUP_ENABLE_StandAlone=NO
            -DVTK_GROUP_ENABLE_Views=NO
            -DVTK_GROUP_ENABLE_Web=NO
            -DVTK_ENABLE_LOGGING=OFF
            -DVTK_ENABLE_REMOTE_MODULES=OFF
            -DVTK_ENABLE_WRAPPING=OFF
            -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmDataModel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmFilters=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ChartsCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonArchive=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonColor=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonComputationalGeometry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonDataModel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonExecutionModel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonMath=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonMisc=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonSystem=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_CommonTransforms=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_DICOMParser=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_DomainsChemistry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_DomainsChemistryOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_DomainsMicroscopy=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_DomainsParallelChemistry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersAMR=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersExtraction=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersFlowPaths=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersGeneric=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersGeometry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersHybrid=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersHyperTree=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersImaging=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersModeling=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersOpenTURNS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelFlowPaths=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelGeometry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelImaging=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelMPI=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelStatistics=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersParallelVerdict=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersPoints=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersProgrammable=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersReebGraph=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersSMP=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersSelection=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersSources=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersStatistics=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersTexture=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersTopology=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_FiltersVerdict=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_GUISupportQt=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_GUISupportQtSQL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_GeovisCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_GeovisGDAL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOADIOS2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOAMR=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOAsynchronous=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOCGNSReader=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOCONVERGECFD=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOChemistry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOCityGML=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOEnSight=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOExodus=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOExport=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOExportGL2PS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOExportPDF=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOFFMPEG=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOFides=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOGDAL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOGeoJSON=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOGeometry=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOH5Rage=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOH5part=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOHDF=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOIOSS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOImage=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOImport=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOInfovis=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOLAS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOLSDyna=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOLegacy=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMINC=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMPIImage=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMPIParallel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMotionFX=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMovie=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOMySQL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IONetCDF=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOODBC=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOOMF=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOOggTheora=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOOpenVDB=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOPDAL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOPIO=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOPLY=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallelExodus=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallelLSDyna=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallelNetCDF=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallelXML=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOParallelXdmf3=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOPostgreSQL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOSQL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOSegY=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOTRUCHAS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOTecplotTable=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOVPIC=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOVeraOut=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOVideo=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOXML=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOXMLParser=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOXdmf2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_IOXdmf3=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingColor=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingFourier=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingGeneral=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingHybrid=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingMath=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingMorphological=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingSources=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingStatistics=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ImagingStencil=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InfovisBoost=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InfovisBoostGraphAlgorithms=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InfovisCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InfovisLayout=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InteractionImage=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InteractionStyle=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_InteractionWidgets=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_MomentInvariants=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ParallelCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ParallelDIY=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ParallelMPI=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_PoissonReconstruction=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_Powercrust=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_PythonInterpreter=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingContext2D=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingExternal=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingFFMPEGOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingFreeType=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingFreeTypeFontConfig=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingGL2PSOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingImage=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingLICOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingLOD=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingLabel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingLookingGlass=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingMatplotlib=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingOpenVR=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingParallel=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingParallelLIC=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingQt=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingRayTracing=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingSceneGraph=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingUI=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingVR=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingVolume=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingVolumeAMR=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_RenderingVtkJS=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_SignedTensor=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_SplineDrivenImageSlicer=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_TestingCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_TestingGenericBridge=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_TestingIOSQL=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_TestingRendering=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_UtilitiesBenchmarks=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ViewsContext2D=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ViewsCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ViewsInfovis=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ViewsQt=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_WebCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_WebGLExporter=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_WrappingPythonCore=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_WrappingTools=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_cgns=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_cli11=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_diy2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_doubleconversion=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_eigen=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_exodusII=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_expat=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_exprtk=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_fides=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_fmt=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_freetype=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_gl2ps=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_glew=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_h5part=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_hdf5=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ioss=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_jpeg=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_jsoncpp=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_kissfft=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_kwiml=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_libharu=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_libproj=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_libxml2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_loguru=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_lz4=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_lzma=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_metaio=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_netcdf=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_octree=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_ogg=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_opengl=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_pegtl=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_png=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_pugixml=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_sqlite=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_theora=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_tiff=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_utf8=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_verdict=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_vpic=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_vtkDICOM=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_vtkm=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_vtksys=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_xdmf2=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_xdmf3=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_zfp=DONT_WANT
            -DVTK_MODULE_ENABLE_VTK_zlib=DONT_WANT
        BUILD_BYPRODUCTS
            ${VTK_BUILD_BYPRODUCTS}
    )

    ExternalProject_Get_Property(ext_vtk INSTALL_DIR)
    set(VTK_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
    set(VTK_INCLUDE_DIRS "${INSTALL_DIR}/include/vtk-${VTK_VERSION}/")

else() #### download prebuilt vtk

    if(LINUX_AARCH64)
        message(FATAL "No precompiled vtk for platform. Enable BUILD_VTK_FROM_SOURCE")
    elseif(APPLE_AARCH64)
        message(FATAL "No precompiled vtk for platform. Enable BUILD_VTK_FROM_SOURCE")
    elseif(APPLE)
        set(VTK_URL
            https://github.com/isl-org/open3d_downloads/releases/download/vtk${VTK_VERSION}/vtk_${VTK_VERSION}_macos_10.15.tar.gz
        )
        set(VTK_SHA256 eb81112dc62ea7ab39ca11d899399247311867db4b53a367f3efb4f5535e582b)
    elseif(UNIX)
        set(VTK_URL
            https://github.com/isl-org/open3d_downloads/releases/download/vtk${VTK_VERSION}/vtk_${VTK_VERSION}_linux_x86_64.tar.gz
        )
        set(VTK_SHA256 ab388f476e202aa0c2d1349c2047cf680d95253c044792830b8b80a0285c4afb)
    elseif(WIN32)
        if (STATIC_WINDOWS_RUNTIME)
            set(VTK_URL
                https://github.com/isl-org/open3d_downloads/releases/download/vtk${VTK_VERSION}/vtk_${VTK_VERSION}_win_staticrt.tar.gz
            )
            set(VTK_SHA256 5c445a3015bc48ce74381306e7f35f4e3f330f987bde73dfb95d4c0486143b85)
        else()
            set(VTK_URL
                https://github.com/isl-org/open3d_downloads/releases/download/vtk${VTK_VERSION}/vtk_${VTK_VERSION}_win.tar.gz
            )
            set(VTK_SHA256 f24d96b6f45a15c9dada87443f892328cbf9d8017d756b6e8c5f83e9f34ec99d)
        endif()
    else()
        message(FATAL "Unsupported platform")
    endif()

    ExternalProject_Add(
        ext_vtk
        PREFIX vtk
        URL ${VTK_URL}
        URL_HASH SHA256=${VTK_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vtk"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ""
    )

    ExternalProject_Get_Property(ext_vtk SOURCE_DIR)
    set(VTK_LIB_DIR "${SOURCE_DIR}/lib")
    set(VTK_INCLUDE_DIRS "${SOURCE_DIR}/include/vtk-${VTK_VERSION}/")

endif() # BUILD_VTK_FROM_SOURCE
