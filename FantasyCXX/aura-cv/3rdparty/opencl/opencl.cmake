include(ExternalProject)

set(3RDPARTY_OPENCL_INSTALL_DIRS "${PROJECT_BINARY_DIR}/3rdparty/opencl/OpenCL-Headers")
set(3RDPARTY_OPENCL_INC_DIRS     "${3RDPARTY_OPENCL_INSTALL_DIRS}/src/OpenCL-Headers")

include_directories(${3RDPARTY_OPENCL_INC_DIRS})

# GIT_TAG v2021.06.30 will affect the header file API declaration
# Please do not modify it at will
externalproject_add(OpenCL-Headers
    GIT_REPOSITORY          https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG                 v2021.06.30
    GIT_SHALLOW             true
    PREFIX                  "${3RDPARTY_OPENCL_INSTALL_DIRS}"
    DOWNLOAD_NAME           "OpenCL-Headers"
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    INSTALL_COMMAND         ""
    UPDATE_COMMAND          ""
    TEST_COMMAND            ""
)

set(3RDPARTY_OPENCL_HPP_INSTALL_DIRS "${PROJECT_BINARY_DIR}/3rdparty/opencl/OpenCL-Headers-Hpp")
set(3RDPARTY_OPENCL_HPP_INC_DIRS     "${3RDPARTY_OPENCL_HPP_INSTALL_DIRS}/src/OpenCL-Headers-Hpp/include")

include_directories(${3RDPARTY_OPENCL_HPP_INC_DIRS})

externalproject_add(OpenCL-Headers-Hpp
    GIT_REPOSITORY          https://github.com/KhronosGroup/OpenCL-CLHPP.git
    GIT_TAG                 v2.0.15
    GIT_SHALLOW             true
    PREFIX                  "${3RDPARTY_OPENCL_HPP_INSTALL_DIRS}"
    DOWNLOAD_NAME           "OpenCL-Headers-Hpp"
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    INSTALL_COMMAND         ""
    UPDATE_COMMAND          ""
    TEST_COMMAND            ""
)

install(DIRECTORY "${3RDPARTY_OPENCL_INC_DIRS}/CL"     DESTINATION include/3rdparty/opencl)
install(DIRECTORY "${3RDPARTY_OPENCL_HPP_INC_DIRS}/CL" DESTINATION include/3rdparty/opencl)