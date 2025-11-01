# xtensa path
if(NOT DEFINED ENV{XTENSA_SDK_PATH})
    message(FATAL_ERROR "xtensa path error")
endif()

# income parameters for try compile
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES AURA_XTENSA_CORE AURA_BUILD_XTENSA AURA_BUILD_XPLORER)

# check xtensa core
if(NOT AURA_XTENSA_CORE MATCHES "vq8")
    message(FATAL_ERROR "xtensa_core can only support vq8")
endif()

# set path
set(XTENSA_TOOLCHAIN_INSTALL    $ENV{XTENSA_SDK_PATH}/XtDevTools/install)
set(XTENSA_SYSTEM               ${XTENSA_TOOLCHAIN_INSTALL}/tools/RI-2022.10-linux/XtensaTools/config)
set(XTENSA_TOOLCHAIN_ROOT       ${XTENSA_TOOLCHAIN_INSTALL}/tools/RI-2022.10-linux)
set(XTENSA_COREFLAG             "--xtensa-core=${AURA_XTENSA_CORE}")

# set tool dir
set(XTENSA_TOOLCHAIN_BIN_PATH   ${XTENSA_TOOLCHAIN_ROOT}/XtensaTools/bin)
set(XTENSA_TOOLCHAIN_TYPE       xt)

# compiler
set(CMAKE_C_COMPILER            ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-clang)
set(CMAKE_CXX_COMPILER          ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-clang++)
set(CMAKE_ASM_COMPILER          ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-as)
set(CMAKE_AR                    ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-ar)
set(CMAKE_RANLIB                ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-ranlib)
set(CMAKE_NM                    ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-nm)
set(CMAKE_OBJDUMP               ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-objdump)
set(CMAKE_OBJCOPY               ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-objcopy)
set(CMAKE_LINK                  ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-ld)
set(CMAKE_STRIP                 ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-strip ${XTENSA_COREFLAG})
set(CMAKE_PKG_LOADLIB           ${XTENSA_TOOLCHAIN_BIN_PATH}/${XTENSA_TOOLCHAIN_TYPE}-pkg-loadlib ${XTENSA_COREFLAG})

# CXX compiler flags
set(XTENSA_COMPILER_FLAGS)
list(APPEND XTENSA_COMPILER_FLAGS
     -DPROC_${AURA_XTENSA_CORE}        # set xtensa proc
     -DCONFIG_${AURA_XTENSA_CORE}      # set xtensa config
     --xtensa-system=${XTENSA_SYSTEM}  # set xtensa system
     ${XTENSA_COREFLAG}                # set xtensa core
     -DFIK_FRAMEWORK                   # open FIK_FRAMEWORK
     -DXI_XV_TILE_COMPATIBILITY        # open XI_XV_TILE_COMPATIBILITY
     -DMAX_NUM_TILES=32                # set MAX_NUM_TILES=32
     -DMAX_NUM_FRAMES=4                # set MAX_NUM_FRAMES=4
     -DMAX_NUM_TILES3D=1               # set MAX_NUM_TILES3D=1
     -DMAX_NUM_FRAMES3D=1              # set MAX_NUM_FRAMES3D=1
     -fno-exceptions)                  # Close exceptions

if(AURA_BUILD_XTENSA)
    list(APPEND XTENSA_COMPILER_FLAGS -fno-builtin)          # Close built-in
endif()

# replace ; with " "
string(REPLACE ";" " " XTENSA_COMPILER_FLAGS "${XTENSA_COMPILER_FLAGS}")

# set CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_ASM_FLAGS
set(CMAKE_C_FLAGS   "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_CXX_FLAGS "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_ASM_FLAGS "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_C_FLAGS   ${XTENSA_COMPILER_FLAGS})
set(CMAKE_CXX_FLAGS ${XTENSA_COMPILER_FLAGS})
set(CMAKE_ASM_FLAGS ${XTENSA_COMPILER_FLAGS})

# set xtensa-core
set(CMAKE_STATIC_LINKER_FLAGS "${XTENSA_COREFLAG}")
set(CMAKE_EXE_LINKER_FLAGS "${XTENSA_COREFLAG}")

# set link executable command
if(AURA_BUILD_XTENSA)
    set(CMAKE_C_LINK_EXECUTABLE   "${CMAKE_C_COMPILER} <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_COMPILER} -stdlib=libc++ <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
elseif(AURA_BUILD_XPLORER)
    set(CMAKE_C_LINK_EXECUTABLE   "${CMAKE_C_COMPILER} -o <TARGET> <OBJECTS> <LINK_LIBRARIES> <LINK_FLAGS>")
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_COMPILER} -stdlib=libc++ -o <TARGET> <OBJECTS> <LINK_LIBRARIES> <LINK_FLAGS>")
else()
    message(FATAL_ERROR "not set AURA_BUILD_XPLORER or AURA_BUILD_XTENSA for xtensa toolchain")
endif()

# set create static library command
set(CMAKE_C_CREATE_STATIC_LIBRARY
    "${CMAKE_AR} -rcs <TARGET> <LINK_FLAGS> <OBJECTS>;${CMAKE_RANLIB} <LINK_FLAGS> <TARGET>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY
    "${CMAKE_AR} -rcs <TARGET> <LINK_FLAGS> <OBJECTS>;${CMAKE_RANLIB} <LINK_FLAGS> <TARGET>")
