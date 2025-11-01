# hexagon sdk path
if(NOT DEFINED ENV{HEXAGON_SDK_PATH})
    message(FATAL_ERROR "hexagon sdk error")
endif()

set(HEXAGON_SDK_PATH $ENV{HEXAGON_SDK_PATH})

# check host system name
if(NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    message(FATAL_ERROR "build host_hexagon can only in Ubuntu system")
endif()

# hexagon sdk inc dir
list(APPEND 3RDPARTY_HEXAGON_INC_DIRS "${HEXAGON_SDK_PATH}/incs/stddef")
list(APPEND 3RDPARTY_HEXAGON_INC_DIRS "${HEXAGON_SDK_PATH}/incs")
include_directories(${3RDPARTY_HEXAGON_INC_DIRS})

# hexagon tool version
find_hexagon_tool(${HEXAGON_SDK_PATH} HEXAGON_TOOLCHAIN_ROOT HEXAGON_TOOL_VERSION HEXAGON_SDK_VERSION)

# os version
execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE UBUNTU_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[0-2][0-9]\\.[0-9][0-9]" OS_VER_ID ${UBUNTU_VERSION})

if(${OS_VER_ID} GREATER_EQUAL "20")
    set(UBUNTU_NAME "Ubuntu20")
elseif(${OS_VER_ID} MATCHES "18")
    set(UBUNTU_NAME "Ubuntu18")
elseif(${OS_VER_ID} MATCHES "16")
    set(UBUNTU_NAME "Ubuntu16")
else()
    message(FATAL_ERROR "build host_hexagon can only in Ubuntu16/18/20 system")
endif()

# tool and lib path
if(HEXAGON_TOOL_VERSION GREATER_EQUAL "8.4")
    if (HEXAGON_TOOL_VERSION GREATER_EQUAL "8.8")
        set(HEXAGON_QAIC_PATH "${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/Ubuntu/qaic")
    else()
        set(HEXAGON_QAIC_PATH "${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/${UBUNTU_NAME}/qaic")
    endif()
elseif(HEXAGON_TOOL_VERSION MATCHES "(8.3)")
    set(HEXAGON_QAIC_PATH "${HEXAGON_SDK_PATH}/tools/qaic/${UBUNTU_NAME}/qaic")
elseif()
    message(FATAL_ERROR "Cannot support Hexagon toolchain version")
endif()

message(STATUS "HEXAGON_TOOLCHAIN_ROOT  = ${HEXAGON_TOOLCHAIN_ROOT}")
message(STATUS "HEXAGON_SDK_VERSION     = ${HEXAGON_SDK_VERSION}")
message(STATUS "HEXAGON_TOOL_VERSION    = ${HEXAGON_TOOL_VERSION}")
message(STATUS "HEXAGON_QAIC_PATH       = ${HEXAGON_QAIC_PATH}")