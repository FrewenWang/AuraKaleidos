set(3RDPARTY_XTENSA_TM_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xtensa/tilemanager")

if(AURA_BUILD_XTENSA)
    set(3RDPARTY_XTENSA_INC_DIRS ${3RDPARTY_XTENSA_TM_DIRS}/device/include)
    aux_source_directory("${3RDPARTY_XTENSA_TM_DIRS}/device/src" 3RDPARTY_LIB_XTENSA_SRCS)
elseif(AURA_BUILD_XPLORER)
    set(3RDPARTY_XTENSA_INC_DIRS ${3RDPARTY_XTENSA_TM_DIRS}/host/include)
    aux_source_directory("${3RDPARTY_XTENSA_TM_DIRS}/host/src" 3RDPARTY_LIB_XTENSA_SRCS)
endif()

if(AURA_BUILD_XTENSA OR AURA_BUILD_XPLORER)
    include_directories(${3RDPARTY_XTENSA_INC_DIRS})
    install(DIRECTORY "${3RDPARTY_XTENSA_INC_DIRS}/" DESTINATION include/3rdparty/tile_manager)
endif()