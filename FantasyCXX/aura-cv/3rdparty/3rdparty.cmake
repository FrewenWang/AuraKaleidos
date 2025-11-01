if(AURA_BUILD_ANDROID)
    include(3rdparty/dmabufheap/dmabufheap.cmake)

    if(AURA_ENABLE_OPENCL)
        include(3rdparty/opencl/opencl.cmake)
    endif(AURA_ENABLE_OPENCL)

    if(AURA_ENABLE_HEXAGON)
        include(3rdparty/hexagon/host.cmake)
    endif()

elseif(AURA_BUILD_HEXAGON)
    include(3rdparty/hexagon/device.cmake)
endif()

if(AURA_BUILD_XPLORER OR AURA_ENABLE_XTENSA OR AURA_BUILD_XTENSA)
    include(3rdparty/xtensa/xtensa.cmake)
endif()

if(AURA_ENABLE_NN)
    include(3rdparty/nn/nn.cmake)
endif()