if(AURA_BUILD_ANDROID)
    if(AURA_SHARED_LIBRARY)
        set(AURA_PUSH_LIBRARIES "adb push --sync ../../lib/libaura.so                                   \${device_path}")
        set(AURA_SET_LIBRARIES "export LD_LIBRARY_PATH=./;")
    endif()

    if(AURA_ENABLE_NN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_and_run-android-nn.sh.in"
                       "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run-nn.sh"
                       @ONLY)

        install(FILES     "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run-nn.sh"    DESTINATION sample/nn RENAME build_and_run.sh)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nn"                                  DESTINATION sample)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/nn"                          DESTINATION sample/nn/data)
    endif()

    if(NOT AURA_ENABLE_NN_LITE)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_and_run-android.sh.in"
                       "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.sh"
                       @ONLY)

        install(FILES     "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.sh"           DESTINATION sample/examples RENAME build_and_run.sh)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples"                                DESTINATION sample)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helper"                                  DESTINATION include)
        # 暂时先不进行install测试数据
        # install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/comm"                            DESTINATION sample/examples/data)
    endif()
elseif(AURA_BUILD_LINUX)
    if(AURA_SHARED_LIBRARY)
        set(AURA_SET_LIBRARIES "export LD_LIBRARY_PATH=../lib;")
    endif()

    if(${AURA_LINUX_ABI} MATCHES "(x86|X86)")
        set(AURA_ADD_COMPILE_OPTIONS "-DCMAKE_C_FLAGS=-m32 -DCMAKE_CXX_FLAGS=-m32")
    endif()

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_and_run-linux.sh.in"
                   "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.sh"
                   @ONLY)

    install(FILES     "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.sh"           DESTINATION sample RENAME build_and_run.sh)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"                               DESTINATION sample)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helper"                                  DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/comm"                            DESTINATION sample/data)
elseif(AURA_BUILD_WIN)
    if(AURA_SHARED_LIBRARY)
        set(AURA_SET_LIBRARIES "set PATH=%PATH%;../lib")
    endif()

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_and_run-win.bat.in"
                   "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.bat"
                   @ONLY)

    install(FILES     "${CMAKE_BINARY_DIR}/configure_gen/sample/build_and_run.bat"          DESTINATION sample RENAME build_and_run.bat)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"                               DESTINATION sample)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helper"                                  DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/comm"                            DESTINATION sample/data)
endif()
