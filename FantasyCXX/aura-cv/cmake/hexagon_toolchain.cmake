# hexagon sdk path
if(NOT DEFINED ENV{HEXAGON_SDK_PATH})
    message(FATAL_ERROR "hexagon sdk error")
endif()

set(HEXAGON_SDK_PATH $ENV{HEXAGON_SDK_PATH})

# check host system name
if(NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    message(FATAL_ERROR "build host_hexagon can only in Ubuntu system")
endif()

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

# build hexagon
set(AURA_BUILD_HEXAGON ON)

# hexagon tool version
include(${CMAKE_CURRENT_LIST_DIR}/macros.cmake)
find_hexagon_tool(${HEXAGON_SDK_PATH} HEXAGON_TOOLCHAIN_ROOT HEXAGON_TOOL_VERSION HEXAGON_SDK_VERSION)

# set tool dir
if(AURA_HEXAGON_ARCH)
    set(HEXAGON_TOOLCHAIN_LIB ${HEXAGON_TOOLCHAIN_ROOT}/target/hexagon/lib/${AURA_HEXAGON_ARCH}/G0/pic)
endif()

# set complier excute program
set(CMAKE_C_COMPILER      ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-clang)
set(CMAKE_CXX_COMPILER    ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-clang++)
set(CMAKE_ASM_COMPILER    ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-clang++)
set(CMAKE_AR              ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-ar)
set(CMAKE_RANLIB          ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-ranlib)
set(CMAKE_NM              ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-nm)
set(CMAKE_OBJDUMP         ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-objdump)
set(CMAKE_OBJCOPY         ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-objcopy)
set(CMAKE_LINK            ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-link)
set(CMAKE_STRIP           ${HEXAGON_TOOLCHAIN_ROOT}/bin/hexagon-strip)

# set compiler id
set(CMAKE_C_COMPILER_ID,    "Clang")
set(CMAKE_CXX_COMPILER_ID,  "Clang")
set(CMAKE_ASM_COMPILER_ID,  "Clang")

# set compile flags
set(HEXAGON_COMPILER_FLAGS)
if(AURA_HEXAGON_ARCH)
    list(APPEND HEXAGON_COMPILER_FLAGS
    -m${AURA_HEXAGON_ARCH})           # Specify the target architecture and generate the corresponding architecture code
endif()

list(APPEND HEXAGON_COMPILER_FLAGS
    -G0                               # Limit data object allocation size to 0 (default setting)
    -O3                               # Compile optimization level
    -Wall                             # Turn on all warnings
    -Werror                           # Warnings are treated as errors
    -Wno-cast-align                   # Turn off alignment warnings
    -Wpointer-arith                   # Turn on warnings null pointers
    -Wno-missing-braces               # Turn off missing brace initialization warnings
    -Wno-strict-aliasing              # Close the strict aliasing restriction
    -fno-exceptions                   # Close exceptions
    -fno-strict-aliasing              # Close the strict aliasing restriction
    -fno-zero-initialized-in-bss      # Close bss segment 0 value initialization
    -fdata-sections                   # Create separate segments for symbols
    -fstack-protector                 # Add stack protection
    -mhvx                             # Enable hvx support
    -mhvx-length=128B)                # Set hvx vector length 128Bytes

if(HEXAGON_TOOL_VERSION MATCHES "8.6|8.7") # Temporary fix sdk5.3 tool8.6/tool8.7 compile bug
    list(APPEND HEXAGON_COMPILER_FLAGS
    -Wno-error=unused-command-line-argument  # only avoid build warnings
    -mllvm -hexagon-vector-combine=false)    # fix sdk5.3 tool8.6/8.7 compile bug
endif()

# add -D__V_DYNAMIC__ for shared library
if(AURA_SHARED_LIBRARY)
    list(APPEND HEXAGON_COMPILER_FLAGS -D__V_DYNAMIC__)
endif()

# replace ; with " "
string(REPLACE ";" " " HEXAGON_COMPILER_FLAGS "${HEXAGON_COMPILER_FLAGS}")

# set CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_ASM_FLAGS
set(CMAKE_C_FLAGS   "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_CXX_FLAGS "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_ASM_FLAGS "" CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_C_FLAGS   "${HEXAGON_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS "${HEXAGON_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS "${HEXAGON_COMPILER_FLAGS}")

# set link flags
set(HEXAGON_LINKER_START_FLAGS)
if(AURA_RELEASE)
    list(APPEND HEXAGON_LINKER_START_FLAGS -s)
endif()
list(APPEND HEXAGON_LINKER_START_FLAGS
    -z                                    # Protection options
    relro                                 # Read-only Relocation
    --hash-style=sysv                     # Setting file style
    -march=hexagon                        # Target platform
    -mcpu=hexagon${AURA_HEXAGON_ARCH}     # Target cpu architecture
    -shared                               # Generate shared files
    -call_shared                          # Used with -shared
    -G0                                   # Limit data object allocation size to 0 (default setting)
    "-o <TARGET>"                         # Specify file name
    ${HEXAGON_TOOLCHAIN_LIB}/initS.o      # Hexagon dynamic library specified file
    -L${HEXAGON_TOOLCHAIN_LIB}            # Add library path
    --no-threads                          # Uses a single thread
    -Bsymbolic                            # Force local global variable definitions
    --wrap=malloc                         # Use internal malloc
    --wrap=calloc                         # Use internal calloc
    --wrap=free                           # Use internal free
    --wrap=realloc                        # Use internal realloc
    --wrap=memalign                       # Use internal memalign
    -lc++                                 # Link libc++
    "-soname=<TARGET_SONAME>"             # Specify soname
    --start-group                         # Group start flag
    <OBJECTS>                             # List of linked files
    <LINK_LIBRARIES>                      # List of dependency libraries
    --end-group)                          # Group end flag

# replace ; with " "
string(REPLACE ";" " " HEXAGON_LINKER_START_FLAGS "${HEXAGON_LINKER_START_FLAGS}")

# set create shared library command
set(CMAKE_C_CREATE_SHARED_LIBRARY
    "${CMAKE_LINK} ${HEXAGON_LINKER_START_FLAGS} -lgcc ${HEXAGON_TOOLCHAIN_LIB}/finiS.o")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY
    "${CMAKE_LINK} ${HEXAGON_LINKER_START_FLAGS} -lc++ -lc++abi ${HEXAGON_TOOLCHAIN_LIB}/finiS.o")

# set create static library command
set(CMAKE_C_CREATE_STATIC_LIBRARY
    "${CMAKE_AR} -rcs <TARGET> <LINK_FLAGS> <OBJECTS>;${CMAKE_RANLIB} <TARGET>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY
    "${CMAKE_AR} -rcs <TARGET> <LINK_FLAGS> <OBJECTS>;${CMAKE_RANLIB} <TARGET>")

# do not add run time path information
set(CMAKE_SKIP_RPATH TRUE CACHE BOOL SKIP_RPATH FORCE)

# compiler root paths
set(CMAKE_FIND_ROOT_PATH get_file_component(${C_COMPILER} PATH))
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# remove -rdynamic flag
set(__LINUX_COMPILER_GNU 1)
macro(__linux_compiler_gnu lang)
    set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "")
endmacro()