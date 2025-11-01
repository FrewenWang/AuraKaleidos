

option(BUILD_EXAMPLE        "build example"          ON)

add_subdirectory(src)

if (BUILD_EXAMPLE)
    add_subdirectory(examples)
endif ()
