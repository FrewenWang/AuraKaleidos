set(ARCH osx-x86_64)
set(PLAINC_BIN ${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/${ARCH}/plainc)
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/person.plain.cpp
        COMMAND ${PLAINC_BIN} -I=${PROTO_SRC_DIR} --cpp_out=${PROTO_SRC_DIR} person.proto
        COMMENT "compiling proto messages..."
        VERBATIM
)

list(APPEND example_srcs
        main.cpp
        person.plain.cpp
        ../src/error.cpp
        ../src/schema.cpp
        ../src/schema_parser.cpp
        ../src/serializer.cpp
        ../src/text_format.cpp
        ../src/util.cpp)

#add_executable(DemoLightBuffer ${example_srcs})
#target_compile_options(DemoLightBuffer PRIVATE -O3 -std=c++11)
#target_include_directories(DemoLightBuffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)