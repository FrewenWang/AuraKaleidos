include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(tool_name caffe2ncnn)

add_executable(
    ${tool_name}
    caffe2ncnn.cpp
    caffe.pb.cc
)

# find_package(Protobuf REQUIRED)：在系统目录下寻找Protobuf库的头文件位置、库文件位置以及库文件名称,并将其设为变量
find_package(Protobuf REQUIRED)
IF(PROTOBUF_FOUND)
    message(STATUS "protobuf library found")
    message(STATUS "Protobuf version : ${Protobuf_VERSION}" )
    message(STATUS "Protobuf include path : ${Protobuf_INCLUDE_DIRS}" )
    message(STATUS "Protobuf libraries : ${Protobuf_LIBRARIES}" )
ELSE()
    message(FATAL_ERROR "protobuf library is needed but cant be found")
ENDIF()

target_include_directories(${tool_name} PUBLIC ${PROTOBUF_INCLUDE_DIRS}) # /usr/local/include
target_link_libraries(${tool_name} PUBLIC  ${Protobuf_LIBRARIES})

set_target_properties(${tool_name} PROPERTIES CXX_STANDARD 11)