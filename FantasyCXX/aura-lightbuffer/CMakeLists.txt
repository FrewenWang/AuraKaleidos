# AuraLightBuffer
# 轻量级结构化二进制序列化（LightBuffer）库。仅依赖标准库 + 内置 FileUtil。
#
# 说明：本文件早先是一份从别处拷贝来的 CMake（误引用了并不存在/不属于本模块的
# AuraLib、AuraCV、OPENCV_LIB 目标，以及 OpenCL 等），导致无法参与顶层构建。
# 实际源码（src/ 下）只用到标准库与 aura/auralib/utils/FileUtil.h，
# 因此这里改为只构建一个静态库 light_buffer。
#
# 注意：example/ 引用的是另一套 "plainbuffer" 示例源码（error.cpp / schema_parser.cpp /
# serializer.cpp / text_format.cpp / util.cpp）以及一个缺失的 prebuilt/plainc 代码生成工具，
# 与 src/ 实际源码不一致，故默认不构建 example。
set(LIB_NAME light_buffer)
message(STATUS "[AuraLightBuffer] =================== AuraLightBuffer Build ===================")

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

collect_source_files(LIB_SRCS GLOB_RECURSE .)

# 静态库：不依赖 OpenCV / AuraLib / AuraCV（这些是旧工程布局的残留引用，本模块用不到）
add_library(${LIB_NAME} STATIC ${LIB_SRCS})
target_include_directories(${LIB_NAME} PUBLIC ${INCLUDE_ROOT})
target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(${LIB_NAME} PRIVATE $<$<CONFIG:Release>:-O3>)

message(STATUS "[AuraLightBuffer] =================== AuraLightBuffer Build End===================")
