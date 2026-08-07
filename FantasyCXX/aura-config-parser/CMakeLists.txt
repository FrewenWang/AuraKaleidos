# aura-config-parser
# 轻量级 INI 配置文件解析库（仅依赖标准库）。
set(TARGET_NAME aura-config-parser)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

collect_source_files(LIB_SRCS GLOB_RECURSE .)

# 静态库：IniParser 实现见 src/ini_parser.cpp
add_library(${TARGET_NAME} STATIC ${LIB_SRCS})
target_include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_ROOT})

# 示例程序（注意：samples/ini_parser 已占用 DemoIniParser 名称，这里用不同名称避免冲突）
add_executable(DemoAuraConfigParser examples/main.cpp)
target_link_libraries(DemoAuraConfigParser PRIVATE ${TARGET_NAME})

message(STATUS "[${TARGET_NAME}] ====================Build End============================")
