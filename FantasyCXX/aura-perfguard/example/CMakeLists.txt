message(STATUS "[AuraPerfGuard] ==================== AuraPerfGuardDemo Examples Build Begin =========================")
# 设置工具类库的名字
set(APP_NAME DemoPerfGuard)

# 设置工具类库的文件列表
list(APPEND app_srcs main.cpp)

# 编译成可执行程序
add_executable(${APP_NAME} ${app_srcs})


# 设置目标产物的include头文件的路径
target_include_directories(${APP_NAME} PUBLIC
        ${CMAKE_SOURCE_DIR}/AuraUtils/include)

# 设置目标产物link链接依赖库
message(STATUS "[AuraPerfGuard] OPENCV_LIB:${OPENCV_LIB}")
target_link_libraries(${APP_NAME} PUBLIC
        AuraUtils)

message(STATUS "[AuraPerfGuard] ==================== AuraPerfGuardDemo Examples Build End============================")