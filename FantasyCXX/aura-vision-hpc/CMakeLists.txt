# 设置工具类库的名字
set(TARGET_NAME AuraVisionHPC)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")

# 设置附属生成物的编译控制
if (BUILD_UNIT_TEST_LIB)
    # add_subdirectory(test)
endif ()

message(STATUS "[${TARGET_NAME}] ====================Build End============================")

add_subdirectory(examples)