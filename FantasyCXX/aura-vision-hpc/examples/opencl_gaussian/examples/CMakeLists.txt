cmake_minimum_required(VERSION 3.28.3)

set(TARGET_NAME gaussian_cl_example)

# add include libs
get_property(GAUSSIAN_CL_LIB_INC_DIR GLOBAL PROPERTY "GAUSSIAN_CL_LIB_INC_DIR")

# add inc dir
#message(STATUS "[${TARGET_NAME}] GAUSSIAN_CL_LIB_INC_DIR = ${GAUSSIAN_CL_LIB_INC_DIR}")
include_directories(${GAUSSIAN_CL_LIB_INC_DIR})

# add executable
#add_executable(${TARGET_NAME} main.cpp)
#add_executable(${TARGET_NAME} main_sigma.cpp)
add_executable(${TARGET_NAME} gaussian_cl_example.cpp)

# target link libraries
target_link_libraries(${TARGET_NAME} gaussian_cl)

# install
install(TARGETS ${TARGET_NAME} DESTINATION gaussian_cl/bin)


add_subdirectory(test_images)