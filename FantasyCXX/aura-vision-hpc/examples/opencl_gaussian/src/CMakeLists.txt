
set(TARGET_LIB_NAME gaussian_cl)
# set public include dir
set(GAUSSIAN_CL_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../inc")
list(APPEND GAUSSIAN_CL_LIB_INC_DIR ${GAUSSIAN_CL_INC_DIR})

# set private include dir
# not exits

include_directories(${GAUSSIAN_CL_LIB_INC_DIR})
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/../src" TARGET_SRCS)

#message(STATUS "[${TARGET_NAME}] GAUSSIAN_CL_INC_DIR:${GAUSSIAN_CL_INC_DIR}")
#message(STATUS "[${TARGET_NAME}] TARGET_SRCS:${TARGET_SRCS}")
set_property(GLOBAL PROPERTY GAUSSIAN_CL_LIB_INC_DIR ${GAUSSIAN_CL_LIB_INC_DIR})

add_library(${TARGET_LIB_NAME} SHARED ${TARGET_SRCS})


target_include_directories(
        ${TARGET_LIB_NAME} PUBLIC
        ${INCLUDE_ROOT}
        ${SRC_ROOT})

install(DIRECTORY ${GAUSSIAN_CL_LIB_INC_DIR} DESTINATION install/${TARGET_LIB_NAME}/include FILES_MATCHING PATTERN "*.h")
install(TARGETS ${TARGET_LIB_NAME} DESTINATION install/${TARGET_LIB_NAME}/lib)