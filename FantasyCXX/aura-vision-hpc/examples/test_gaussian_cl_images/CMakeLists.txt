# This file handles building each task as sub-project.
#
# Each task becomes an independent project in which you can experiment.
#
# Tasks are added as separate sub-projects so as compilation errors in
# one task don't affect another task.


add_executable(test_image test_images.cpp)

target_link_libraries(test_image ${OEPNCL_PIB})

install(TARGETS test_image DESTINATION install/gaussian_cl/bin)