# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# add op
aura_add_op(resize)
aura_add_op(filter)
aura_add_op(matrix)
aura_add_op(pyramid)
aura_add_op(misc)
aura_add_op(feature2d)
aura_add_op(cvtcolor)
aura_add_op(morph)
aura_add_op(hist)
aura_add_op(warp)

# update
aura_update_group_vars()