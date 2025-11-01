include_directories(include)

function(add_module_exe module)
    include_directories(${module}/include)
    aux_source_directory(${module}/src SRC_CODE)

    string(CONCAT exe_name "sample_" ${module})

    add_executable(${exe_name} ${SRC_CODE})
    target_link_libraries(${exe_name} ${Aura_LIBS})

    install(TARGETS ${exe_name} DESTINATION ops)
endfunction()

add_module_exe(cvtcolor)
add_module_exe(feature2d)
add_module_exe(filter)
add_module_exe(hist)
add_module_exe(matrix)
add_module_exe(misc)
add_module_exe(morph)
add_module_exe(pyramid)
add_module_exe(resize)
add_module_exe(warp)
# ...