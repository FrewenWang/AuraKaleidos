include_directories(include)

function(add_module_exe module)
    include_directories(${module}/include)
    aux_source_directory(${module}/src SRC_CODE)

    string(CONCAT exe_name "sample_" ${module})

    add_executable(${exe_name} ${SRC_CODE})
    target_link_libraries(${exe_name} ${Aura_LIBS})

    install(TARGETS ${exe_name} DESTINATION runtime)
endfunction()

add_module_exe(array)
add_module_exe(context)
add_module_exe(memory)