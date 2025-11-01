function(logger LEVEL MSG)
    if (NOT ENABLE_LOG)
        return()
    endif ()

    # 日志级别对应权重
    set(_levels DEBUG INFO WARN ERROR NONE)
    list(FIND _levels ${LEVEL} _level_index)
    list(FIND _levels ${LOG_LEVEL} _cfg_index)

    if (_level_index EQUAL -1)
        message(WARNING "Unknown log level: ${LEVEL}")
        return()
    endif ()
    # 如果当前日志等级 >= 配置等级，才打印
    if (_level_index GREATER_EQUAL _cfg_index)
        if (${LEVEL} STREQUAL "ERROR")
            message(FATAL_ERROR "[build]--[${LEVEL}] ${MSG}")
        elseif (${LEVEL} STREQUAL "WARN")
            message(WARNING "[build]--[${LEVEL}] ${MSG}")
        else ()
            message(STATUS "[build]--[${LEVEL}] ${MSG}")
        endif ()
    endif ()
endfunction()


macro(logger_d MSG)
    logger(DEBUG "${MSG}")
endmacro()

macro(logger_i MSG)
    logger(INFO "${MSG}")
endmacro()

macro(logger_w MSG)
    logger(WARN "${MSG}")
endmacro()

macro(logger_e MSG)
    logger(ERROR "${MSG}")
endmacro()