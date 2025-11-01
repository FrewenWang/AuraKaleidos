//
// Created by Frewen.Wang on 25-9-14.
//
# pragma once

namespace aura::utils
{

/**
 * @brief Enumeration representing the status of an operation.
 */
enum class Status
{
    OK    = 0,                  /*!< Operation completed successfully. */
    ERR_UNKNOWN  = -1,          /*!< Operation encountered an error. */
    ERR_NULL_PTR = -2,          /*!< Operation encountered an error. */
    ERR_ARGS = -3,              /*!< Operation encountered an error. */
};

}
