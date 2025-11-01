#ifndef AURA_RUNTIME_CORE_STATUS_HPP__
#define AURA_RUNTIME_CORE_STATUS_HPP__

#include "aura/runtime/core/defs.hpp"

#include <iostream>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup types Runtime Core Types
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup types
 * @{
*/

/**
 * @brief Enumeration representing the status of an operation.
 */
enum class Status
{
    OK    = 0,     /*!< Operation completed successfully. */
    ERROR = -1,    /*!< Operation encountered an error. */
    ABORT = -2,    /*!< Operation was aborted. */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, Status status)
{
    switch (status)
    {
        case Status::OK:
        {
            os << "Status::OK";
            break;
        }

        case Status::ERROR:
        {
            os << "Status::ERROR";
            break;
        }

        case Status::ABORT:
        {
            os << "Status::ABORT";
            break;
        }

        default:
        {
            os << "INVALID";
            break;
        }
    }

    return os;
}

AURA_INLINE std::string StatusToString(Status status)
{
    std::ostringstream ss;
    ss << status;
    return ss.str();
}

/**
 * @brief Bitwise OR assignment operator for combining Status values.
 * 
 * If either of the Status values is ERROR, the result is ERROR.
 * If either of the Status values is ABORT, the result is ABORT.
 * Otherwise, the result is OK. (Both values are OK)
 * 
 * @param s0 The first Status value.
 * @param s1 The second Status value.
 * 
 * @return The combined Status value.
 */
AURA_INLINE Status& operator|=(Status &s0, Status s1)
{
    if (Status::ERROR == s0 || Status::ERROR == s1)
    {
        return (s0 = Status::ERROR);
    }
    if (Status::ABORT == s0 || Status::ABORT == s1)
    {
        return (s0 = Status::ABORT);
    }
    return (s0 = Status::OK);
}

/**
 * @brief Bitwise OR operator for combining Status values.
 * 
 * If either of the Status values is ERROR, the result is ERROR.
 * If either of the Status values is ABORT, the result is ABORT.
 * Otherwise, the result is OK. (Both values are OK)
 * 
 * @param s0 The first Status value.
 * @param s1 The second Status value.
 * 
 * @return The combined Status value.
 */
AURA_INLINE Status operator|(Status &s0, Status s1)
{
    if (Status::ERROR == s0 || Status::ERROR == s1)
    {
        return Status::ERROR;
    }
    if (Status::ABORT == s0 || Status::ABORT == s1)
    {
        return Status::ABORT;
    }
    return Status::OK;
}

/**
 * @}
*/
} // namespace aura
#endif // AURA_RUNTIME_CORE_STATUS_HPP__