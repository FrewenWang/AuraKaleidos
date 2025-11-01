#ifndef AURA_OPS_MISC_CCL_HPP__
#define AURA_OPS_MISC_CCL_HPP__

#include "aura/ops/core.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup misc Miscellaneous Iaura Process
 * @}
 */

/**
 * @addtogroup misc
 * @{
 */
namespace aura
{
/**
 * @addtogroup misc
 * @{
 */

/**
 * @brief Enum class representing different connectivity types.
 */
enum class ConnectivityType
{
    CROSS = 0, /*!< 4 connectivity for 2D orthogonal*/
    SQUARE,    /*!< 8 connectivity for 2D*/
    DIAGONAL,  /*!< 4 connectivity for 2D diagonal, not support now */
    CUBE,      /*!< 26 connectivity for 3D, not support now */
};

/**
 * @brief Enum class representing different equivalence solvers.
 */
enum class EquivalenceSolver
{
    UNION_FIND = 0,           /*!< Union-Find for most CCL algos */
    UNION_FIND_PATH_COMPRESS, /*!< Union-Find with path/rank compression optimization */
    REM_SPLICING,             /*!< Interleaved Rem algorithm with SPlicing for <<A New Parallel Algorithm for Two - Pass Connected Component Labeling>> */
    THREE_TABLE_ARRAYS,       /*!< special Union-Find for <<A Run-Based Two-Scan Labeling Algorithm>> */
};

/**
 * @brief Enum class representing different connected component labeling(CCL) algorithms.
 */
enum class CCLAlgo
{
    SAUF = 0,  /*!< Scan plus Array-based Union-Find for 4/8 connectivity CCL */
    BBDT,      /*!< Block-Based Decision Table for 8 connectivity CCL */
    SPAGHETTI, /*!< DAG auto-generatation for 4/8 connectivity CCL */
    HA_GPU,    /*!< Hardware Accelerated for 4/8 connectivity CCL */
};

/**
 * @brief Overloaded stream insertion operator for ConnectivityType.
 *
 * This function converts a ConnectivityType to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The ConnectivityType to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ConnectivityType type)
{
    switch (type)
    {
        case ConnectivityType::CROSS:
        {
            os << "cross";
            break;
        }
        case ConnectivityType::SQUARE:
        {
            os << "square";
            break;
        }
        case ConnectivityType::DIAGONAL:
        {
            os << "diagonal";
            break;
        }
        case ConnectivityType::CUBE:
        {
            os << "cube";
            break;
        }
        default:
        {
            os << "Invalid";
            break;
        }
    }

    return os;
}

/**
 * @brief Overloaded stream insertion operator for EquivalenceSolver.
 *
 * This function converts a EquivalenceSolver to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The EquivalenceSolver to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, EquivalenceSolver type)
{
    switch (type)
    {
        case EquivalenceSolver::UNION_FIND:
        {
            os << "UnionFindSolver";
            break;
        }
        case EquivalenceSolver::UNION_FIND_PATH_COMPRESS:
        {
            os << "UFPCSolver";
            break;
        }
        case EquivalenceSolver::REM_SPLICING:
        {
            os << "RemSpliceSolver";
            break;
        }
        case EquivalenceSolver::THREE_TABLE_ARRAYS:
        {
            os << "TTASolver";
            break;
        }
        default:
        {
            os << "Invalid";
            break;
        }
    }

    return os;
}

/**
 * @brief Overloaded stream insertion operator for CCLAlgo.
 *
 * This function converts a CCLAlgo to its corresponding string representation
 * and inserts it into an output stream.
 *
 * @param os The output stream.
 * @param binary_type The CCLAlgo to be inserted.
 *
 * @return A reference to the output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, CCLAlgo type)
{
    switch (type)
    {
        case CCLAlgo::SAUF:
        {
            os << "SAUF";
            break;
        }
        case CCLAlgo::BBDT:
        {
            os << "BBDT";
            break;
        }
        case CCLAlgo::SPAGHETTI:
        {
            os << "SPAGHETTI";
            break;
        }
        case CCLAlgo::HA_GPU:
        {
            os << "HA";
            break;
        }
        default:
        {
            os << "Invalid";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts a EquivalenceSolver to its string representation.
 *
 * @param method The EquivalenceSolver to be converted.
 *
 * @return The string representation of the EquivalenceSolver.
 */
AURA_INLINE std::string EquivalenceSolverToString(EquivalenceSolver type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief Converts a ConnectivityType to its string representation.
 *
 * @param method The ConnectivityType to be converted.
 *
 * @return The string representation of the ConnectivityType.
 */
AURA_INLINE std::string ConnectivityTypeToString(ConnectivityType type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief Converts a CCLAlgo to its string representation.
 *
 * @param method The CCLAlgo to be converted.
 *
 * @return The string representation of the CCLAlgo.
 */
AURA_INLINE std::string CCLAlgoTypeToString(CCLAlgo type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief Class representing the ConnectComponentLabel operation.
 *
 * This class represents a connected components labeling operation.
 * It is recommended to use the `IConnectComponentLabel` API, which internally calls this class.
 *
 * The approximate internal call within the `IConnectComponentLabel` function is as follows:
 *
 * @code
 * ConnectComponentLabel ccl(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, ccl, &src, &dst, algo_type, connectivity_type, solver_type);
 * @endcode
 */
class AURA_EXPORTS ConnectComponentLabel : public Op
{
public:
    /**
     * @brief Constructor for ConnectComponentLabel class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    ConnectComponentLabel(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the Threshold operation.
     *
     * For more details, please refer to @ref connect_component_label_details
     */
    Status SetArgs(const Array *src, Array *dst, CCLAlgo algo_type = CCLAlgo::SPAGHETTI,
                   ConnectivityType connectivity_type = ConnectivityType::CROSS,
                   EquivalenceSolver solver_type = EquivalenceSolver::UNION_FIND_PATH_COMPRESS);

    /**
     * @brief Generate ConnectComponentLabel opencl precompiled cache.
     *
     * @param dst_elem_type The dst array element type.
     * @param algo_type The SAUF/BBDT/SPAGHETTI algorithms is on CPU, for binary iaura inputs with different densities and granularities,
     * their performance has its own advantages and disadvantages. It is recommended to test and choose the best algorithm on your own.
     * HA_GPU is on GPU and is optimal for large iaura size, It is only supported on QCOM Adreno GPU and recommended to use HA_GPU
     * on platforms 8650 and after it, otherwise perfermance maybe worse than CPU on 8550 or before.
     * @param connectivity_type The component connectivity detection mode(CROSS or SQUARE for now).
     */
    static Status CLPrecompile(Context *ctx, ElemType dst_elem_type, CCLAlgo algo_type, ConnectivityType connectivity_type);
};

/**
 * @brief Apply a connected components labeling operation of src mat.
 *
 * @anchor connect_component_label_details
 * This function applies a connected components labeling operation to the source matrix.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix, must be binary iaura which channel num must be 1 and data type must be U8.
 * @param dst The destination matrix which channel num must be 1 and The upper bound of the data type must be greater
 * than the maximum number of possible connected components, please use 32bit when unsure.
 * @param algo_type The SAUF/BBDT/SPAGHETTI algorithms is on CPU, for binary iaura inputs with different densities and granularities,
 * their performance has its own advantages and disadvantages. It is recommended to test and choose the best algorithm on your own.
 * HA_GPU is on GPU and is optimal for large iaura size, It is only supported on QCOM Adreno GPU and recommended to use HA_GPU
 * on platforms 8650 and after it, otherwise perfermance maybe worse than CPU on 8550 or before.
 * @param connectivity_type The component connectivity detection mode(CROSS or SQUARE for now).
 * @param solver_type The union-find set algorithm for equivalence solver, for binary iaura inputs with different densities and granularities,
 * their performance has its own advantages and disadvantages. It is recommended to test and choose the best algorithm on your own. When using
 * HA_GPU algo, solver_type is unused and could be anyone.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms    |  src   |        dst    |       CCLAlgo       |     EquivalenceSolver     | ConnectivityType |
 * -------------|--------|---------------|---------------------|---------------------------|------------------|
 * NONE         |  U8C1  |  U32C1 S32C1  | SAUF/BBDT/SPAGHETTI |            ALL            |   CROSS/SQUARE   |
 * NONE         |  U8C1  |  U8C1  U16C1  | SAUF/BBDT/SPAGHETTI | except THREE_TABLE_ARRAYS |   CROSS/SQUARE   |
 * OpenCL(QCOM) |  U8C1  |  U32C1 S32C1  |        HA_GPU       |        Not use, Any       |   CROSS/SQUARE   |

 *
 * @note ConnectivityType only support CROSS and SQUARE for now.
 */
AURA_EXPORTS Status IConnectComponentLabel(Context *ctx, const Mat &src, Mat &dst, CCLAlgo algo_type = CCLAlgo::SPAGHETTI,
                                           ConnectivityType connectivity_type = ConnectivityType::CROSS,
                                           EquivalenceSolver solver_type = EquivalenceSolver::UNION_FIND_PATH_COMPRESS,
                                           const OpTarget &target = OpTarget::Default());
} // namespace aura

#endif // AURA_OPS_MISC_CCL_HPP__