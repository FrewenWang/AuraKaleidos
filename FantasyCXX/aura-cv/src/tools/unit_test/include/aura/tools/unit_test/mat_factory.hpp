#ifndef AURA_TOOLS_UNIT_TEST_MAT_FACTORY_HPP__
#define AURA_TOOLS_UNIT_TEST_MAT_FACTORY_HPP__

#include "aura/runtime/mat.h"

#include <random>
#include <mutex>
#include <list>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup unit_test Unit Test
 * @}
 */

#define MAT_FACTORY_MAX_PATH         (256)

namespace aura
{
/**
 * @addtogroup unit_test
 * @{
 */

/**
 * @brief Structure representing the descriptor of a matrix in AURA.
 *
 * The MatDesc structure provides detailed information about a matrix, including its type, parameters,
 * element type, memory type, sizes, and strides.
 *
 * It serves as a comprehensive descriptor for defining the characteristics of matrices within the AURA framework.
 */
struct AURA_EXPORTS MatDesc
{
    /**
     * @brief Enumeration defining types of matrices.
     */
    enum class Type
    {
        FILE    = 0, /*!< Matrix loaded from a file. */
        RAND,        /*!< Matrix with random data. */
        DERIVED,     /*!< Matrix derived from a base matrix. */
        EMPTY,       /*!< Empty matrix. */
    };

    /**
     * @brief Structure representing a range for random values.
     */
    struct RandRange
    {
        DT_F32 min; /*!< Minimum value of the range. */
        DT_F32 max; /*!< Maximum value of the range. */
    };

    /**
     * @brief Structure representing parameters for a derived matrix.
     */
    struct DerivedParam
    {
        Mat    *base; /*!< Pointer to the base matrix. */
        DT_F32 alpha; /*!< Alpha parameter for derivation. */
        DT_F32 beta;  /*!< Beta parameter for derivation. */
    };

    Type type;        /*!< Type of the matrix. */
    union
    {
        DT_CHAR      file_path[MAT_FACTORY_MAX_PATH];   /*!< File path for FILE type. */
        RandRange    rand_range;                        /*!< Range for RAND type. */
        DerivedParam derived_param;                     /*!< Parameters for DERIVED type. */
    } param;

    ElemType elem_type; /*!< Element type of the matrix. */
    DT_S32   mem_type;  /*!< Memory type of the matrix. */
    Sizes3   sizes;     /*!< Sizes of the matrix. */
    Sizes    strides;   /*!< Strides of the matrix. */
};

/**
 * @brief Structure representing information about a matrix.
 */
struct AURA_EXPORTS MatInfo
{
    MatDesc desc;       /*!< Description of the matrix. */
    Mat     mat;        /*!< Actual matrix. */
    DT_BOOL available;  /*!< Flag indicating if the matrix is available. */
};

/**
 * @brief Class for generating matrices.
 * 
 * The `MatFactory` class provides comprehensive functionality for generating matrices with various characteristics.
 * It supports creating matrices with random data, derived matrices from a base matrix, matrices loaded from files, and empty matrices.
 * Additionally, the class manages memory usage and can load matrices into a base list from files.
 */
class AURA_EXPORTS MatFactory
{
public:
    /**
     * @brief Constructor for MatFactory.
     *
     * @param ctx The pointer to the Context object.
     * @param max_mem The maximum memory allowed for matrices in megabytes (default is 1024MB).
     * @param seed The seed for the random number generator (default is std::mt19937_64::default_seed).
     */
    MatFactory(Context *ctx,
               DT_S64 max_mem = 1024,
               DT_S64 seed = std::mt19937_64::default_seed) : m_ctx(ctx), m_total_mem(0),
                                                              m_rand_seed(seed), m_rand_engine(m_rand_seed)
    {
        m_max_mem = (max_mem << 20); // convert MB into Byte
    }

    /**
     * @brief Destructor for MatFactory.
     * 
     * Cleans up resources associated with the MatFactory instance.
     */
    ~MatFactory()
    {
        Clear();
    }

    /**
     * @brief Deleted copy constructor and copy assignment operator to prevent copying instances.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(MatFactory);

    /**
     * @brief Get an empty matrix.
     *
     * @param elem_type The element type of the matrix.
     * @param sizes The sizes of the matrix.
     * @param mem_type The memory type of the matrix (default is AURA_MEM_DEFAULT).
     * @param strides The strides of the matrix (default is empty).
     * 
     * @return The empty matrix.
     */
    Mat GetEmptyMat(const ElemType &elem_type, const Sizes3 &sizes, 
                    DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

    /**
     * @brief Get a matrix with random data.
     *
     * @param min The minimum value for random data.
     * @param max The maximum value for random data.
     * @param elem_type The element type of the matrix.
     * @param sizes The sizes of the matrix.
     * @param mem_type The memory type of the matrix (default is AURA_MEM_DEFAULT).
     * @param strides The strides of the matrix (default is empty).
     * 
     * @return The matrix with random data.
     */
    Mat GetRandomMat(DT_F32 min, DT_F32 max, const ElemType &elem_type, const Sizes3 &sizes, 
                     DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

    /**
     * @brief Get a matrix derived from a base matrix. (dst pixel = alpha * src pixel + beta)
     *
     * @param alpha The alpha parameter for derivation.
     * @param beta The beta parameter for derivation.
     * @param elem_type The element type of the matrix.
     * @param sizes The sizes of the matrix.
     * @param mem_type The memory type of the matrix (default is AURA_MEM_DEFAULT).
     * @param strides The strides of the matrix (default is empty).
     * @param file_path The file path for the matrix (default is empty).
     * 
     * @return The derived matrix.
     */
    Mat GetDerivedMat(DT_F32 alpha, DT_F32 beta, const ElemType &elem_type, const Sizes3 &sizes, 
                      DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes(),
                      const std::string &file_path = std::string());

    /**
     * @brief Get a matrix from a file.
     *
     * @param file_path The file path for the matrix.
     * @param elem_type The element type of the matrix.
     * @param sizes The sizes of the matrix.
     * @param mem_type The memory type of the matrix (default is AURA_MEM_DEFAULT).
     * @param strides The strides of the matrix (default is empty).
     * 
     * @return The matrix loaded from the file.
     */
    Mat GetFileMat(const std::string &file_path, const ElemType &elem_type, const Sizes3 &sizes, 
                   DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

    /**
     * @brief Load matrices into the base list from a file.
     *
     * @param file_path The file path for loading matrices.
     * @param elem_type The element type of the matrices.
     * @param sizes The sizes of the matrices.
     * @param mem_type The memory type of the matrices (default is AURA_MEM_DEFAULT).
     * @param strides The strides of the matrices (default is empty).
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status LoadBaseMat(const std::string &file_path, const ElemType &elem_type, const Sizes3 &sizes, 
                       DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

    /**
     * @brief Put a Mat object back into the MatFactory's dynamic list.
     *
     * @param mat Reference to the Mat object to be put back into the dynamic list.
     */
    DT_VOID PutMats(Mat &mat);

    /**
     * @brief Put multiple Mat objects back into the MatFactory's dynamic list.
     *
     * @param mat Reference to the first Mat object to be put back into the dynamic list.
     * @param mats Additional Mat objects to be put back into the dynamic list.
     */
    template<typename... Mats>
    DT_VOID PutMats(Mat &mat, Mats &&... mats)
    {
        PutMats(mat);
        PutMats(mats...);
    }

    /**
     * @brief Puts all Mat objects in the MatFactory's dynamic list back for reuse.
     */
    DT_VOID PutAllMats();

    /**
     * @brief Clear all matrices from the factory.
     */
    DT_VOID Clear();

    /**
     * @brief Print information about matrices in the factory.
     */
    DT_VOID PrintInfo();

    /**
     * @brief Check the total memory used by matrices in the factory.
     */
    DT_VOID CheckTotalMemory();

private:
    /**
     * @brief Find a base matrix with the specified channel and file path.
     *
     * @param channel The channel of the base matrix.
     * @param file_path The file path of the base matrix.
     * 
     * @return The pointer to the base matrix if found, otherwise nullptr.
     */
    Mat* FindBaseMat(const DT_S32 channel, const std::string &file_path);

    /**
     * @brief Find a dynamic matrix based on the provided description.
     *
     * @param desc The description of the matrix.
     * 
     * @return The found dynamic matrix.
     */
    Mat FindDynamicMat(const MatDesc &desc);

    /**
     * @brief Create a matrix based on the provided description.
     *
     * @param desc The description of the matrix.
     * 
     * @return The created matrix.
     */
    Mat CreateMat(const MatDesc &desc);

    /**
     * @brief Create a matrix from a file.
     *
     * @param mat The matrix to create.
     * @param file The file path for loading the matrix.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateMat(Mat &mat, const std::string &file);

    /**
     * @brief Create a matrix with random data.
     *
     * @param mat The matrix to create.
     * @param min The minimum value for random data.
     * @param max The maximum value for random data.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateMat(Mat &mat, DT_F32 min, DT_F32 max);

    /**
     * @brief Create a matrix by deriving it from a base matrix.
     *
     * @param src The source matrix for derivation.
     * @param dst The destination matrix for the derived result.
     * @param alpha The alpha parameter for derivation.
     * @param beta The beta parameter for derivation.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateMat(const Mat &src, Mat &dst, DT_F32 alpha = 1.0f, DT_F32 beta = 0.0f);

    Context            *m_ctx;              /*!< The context associated with the matrices. */
    DT_S64             m_max_mem;           /*!< The maximum memory allowed for matrices in megabytes. */
    DT_S64             m_total_mem;         /*!< The total memory used by matrices in the factory. */
    DT_S64             m_rand_seed;         /*!< The seed for the random number generator. */
    std::mt19937_64    m_rand_engine;       /*!< The random number generator engine. */
    std::mutex         m_handle_lock;       /*!< Mutex for handling concurrent access to the factory. */
    std::list<MatInfo> m_dynamic_list;      /*!< List of dynamic matrices, deletes head matrix when memory exceeds the limit. */
    std::list<MatInfo> m_base_list;         /*!< List of constant matrices, keeps matrices of FILE type only. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_MAT_FACTORY_HPP__
