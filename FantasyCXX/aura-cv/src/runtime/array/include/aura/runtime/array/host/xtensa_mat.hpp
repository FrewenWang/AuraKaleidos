#ifndef AURA_RUNTIME_ARRAY_HOST_XTENSA_MAT_HPP__
#define AURA_RUNTIME_ARRAY_HOST_XTENSA_MAT_HPP__

#include "aura/runtime/array/mat.hpp"
#include "aura/runtime/xtensa/host/xtensa_engine.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup array Array
 *      @{
 *          @defgroup xtensa_mat_class XtensaMat Class
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup xtensa_mat_class
 * @{
*/

/**
 * @brief Enumeration class representing synchronization types for Xtensa memory operations.
 */
enum class XtensaSyncType
{
    INVALID = 0, /*!< An invalid synchronization type. */
    WRITE,       /*!< Synchronization for write operations on Xtensa memory. */
    READ         /*!< Synchronization for read operations on Xtensa memory. */
};

AURA_INLINE std::string XtensaSyncTypeToString(XtensaSyncType m_type)
{
    std::string xtensa_sync_str = "INVALID";

    switch (m_type)
    {
        case XtensaSyncType::WRITE:
        {
            xtensa_sync_str = "WRITE";
            break;
        }
        case XtensaSyncType::READ:
        {
            xtensa_sync_str = "READ";
            break;
        }
        default:
        {
            break;
        }
    }

    return xtensa_sync_str;
}

/**
 * @brief Xtensa mat (Buffer) class.
 *
 * This class can create a new XtensaMat memory object, onlys suppose dma buffer. It can also initialize
 * the above memmory objects with the existing buffer, and that is to say that the existing `Mat` or `XtensaMat` object can be
 * used to initialize a new `XtensaMat` object.
 */
class AURA_EXPORTS XtensaMat : public Array
{
public:
    /**
     * @brief Default constructor for creating an empty xtensa object.
     */
    XtensaMat();

    /**
     * @brief Constructor for creating a xtensa object with specified properties and creating a new buffer.
     *
     * @param ctx Pointer to the context associated with the xtensa object.
     * @param elem_type Element type of the xtensa object.
     * @param sizes Size of the xtensa object in three dimensions (height, width, channels).
     * @param strides Strides for each dimension of the xtensa object (default is Sizes of 0, which means no padding).
     */
    XtensaMat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides = Sizes());

    /**
     * @brief Constructor for creating a xtensa object with specified properties and existing buffer.
     *
     * @param ctx Pointer to the context associated with the xtensa object.
     * @param elem_type Element type of the xtensa object.
     * @param sizes Size of the xtensa object in three dimensions (height, width, channels).
     * @param buffer Buffer containing the data for the xtensa object.
     * @param strides Strides for each dimension of the xtensa object (default is Sizes of 0, which means no padding).
     */
    XtensaMat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());

    /**
     * @brief Copy constructor for creating a xtensa object as a shallow copy of another xtensa object.
     *
     * The copy constructor creates a new xtensa object that shares the same data buffer with
     * the source xtensa object. It does not perform a deep copy of the data.
     *
     * @param mat Reference to the source xtensa object for creating a shallow copy.
     */
    XtensaMat(const XtensaMat &xtensa_mat);

    /**
     * @brief Destructor for releasing resources associated with the xtensa object.
     */
    ~XtensaMat();

    /**
     * @brief Release function for deallocating xtensa object data and resetting properties.
     */
    AURA_VOID Release() override;

    /**
     * @brief Assignment operator for performing a shallow copy from another xtensa object.
     *
     * This assignment operator copies the structure of another xtensa object, including its size, element type, and strides,
     * but does not perform a deep copy of the data. Both matrices will share the same underlying data buffer.
     *
     * @param xtensa_mat The source xtensa object from which to perform a shallow copy.
     *
     * @return A reference to the modified xtensa object after the assignment.
     */
    XtensaMat& operator=(const XtensaMat &xtensa_mat);

    /**
     * @brief Create a xtensa object from an Array, such as `Mat` object or `XtensaMat` object.
     *
     * If the type of the input array is `XtensaMat`, then return the input CLMem,
     * If the type of the input array is `Mat`, a new XtensaMat object is constructed with the buffer of the input array.
     *
     * @param ctx The pointer to the Context object.
     * @param array The input Array.
     * 
     * @return A new XtensaMat object created from the input Array.
     */
    static XtensaMat FromArray(Context *ctx, const Array &array);

    /**
     * @brief Check if the XtensaMat object is valid.
     *
     * Verifies whether the XtensaMat object is properly initialized and associated with valid XtensaMat memory.
     *
     * @return MI_TRUE if the XtensaMat is valid, MI_FALSE otherwise.
     */
    MI_BOOL IsValid() const override;

    /**
     * @brief Display information about the XtensaMat object.
     *
     * Prints information such as XtensaMat memory type, dimensions, element type, and associated buffer (if any).
     */
    AURA_VOID Show() const override;

    /**
    * @brief Dumps the XtensaMat contents to a file in binary format.
    *
    * This method dumps the XtensaMat contents to a file specified by fname.
    *
    * @param fname The name of the file to write the XtensaMat dump.
    */
    AURA_VOID Dump(const std::string &fname) const override;

    /**
     * @brief Synchronize the Xtensa with a specified synchronization type.
     *
     * Ensures proper synchronization of the Xtensa object with the specified synchronization type.
     * This function is crucial for maintaining data consistency across different OpenCL devices
     * or host memory.
     *
     * @param xtensa_sync_type The synchronization type.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status
     */
    Status Sync(XtensaSyncType xtensa_sync_type);

private:
    /**
    * @brief Clears the content of the xtensa object.
    *
    * This method is used to reset properties of member vaiables.
    */
    AURA_VOID Clear();

private:
    XtensaEngine *m_xtensa_engine;
    MI_BOOL m_is_external_buffer;
};

/**
 * @}
*/

} // namespace aura

#endif // #define AURA_RUNTIME_ARRAY_HOST_XTENSA_MAT_HPP__