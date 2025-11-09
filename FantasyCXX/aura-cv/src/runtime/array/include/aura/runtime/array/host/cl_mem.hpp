#ifndef AURA_RUNTIME_ARRAY_HOST_CL_MEM_HPP__
#define AURA_RUNTIME_ARRAY_HOST_CL_MEM_HPP__

#include "aura/runtime/mat.h"
#include "aura/runtime/opencl/cl_runtime.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup array Array
 *      @{
 *          @defgroup clmem_class CLMem Class
 *      @}
 * @}
*/

namespace aura
{

/**
 * @addtogroup clmem_class
 * @{
*/

/**
 * @brief Enumeration class representing synchronization types for OpenCL memory operations.
 */
enum class CLMemSyncType
{
    INVALID = 0, /*!< An invalid synchronization type. */
    WRITE,       /*!< Synchronization for write operations on OpenCL memory. */
    READ         /*!< Synchronization for read operations on OpenCL memory. */
};

AURA_INLINE std::string GetCLMemSyncTypeToString(CLMemSyncType m_type)
{
    std::string cl_mem_sync_str = "INVALID";

    switch (m_type)
    {
        case CLMemSyncType::WRITE:
        {
            cl_mem_sync_str = "WRITE";
            break;
        }
        case CLMemSyncType::READ:
        {
            cl_mem_sync_str = "READ";
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_mem_sync_str;
}

AURA_INLINE std::string CLMemFlagToString(cl_mem_flags cl_flags)
{
    std::string cl_mem_flag_str = "INVALID";

    switch (cl_flags)
    {
        case CL_MEM_READ_ONLY:
        {
            cl_mem_flag_str = "CL_MEM_READ_ONLY";
            break;
        }
        case CL_MEM_WRITE_ONLY:
        {
            cl_mem_flag_str = "CL_MEM_WRITE_ONLY";
            break;
        }
        case CL_MEM_READ_WRITE:
        {
            cl_mem_flag_str = "CL_MEM_READ_WRITE";
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_mem_flag_str;
}

AURA_INLINE std::string CLChannelOrderToString(cl_channel_order cl_ch_order)
{
    std::string ch_order_str = "INVALID";

    switch (cl_ch_order)
    {
        case CL_R:
        {
            ch_order_str = "CL_R";
            break;
        }
        case CL_A:
        {
            ch_order_str = "CL_A";
            break;
        }
        case CL_RG:
        {
            ch_order_str = "CL_RG";
            break;
        }
        case CL_RA:
        {
            ch_order_str = "CL_RA";
            break;
        }
        case CL_RGB:
        {
            ch_order_str = "CL_RGB";
            break;
        }
        case CL_RGBA:
        {
            ch_order_str = "CL_RGBA";
            break;
        }
        case CL_BGRA:
        {
            ch_order_str = "CL_BGRA";
            break;
        }
        case CL_ARGB:
        {
            ch_order_str = "CL_ARGB";
            break;
        }
        default:
        {
            break;
        }
    }

    return ch_order_str;
}

/**
 * @brief Parameter class used to config the properties of the OpenCL's buffer or iaura object.
 */
class AURA_EXPORTS CLMemParam
{
public:
    /**
     * @brief Default constructor. Initializes with an invalid memory type.
     */
    CLMemParam() : m_type(CLMemType::INVALID)
    {
        memset(&m_param, 0, sizeof(m_param));
    }

    /**
     * @brief Constructor for buffer memory type.
     *
     * @param cl_flags OpenCL memory flags.
     */
    CLMemParam(cl_mem_flags cl_flags) : m_type(CLMemType::BUFFER)
    {
        m_param.buffer.cl_flags = cl_flags;
    }

    /**
     * @brief Constructor for 2D iaura memory type.
     *
     * @param cl_flags OpenCL memory flags.
     * @param cl_ch_order OpenCL iaura channel order.
     * @param is_norm Flag indicating normalization.
     */
    CLMemParam(cl_mem_flags cl_flags, cl_channel_order cl_ch_order, DT_BOOL is_norm = DT_FALSE)
               : m_type(CLMemType::IAURA2D)
    {
        m_param.iaura2d.cl_flags    = cl_flags;
        m_param.iaura2d.cl_ch_order = cl_ch_order;
        m_param.iaura2d.is_norm     = is_norm;
    }

    /**
     * @brief Constructor for 3D iaura memory type.
     *
     * @param cl_flags OpenCL memory flags.
     * @param cl_ch_order OpenCL iaura channel order.
     * @param depth Depth of the 3D iaura.
     * @param is_norm Flag indicating normalization.
     */
    CLMemParam(cl_mem_flags cl_flags, cl_channel_order cl_ch_order, DT_S32 depth, DT_BOOL is_norm = DT_FALSE)
               : m_type(CLMemType::IAURA3D)
    {
        m_param.iaura3d.cl_flags    = cl_flags;
        m_param.iaura3d.cl_ch_order = cl_ch_order;
        m_param.iaura3d.depth       = depth;
        m_param.iaura3d.is_norm     = is_norm;
    }

    /**
     * @brief Destructor. Resets the memory type and parameters.
     */
    ~CLMemParam()
    {
        m_type = CLMemType::INVALID;
        memset(&m_param, 0, sizeof(m_param));
    }

    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const CLMemParam &cl_mem_param)
    {
        os << "cl type:" << std::endl
           << "  |- type           : " << CLMemTypeToString(cl_mem_param.m_type) << std::endl;

        switch(cl_mem_param.m_type)
        {
            case CLMemType::BUFFER:
            {
                os << "  |- mem_flag       : " << CLMemFlagToString(cl_mem_param.m_param.buffer.cl_flags)  << std::endl;
                break;
            }
            case CLMemType::IAURA2D:
            {
                os << "  |- mem_flag       : " << CLMemFlagToString(cl_mem_param.m_param.iaura2d.cl_flags)  << std::endl
                   << "  |- cl_ch_order    : " << CLChannelOrderToString(cl_mem_param.m_param.iaura2d.cl_ch_order) << std::endl
                   << "  |- is_norm        : " << cl_mem_param.m_param.iaura2d.is_norm << std::endl;
                break;
            }
            case CLMemType::IAURA3D:
            {
                os << "  |- mem_flag       : " << CLMemFlagToString(cl_mem_param.m_param.iaura3d.cl_flags)  << std::endl
                   << "  |- cl_ch_order    : " << CLChannelOrderToString(cl_mem_param.m_param.iaura3d.cl_ch_order) << std::endl
                   << "  |- is_norm        : " << cl_mem_param.m_param.iaura3d.is_norm << std::endl
                   << "  |- depth          : " << cl_mem_param.m_param.iaura3d.depth << std::endl;
                break;
            }
            default:
            {
                break;
            }
        }

        return os;
    }

    DT_BOOL operator==(const CLMemParam &param) const
    {
        if (m_type != param.m_type)
        {
            return DT_FALSE;
        }

        switch (m_type)
        {
            case CLMemType::BUFFER:
            {
                return (m_param.buffer.cl_flags == param.m_param.buffer.cl_flags);
            }
            case CLMemType::IAURA2D:
            {
                return (m_param.iaura2d.cl_flags    == param.m_param.iaura2d.cl_flags &&
                        m_param.iaura2d.cl_ch_order == param.m_param.iaura2d.cl_ch_order &&
                        m_param.iaura2d.is_norm     == param.m_param.iaura2d.is_norm);
            }
            case CLMemType::IAURA3D:
            {
                return (m_param.iaura3d.cl_flags    == param.m_param.iaura3d.cl_flags &&
                        m_param.iaura3d.cl_ch_order == param.m_param.iaura3d.cl_ch_order &&
                        m_param.iaura3d.is_norm     == param.m_param.iaura3d.is_norm &&
                        m_param.iaura3d.depth       == param.m_param.iaura3d.depth);
            }
            case CLMemType::INVALID:
            {
                return DT_TRUE;
            }
        }

        return DT_FALSE;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

public:
    CLMemType m_type;

    union Param
    {
        struct BufferParam
        {
            cl_mem_flags cl_flags;
        } buffer;

        struct Iaura2DParam
        {
            cl_mem_flags     cl_flags;
            cl_channel_order cl_ch_order;
            DT_BOOL          is_norm;
        } iaura2d;

        struct Iaura3DParam
        {
            cl_mem_flags     cl_flags;
            cl_channel_order cl_ch_order;
            DT_BOOL          is_norm;
            size_t           depth;
        } iaura3d;
    } m_param;
};

/**
 * @brief OpenCL Memory (Buffer/Iaura2d/Iaura3d) class.
 *
 * This class can create a new OpenCL memory object, such as buffers, iaura2d and iaura3d. It can also initialize
 * the above memmory objects with the existing buffer, and that is to say that the existing `Mat` or `CLMem` object can be
 * used to initialize a new `CLMem` object.
 */
class AURA_EXPORTS CLMem : public Array
{
public:
    /**
     * @brief Default constructor for creating an empty CLMem object with no associated memory.
     */
    CLMem();

    /**
     * @brief Constructor for CLMem with specified parameters.
     *
     * Creates a CLMem object with the provided context, OpenCL memory parameters, element type,
     * dimensions, and optional strides.
     *
     * @param ctx The pointer to the Context object.
     * @param cl_param The OpenCL memory parameters.
     * @param elem_type The element type of the CLMem.
     * @param sizes The dimensions of the CLMem.
     * @param strides The strides of the CLMem, default is Sizes().
     */
    CLMem(Context *ctx, const CLMemParam &cl_param, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides = Sizes());

    /**
     * @brief Constructor for CLMem with specified buffer.
     *
     * Creates a CLMem object with the provided context, OpenCL memory parameters, element type,
     * dimensions, associated buffer, and optional strides.
     *
     * @param ctx The pointer to the Context object.
     * @param cl_param The OpenCL memory parameters.
     * @param elem_type The element type of the CLMem.
     * @param sizes The dimensions of the CLMem.
     * @param buffer The buffer to be associated with the CLMem.
     * @param strides The strides of the CLMem, default is Sizes().
     */
    CLMem(Context *ctx, const CLMemParam &cl_param, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());

    /**
     * @brief Copy constructor for CLMem with a shallow copy of another CLMem object.
     *
     * Creates a new CLMem object by copying the contents of an existing CLMem object.
     *
     * @param mat The CLMem object to be copied.
     */
    CLMem(const CLMem &mat);

    /**
     * @brief Assignment operator for CLMem with a shallow copy of another CLMem object.
     *
     * Assigns the contents of another CLMem object to this object.
     *
     * @param mat The CLMem object to be assigned.
     *
     * @return Reference to the assigned CLMem object.
     */
    CLMem& operator=(const CLMem &mat);

    /**
     * @brief Destructor for CLMem.
     *
     * Releases the resources associated with the CLMem object.
     */
    ~CLMem();

    /**
     * @brief Release the CLMem resources.
     *
     * Deallocates the OpenCL memory and releases associated resources.
     */
    DT_VOID Release() override;

    /**
     * @brief Create a CLMem object from an Array, such as `Mat` object or `CLMem` object.
     *
     * If the type of the input array is `CLMem`, input CLMem must match the declared CLMemParam, if matched, return the input CLMem,
     * otherwise, an empty CLMem object will be returned. (CLMem conversion is not supported for now.)
     * If the type of the input array is `Mat`, a new CLMem object is constructed with the buffer of the input array.
     *
     * @param ctx The pointer to the Context object.
     * @param array The input Array.
     * @param cl_param The OpenCL memory parameters.
     *
     * @return A new CLMem object created from the input Array.
     */
    static CLMem FromArray(Context *ctx, const Array &array, const CLMemParam &cl_param = CLMemParam());

    /**
     * @brief Check if the CLMem object is valid.
     *
     * Verifies whether the CLMem object is properly initialized and associated with valid OpenCL memory.
     *
     * @return DT_TRUE if the CLMem is valid, DT_FALSE otherwise.
     */
    DT_BOOL IsValid() const override;

    /**
     * @brief Display information about the CLMem object.
     *
     * Prints information such as OpenCL memory type, dimensions, element type, and associated buffer (if any).
     */
    DT_VOID Show() const override;

    /**
    * @brief Dumps the CLMem contents to a file in binary format.
    *
    * This method dumps the CLMem contents to a file specified by fname.
    *
    * @param fname The name of the file to write the CLMem dump.
    */
    DT_VOID Dump(const std::string &fname) const override;

    /**
     * @brief Synchronize the CLMem with a specified synchronization type.
     *
     * Ensures proper synchronization of the CLMem object with the specified synchronization type.
     * This function is crucial for maintaining data consistency across different OpenCL devices
     * or host memory.
     *
     * @param cl_sync_type The synchronization type.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status
     */
    Status Sync(CLMemSyncType cl_sync_type);

    /**
     * @brief use fname data to fill cl memory object.
     *
     * Ensures proper synchronization of the CLMem object with the specified synchronization type.
     * This function is crucial for maintaining data consistency across different OpenCL devices
     * or host memory.
     *
     * @param cl_sync_type The synchronization type.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status
     */
    Status Load(const std::string &fname);

    /**
     * @brief Get a pointer to the underlying OpenCL memory with a specified type.
     *
     * Provides a convenient way to obtain a typed pointer to the underlying OpenCL memory.
     *
     * @tparam Tp The desired type of the pointer.
     *
     * @return A pointer to the OpenCL memory with the specified type.
     */
    template<typename Tp>
    Tp* GetCLMemPtr() const
    {
        return reinterpret_cast<Tp*>(m_data);
    }

    /**
     * @brief Get a reference of the underlying OpenCL memory with a specified type.
     *
     * @tparam Tp The desired data type of the reference.
     *
     * @return A reference of the OpenCL memory with the specified type.
     */
    template<typename Tp>
    Tp& GetCLMemRef() const
    {
        return *(reinterpret_cast<Tp*>(m_data));
    }

    /**
    * @brief Get the OpenCL memory parameters associated with the operation.
    *
    * @return Constant reference to the CLMemParam object.
    */
    const CLMemParam& GetCLMemParam() const
    {
        return m_cl_param;
    }

    /**
     * @brief Get the number of channel for a specified OpenCL channel order.
     *
     * Determines the number of channel based on the given OpenCL channel order.
     *
     * @param cl_ch_order The OpenCL channel order.
     *
     * @return The number of channel.
     */
    static DT_S32 GetCLIauraChannelNum(cl_channel_order cl_ch_order);

    /**
     * @brief Get the OpenCL channel data type for a specified element type and normalization flag.
     *
     * Determines the OpenCL channel data type based on the element type and normalization flag.
     *
     * @param elem_type The element type.
     * @param is_norm Flag indicating whether normalization is applied.
     *
     * @return The OpenCL channel data type.
     */
    static cl_channel_type GetCLIauraChannelDataType(ElemType elem_type, DT_BOOL is_norm = DT_FALSE);

    /**
     * @brief Get the width of an OpenCL iaura with a specified element count and channel order.
     *
     * Determines the width of an OpenCL iaura based on the element count and channel order.
     *
     * @param elem_count The element count.
     * @param cl_ch_order The OpenCL channel order.
     *
     * @return The width of the OpenCL iaura.
     */
    static size_t GetCLIauraWidth(DT_S32 elem_count, cl_channel_order cl_ch_order);

private:
    /**
    * @brief Clears the content of the CLMem.
    *
    * This method is used to reset properties of member vaiables.
    */
    DT_VOID Clear();

    /**
     * @brief Initialize the CLMem object.
     *
     * Initializes the CLMem object based on its parameters and the member variable m_buffer.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status InitCLMem();

    /**
     * @brief Initialize the Buffer OpenCL memory of CLMem object.
     *
     * Initializes the CLMem object based on its parameters and the member variable m_buffer.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status InitCLBuffer();

    /**
     * @brief Initialize the Iaura2d OpenCL memory of CLMem object.
     *
     * Initializes the CLMem object based on its parameters and the member variable m_buffer.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status InitCLIaura2D();

    /**
     * @brief Initialize the Iaura3d OpenCL memory of CLMem object.
     *
     * Initializes the CLMem object based on its parameters and the member variable m_buffer.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status InitCLIaura3D();

    /**
     * @brief Create the OpenCL memory object.
     *
     * Creates the OpenCL memory object based on its initialized parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateCLMem();

    /**
     * @brief Create the Buffer OpenCL memory object.
     *
     * Creates the Buffer OpenCL memory object based on its initialized parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateCLBuffer();

    /**
     * @brief Create the Iaura2d OpenCL memory object.
     *
     * Creates the Iaura2d OpenCL memory object based on its initialized parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateCLIaura2D();

    /**
     * @brief Create the Iaura3d OpenCL memory object.
     *
     * Creates the Iaura3d OpenCL memory object based on its initialized parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CreateCLIaura3D();

    /**
     * @brief Synchronize data between the OpenCL devices and host.
     *
     * Synchronize data between the OpenCL memory object(m_data) and the host data(m_buffer.m_data).
     *
     * @param cl_sync_type The synchronization type.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status
     */
    Status EnqueuedData(CLMemSyncType cl_sync_type);

    /**
     * @brief Bind a buffer to the CLMem object.
     *
     * Associates a buffer with the CLMem object.
     *
     * @param buffer The buffer to be associated with the CLMem object.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status BindBuffer(const Buffer &buffer);

private:
    CLMemParam m_cl_param;                          /*!< OpenCL memory parameters for the CLMem. */
    CLMemSyncMethod m_cl_sync_method;               /*!< Synchronization method for the CLMem. */
    void *m_data;                                   /*!< Pointer the OpenCL memory object. */

    std::shared_ptr<CLRuntime> m_cl_rt;             /*!< Shared pointer to the OpenCL runtime. */
    std::shared_ptr<cl::CommandQueue> m_cl_cmd;     /*!< Shared pointer to the OpenCL command queue. */
};

/**
 * @}
*/

} // namespace aura

#endif // AURA_RUNTIME_ARRAY_HOST_CL_MEM_HPP__
