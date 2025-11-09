#ifndef AURA_RUNTIME_ARRAY_MAT_HPP__
#define AURA_RUNTIME_ARRAY_MAT_HPP__

#include "aura/runtime/array/array.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup array Array
 *      @{
 *          @defgroup mat_class Mat Class
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup mat_class
 * @{
*/

/**
 * @brief Enumeration representing the border type.
 */
enum class BorderType
{
    /** 高斯模糊的边界处理：指定常量 **/
    CONSTANT    = 0, /*!< constant    border type(`iiiiii|abcdefgh|iiiiiii`) */
    /** 高斯模糊的边界处理：边界值的复制 **/
    REPLICATE,       /*!< replicate   border type(`aaaaaa|abcdefgh|hhhhhhh`) */
    /// 101镜面反射
    REFLECT_101,     /*!< reflect_101 border type(`gfedcb|abcdefgh|gfedcba`) */
};

/**
 * @brief Overloaded stream insertion operator to convert BorderType to a string for output.
 *
 * @param os The output stream.
 * @param border_type The BorderType enum value.
 *
 * @return Output stream with border type information.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, BorderType border_type)
{
    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            os << "Constant";
            break;
        }

        case BorderType::REPLICATE:
        {
            os << "Replicate";
            break;
        }

        case BorderType::REFLECT_101:
        {
            os << "Reflect_101";
            break;
        }

        default:
        {
            os << "undefined border type";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert BorderType to a string.
 *
 * @param type The BorderType enum value.
 *
 * @return The string representation of the BorderType.
 */
AURA_INLINE std::string BorderTypeToString(BorderType type)
{
    std::ostringstream ss;
    ss << type;
    return ss.str();
}

/**
 * @brief N-dimensional matirx class.
 *
 * The `Mat` Class represents a multi-dimensional dense numerical array, which can be used to store iauras, complex numbers
 * and so on. For data layout, 2-dimensional matrices are stored in row-major order, and 3-dimensional matrices are stored in
 * a channel-last order (e.g. RGB|RGB|RGB|...).
 *
 * The formula for accessing the k-th channel of the i-th row and j-th column in an RGB iaura matrix is as follows:
 * \f[M[i][j][k] = i \cdot \text{m_strides.m_width} + j \cdot \text{m_sizes.m_channel} + k\f]
 */
class AURA_EXPORTS Mat : public Array
{
public:
    /**
     * @brief Default constructor for creating an empty matrix.
     */
    Mat();

    /**
     * @brief Constructor for creating a matrix with specified properties and creating a new buffer.
     *
     * @param ctx Pointer to the context associated with the matrix.
     * @param elem_type Element type of the matrix.
     * @param sizes Size of the matrix in three dimensions (height, width, channels).
     * @param mem_type Memory type for matrix allocation (default is AURA_MEM_DEFAULT).
     * @param strides Strides for each dimension of the matrix (default is Sizes of 0, which means no padding).
     *
     * @note On the Android platform, AURA_MEM_DEFAULT is equivalent to AURA_MEM_DMA_BUF_HEAP, while on other platforms, it is equivalent to AURA_MEM_HEAP.
     */
    Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

    /**
     * @brief Constructor for creating a matrix with specified properties and existing buffer.
     *
     * @param ctx Pointer to the context associated with the matrix.
     * @param elem_type Element type of the matrix.
     * @param sizes Size of the matrix in three dimensions (height, width, channels).
     * @param buffer Buffer containing the data for the matrix.
     * @param strides Strides for each dimension of the matrix (default is Sizes of 0, which means no padding).
     */
    Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());

    /**
     * @brief Copy constructor for creating a matrix as a shallow copy of another matrix.
     *
     * The copy constructor creates a new matrix that shares the same data buffer with
     * the source matrix. It does not perform a deep copy of the data.
     *
     * @param mat Reference to the source matrix for creating a shallow copy.
     */
    Mat(const Mat &mat);

    /**
     * @brief Constructor for creating a matrix as a shallow copy of a region of another matrix.
     *
     * This constructor creates a new matrix that represents a specific region (ROI) of the source matrix.
     * The new matrix shares the same data buffer with the source matrix and does not perform a deep copy of the data.
     *
     * @param mat Reference to the source matrix for creating a shallow copy of the region.
     * @param roi The rectangular region of interest (ROI) within the source matrix.
     */
    Mat(const Mat &mat, const Rect &roi);

    /**
     * @brief Destructor for releasing resources associated with the matrix.
     */
    ~Mat();

    /**
     * @brief Release function for deallocating matrix data and resetting properties.
     */
    DT_VOID Release() override;

    /**
     * @brief Assignment operator for performing a shallow copy from another matrix.
     *
     * This assignment operator copies the structure of another matrix, including its size, element type, and strides,
     * but does not perform a deep copy of the data. Both matrices will share the same underlying data buffer.
     *
     * @param mat The source matrix from which to perform a shallow copy.
     *
     * @return A reference to the modified matrix after the assignment.
     */
    Mat& operator=(const Mat &mat);

    /**
    * @brief Creates a region of interest (ROI) of the current matrix without copying data.
    *
    * This method creates a new matrix representing a region of interest (ROI) within the current matrix.
    * The new matrix shares the same underlying data buffer with the original matrix, and no data is copied.
    *
    * @param roi The rectangular region of interest within the current matrix.
    *
    * @return A new matrix representing the specified ROI without copying data.
    */
    Mat Roi(const Rect &roi) const;

    /**
    * @brief Creates a matrix for the specified row span without copying data.
    *
    * The method makes a new matrix for the specified row span of the matrix.
    *
    * @param start An inclusive 0-based start index of the row span.
    * @param end  An exclusive 0-based ending index of the row span.
    *
    * @return A new matrix from specified row span without copying data.
    */
    Mat RowRange(DT_S32 start, DT_S32 end) const;

    /**
    * @brief Creates a matrix for the specified column span without copying data.
    *
    * The method makes a new matrix for the specified row span of the matrix.
    *
    * @param start An inclusive 0-based start index of the column span.
    * @param end  An exclusive 0-based ending index of the column span.
    *
    * @return A new matrix from specified column span without copying data.
    */
    Mat ColRange(DT_S32 start, DT_S32 end) const;

    /**
     * @brief Create a deep copy of the current matrix.
     *
     * @return New matrix containing a deep copy of the current matrix.
     */
    Mat Clone() const;

    /**
     * @brief Create a deep copy of a region of interest (ROI) from the current matrix.
     *
     * @param roi Region of interest (ROI) specifying the subregion of the matrix.
     * @param strides Strides for each dimension of the new matrix (default is Sizes of 0, which means no padding).
     *
     * @return New matrix containing a deep copy of the specified region of interest.
     */
    Mat Clone(const Rect &roi, const Sizes &strides = Sizes()) const;

    /**
     * @brief Copy the content of the current matrix to another matrix.
     *
     * @param mat Reference to the destination matrix for copying.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CopyTo(Mat &mat) const;

    /**
     * @brief Get a pointer to the raw data of the matrix.
     *
     * @return Pointer to the raw data of the matrix.
     */
    DT_VOID* GetData();

    /**
     * @brief Get a const pointer to the raw data of the matrix.
     *
     * @return Const pointer to the raw data of the matrix.
     */
    const DT_VOID* GetData() const;

    /**
     * @brief Check if the matrix is valid (i.e., allocated and properly initialized).
     *
     * @return True if the matrix is valid, false otherwise.
     */
    DT_BOOL IsValid() const override;

    /**
     * @brief Check if the data memory is continuous
     *
     * memory continuous condition is:
     *   mat is valid 
     *   mat create with no roi
     *   stride.m_width must be valid row bytes
     *
     * @return True if the data memory is continuous, false otherwise.
     */
    DT_BOOL IsContinuous() const;

    /**
     * @brief Get a pointer to the specified row of the matrix.
     *
     * @tparam Tp Type of the elements in the matrix.
     *
     * @param row Index of the row to access.
     *
     * @return Pointer to the specified row of the matrix.
     */
    template<typename Tp>
    Tp* Ptr(DT_S32 row);

    /**
     * @brief Get a const pointer to the specified row of the matrix.
     *
     * @tparam Tp Type of the elements in the matrix.
     *
     * @param row Index of the row to access.
     *
     * @return Const pointer to the specified row of the matrix.
     */
    template<typename Tp>
    const Tp* Ptr(DT_S32 row) const;

    /**
     * @brief Get a pointer to the specified row of the matrix with border handling.
     *
     * @tparam Tp Type of the elements in the matrix.
     *
     * @param row Index of the row to access.
     * @param border_type Border type for handling out-of-bounds access.
     * @param border_value Optional border value used for constant border type.
     *
     * @return Pointer to the specified row of the matrix with border handling.
     */
    template<typename Tp>
    Tp* Ptr(DT_S32 row, BorderType border_type, const Tp *border_value = DT_NULL);

    /**
     * @brief Accesses the raw data pointer for a specified row in the matrix for a specified border type.
     *
     * This method provides a pointer to the raw data in the specified row of the matrix.
     * If the specified row is outside the matrix boundaries, the specified values is used based on the border type.
     *
     * The supported border types are REPLICATE, REFLECT_101, CONSTANT. And each type has a corresponding specialized implementation.
     *
     * @tparam Tp The data type of the matrix elements.
     *
     * @param row The index of the row to access.
     * @param border_value The constant value used for out-of-bounds access (ignored for replicate and reflect_101 border handling).
     *
     * @return A pointer to the raw data in the specified row.
     */
    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE, Tp>::type* = DT_NULL>
    Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL);

    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE, Tp>::type* = DT_NULL>
    Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL);

    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE, Tp>::type* = DT_NULL>
    Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL);

    /**
     * @brief Get a constant pointer to the specified row of the matrix with border handling.
     *
     * @tparam Tp Type of the elements in the matrix.
     *
     * @param row Index of the row to access.
     * @param border_type Border type for handling out-of-bounds access.
     * @param border_value Optional border value used for constant border type.
     *
     * @return Pointer to the specified row of the matrix with border handling.
     */
    template<typename Tp>
    const Tp* Ptr(DT_S32 row, BorderType border_type, const Tp *border_value = DT_NULL) const;

    /**
     * @brief Accesses the constant raw data pointer for a specified row in the matrix for a specified border type.
     *
     * This method provides a pointer to the raw data in the specified row of the matrix.
     * If the specified row is outside the matrix boundaries, the specified values is used based on the border type.
     *
     * The supported border types are REPLICATE, REFLECT_101, CONSTANT. And each type has a corresponding specialized implementation.
     *
     * @tparam Tp The data type of the matrix elements.
     *
     * @param row The index of the row to access.
     * @param border_value The constant value used for out-of-bounds access (ignored for replicate and reflect_101 border handling).
     *
     * @return A pointer to the raw data in the specified row.
     */
    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE, Tp>::type* = DT_NULL>
    const Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL) const;

    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE, Tp>::type* = DT_NULL>
    const Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL) const;

    template<typename Tp, BorderType BORDER_TYPE,
             typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE, Tp>::type* = DT_NULL>
    const Tp* Ptr(DT_S32 row, const Tp *border_value = DT_NULL) const;

    /**
    * @brief Accesses the raw data for a specified element in the matrix.
    *
    * This method provides a reference to the raw data for the specified element in the matrix.
    *
    * @tparam Tp The data type of the matrix elements.
    *
    * @param h The row index of the element.
    * @param w The column index of the element.
    * @param c The channel index of the element (default is 0 for single-channel matrices).
    *
    * @return A reference to the specified element in the matrix.
    */
    template<typename Tp>
    Tp& At(DT_S32 h, DT_S32 w, DT_S32 c = 0);

    /**
    * @brief Accesses the constant raw data for a specified element in the matrix.
    *
    * This method provides a reference to the raw data for the specified element in the matrix.
    *
    * @tparam Tp The data type of the matrix elements.
    *
    * @param h The row index of the element.
    * @param w The column index of the element.
    * @param c The channel index of the element (default is 0 for single-channel matrices).
    *
    * @return A reference to the specified element in the matrix.
    */
    template<typename Tp>
    const Tp& At(DT_S32 h, DT_S32 w, DT_S32 c = 0) const;

    /**
    * @brief Displays the matrix contents.
    *
    * This method prints the matrix contents to the standard output.
    */
    DT_VOID Show() const override;

    /**
    * @brief Prints the matrix contents to a specified file in decimal or hexadecimal format.
    *
    * This method prints the matrix contents to a file specified by fname.
    *
    * @param mode The printing mode (default is decimal format, and other optional value is 16).
    * @param roi The rectangular region of interest to print (default is the entire matrix).
    * @param fname The name of the file to write the output (default is an empty string).
    */
    DT_VOID Print(DT_S32 mode = 10, const Rect &roi = Rect(), const std::string &fname = std::string()) const;

    /**
    * @brief Dumps the matrix contents to a file in binary format.
    *
    * This method dumps the matrix contents to a file specified by fname.
    *
    * @param fname The name of the file to write the matrix dump.
    */
    DT_VOID Dump(const std::string &fname) const override;

    /**
    * @brief Load the matrix contents from a file in binary format.
    *
    * This method load the matrix contents from a file specified by fname.
    *
    * @param fname The name of the file, read file to matrix contents.
    */
    Status Load(const std::string &fname);

    /**
    * @brief Changes the shape of mat with new size without copying data
    *
    * This method change the shape of mat with size param,
    *
    * @param size The new shape of the mat
    */
    Status Reshape(const Sizes3 &sizes);

private:
    /**
    * @brief Clears the content of the matrix.
    *
    * This method is used to reset properties of member vaiables.
    */
    DT_VOID Clear();
};

template<typename Tp>
Tp* Mat::Ptr(DT_S32 row)
{
    const DT_S32 off = row * m_strides.m_width;
    if (off < m_total_bytes)
    {
        return m_buffer.GetData<Tp*>(off);
    }
    return DT_NULL;
}

template<typename Tp>
const Tp* Mat::Ptr(DT_S32 row) const
{
    const DT_S32 off = m_strides.m_width * row;
    if (off < m_total_bytes)
    {
        return (const Tp*)(m_buffer.GetData<Tp*>(off));
    }
    return DT_NULL;
}

template<typename Tp>
Tp* Mat::Ptr(DT_S32 row, BorderType border_type, const Tp *border_value)
{
    DT_S32 height = m_sizes.m_height;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            return (row < 0 || row > height - 1) ? border_value : Ptr<Tp>(row);
        }

        case BorderType::REPLICATE:
        {
            row = Clamp(row, static_cast<DT_S32>(0), height - 1);
            return Ptr<Tp>(row);
        }

        case BorderType::REFLECT_101:
        {
            if (1 == height)
            {
                return Ptr<Tp>(0);
            }

            do
            {
                row = (row < 0) ? Abs(row) : ((height - 1) << 1) - row;
            } while ((DT_U32)row >= (DT_U32)height);

            return Ptr<Tp>(row);
        }

        default:
        {
            return DT_NULL;
        }
    }
}

template<typename Tp, BorderType BORDER_TYPE,
         typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE, Tp>::type*>
Tp* Mat::Ptr(DT_S32 row, const Tp *border_value)
{
    DT_S32 height = m_sizes.m_height;

    return (row < 0 || row > height - 1) ? border_value : Ptr<Tp>(row);
}

template<typename Tp, BorderType BORDER_TYPE,
         typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE, Tp>::type*>
Tp* Mat::Ptr(DT_S32 row, const Tp *border_value)
{
    AURA_UNUSED(border_value);

    DT_S32 height = m_sizes.m_height;

    row = Clamp(row, static_cast<DT_S32>(0), height - 1);
    return Ptr<Tp>(row);
}

template<typename Tp, BorderType BORDER_TYPE,
         typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE, Tp>::type*>
Tp* Mat::Ptr(DT_S32 row, const Tp *border_value)
{
    AURA_UNUSED(border_value);

    DT_S32 height = m_sizes.m_height;

    if (1 == height)
    {
        return Ptr<Tp>(0);
    }

    do
    {
        row = (row < 0) ? Abs(row) : ((height - 1) << 1) - row;
    } while ((DT_U32)row >= (DT_U32)height);

    return Ptr<Tp>(row);
}

template<typename Tp>
const Tp* Mat::Ptr(DT_S32 row, BorderType border_type, const Tp *border_value) const
{
    DT_S32 height = m_sizes.m_height;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            return (row < 0 || row > height - 1) ? border_value : Ptr<Tp>(row);
        }

        case BorderType::REPLICATE:
        {
            row = Clamp(row, static_cast<DT_S32>(0), height - 1);
            return Ptr<Tp>(row);
        }

        case BorderType::REFLECT_101:
        {
            if (1 == height)
            {
                return Ptr<Tp>(0);
            }

            do
            {
                row = (row < 0) ? Abs(row) : ((height - 1) << 1) - row;
            } while ((DT_U32)row >= (DT_U32)height);

            return Ptr<Tp>(row);
        }

        default:
        {
            return DT_NULL;
        }
    }
}


template<typename Tp, BorderType BORDER_TYPE,typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE, Tp>::type*>
const Tp* Mat::Ptr(DT_S32 row, const Tp *border_value) const
{
    DT_S32 height = m_sizes.m_height;

    return (row < 0 || row > height - 1) ? border_value : Ptr<Tp>(row);
}

template<typename Tp, BorderType BORDER_TYPE,
         typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE, Tp>::type*>
const Tp* Mat::Ptr(DT_S32 row, const Tp *border_value) const
{
    AURA_UNUSED(border_value);

    DT_S32 height = m_sizes.m_height;

    row = Clamp(row, static_cast<DT_S32>(0), height - 1);
    return Ptr<Tp>(row);
}

template<typename Tp, BorderType BORDER_TYPE,
         typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE, Tp>::type*>
const Tp* Mat::Ptr(DT_S32 row, const Tp *border_value) const
{
    AURA_UNUSED(border_value);

    DT_S32 height = m_sizes.m_height;

    if (1 == height)
    {
        return Ptr<Tp>(0);
    }

    do
    {
        row = (row < 0) ? Abs(row) : ((height - 1) << 1) - row;
    } while ((DT_U32)row >= (DT_U32)height);

    return Ptr<Tp>(row);
}

template<typename Tp>
Tp& Mat::At(DT_S32 h, DT_S32 w, DT_S32 c)
{
    return Ptr<Tp>(h)[w * m_sizes.m_channel + c];
}

template<typename Tp>
const Tp& Mat::At(DT_S32 h, DT_S32 w, DT_S32 c) const
{
    return Ptr<Tp>(h)[w * m_sizes.m_channel + c];
}

/*!
 * @brief Enumerates different iaura formats.
 */
enum class IauraFormat
{
    None    = 0, /*!< No specific format. */

    Gray,        /*!< Grayscale format. */

    // RGB one plane
    RGB,         /*!< Red, Green, Blue. */
    RGBA,        /*!< Red, Green, Blue, Alpha. */
    BGR,         /*!< Blue, Green, Red. */
    BGRA,        /*!< Blue, Green, Red, Alpha. */

    // YUV
    NV12,        /*!< YUV 4:2:0, NV12 format. */
    NV21,        /*!< YUV 4:2:0, NV21 format. */
    YU12,        /*!< YUV 4:2:0, YU12 format. */
    YV12,        /*!< YUV 4:2:0, YV12 format. */
    P010,        /*!< YUV 4:2:0, P010 format. */
    P016,        /*!< YUV 4:2:0, P016 format. */

    I422,        /*!< YUV 4:2:2, I422 format. */
    I444,        /*!< YUV 4:4:4, I444 format. */
};

/**
 * @brief Overloaded stream operator for IauraFormat.
 *
 * @param os           Output stream.
 * @param iaura_format IauraFormat type to be streamed.
 * 
 * @return Output stream with iaura_format information.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, IauraFormat iaura_format)
{
    switch (iaura_format)
    {
        case IauraFormat::None:
        {
            os << "None";
            break;
        }
        case IauraFormat::Gray:
        {
            os << "Gray";
            break;
        }
        case IauraFormat::RGB:
        {
            os << "RGB";
            break;
        }
        case IauraFormat::RGBA:
        {
            os << "RGBA";
            break;
        }
        case IauraFormat::BGR:
        {
            os << "BGR";
            break;
        }
        case IauraFormat::NV12:
        {
            os << "NV12";
            break;
        }
        case IauraFormat::NV21:
        {
            os << "NV21";
            break;
        }
        case IauraFormat::YU12:
        {
            os << "YU12";
            break;
        }
        case IauraFormat::YV12:
        {
            os << "YV12";
            break;
        }
        case IauraFormat::P010:
        {
            os << "P010";
            break;
        }
        case IauraFormat::P016:
        {
            os << "P016";
            break;
        }
        case IauraFormat::I422:
        {
            os << "I422";
            break;
        }
        case IauraFormat::I444:
        {
            os << "I444";
            break;
        }
        default:
        {
            os << "INVALID iaura format";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts IauraFormat to a string.
 *
 * @param iaura_fmt IauraFormat type to be converted.
 * 
 * @return String representation of the iaura_fmt type.
 */
AURA_INLINE std::string IauraFormatToString(IauraFormat iaura_format)
{
    std::ostringstream oss;
    oss << iaura_format;
    return oss.str();
}

/**
 * @brief Get the sizes and strides of individual planes for a given iaura format.
 *
 * @param fmt         The iaura format.
 * @param img_sizes   The sizes of the original iaura.
 * @param img_strides The strides of the original iaura.
 * @param mat_sizes   Output vector containing sizes for individual planes.
 * @param mat_strides Output vector containing strides for individual planes.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_EXPORTS Status GetIauraFormatSize(IauraFormat fmt, Sizes img_sizes, const Sizes &img_strides,
                                       std::vector<Sizes3> &mat_sizes, std::vector<Sizes> &mat_strides);

/**
 * @brief Get the ROIs (regions of interest) for individual planes of a given iaura format.
 *
 * @param fmt      The iaura format.
 * @param img_roi  The region of interest in the original iaura.
 * @param mat_rois Output vector containing ROIs for individual planes.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_EXPORTS Status GetIauraFormatRoi(IauraFormat fmt, const Rect &img_roi, std::vector<Rect> &mat_rois);

/**
 * @brief Read a BMP iaura from a file.
 *
 * This function reads a BMP iaura from the specified file and returns it as a Mat object.
 *
 * @param ctx      The pointer to the Context object.
 * @param fname    The file name of the BMP iaura to read.
 * @param mem_type The type of memory to use for storing the iaura. Default is AURA_MEM_DEFAULT.
 * @param strides  The strides for the iaura data. Default is an empty Sizes object.
 *
 * @return A Mat object containing the iaura data. An empty Mat will be returned if error occurs.
 */
AURA_EXPORTS Mat ReadBmp(Context *ctx, const std::string &fname, DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

/**
 * @brief Write a BMP iaura to a file.
 *
 * This function writes the provided Mat object to a BMP file.
 *
 * @param ctx   The pointer to the Context object.
 * @param mat   The Mat object containing the iaura data.
 * @param fname The file name to write the BMP iaura to.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_EXPORTS Status WriteBmp(Context *ctx, const Mat &mat, const std::string &fname);

/**
 * @brief Read a YUV iaura from a file.
 *
 * This function reads a YUV iaura from the specified file and returns a vector of Mat objects representing the individual planes.
 *
 * @param ctx      The pointer to the Context object.
 * @param fname    The file name of the YUV iaura to read.
 * @param fmt      The iaura format describing the YUV format.
 * @param sizes    The sizes of the iaura planes.
 * @param mem_type The type of memory to use for storing the iaura planes. Default is AURA_MEM_DEFAULT.
 * @param strides  The strides for the iaura data. Default is an empty Sizes object.
 *
 * @return A vector of Mat objects containing the iaura data for each plane. An empty vector will be returned if error occurs.
 *
 * @note Supported IauraFormat: NV12/NV21/YU12/YV12/P010/P016/I422/I444.
 */
AURA_EXPORTS std::vector<Mat> ReadYuv(Context *ctx, const std::string &fname, IauraFormat fmt, const Sizes &sizes,
                                      DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());

/**
 * @brief Write a YUV iaura to a file.
 *
 * This function writes the provided vector of Mat objects, representing the YUV planes, to a file.
 *
 * @param ctx   The pointer to the Context object.
 * @param mats  A vector of Mat objects containing the iaura data for each plane.
 * @param fmt   The iaura format describing the YUV format.
 * @param fname The file name to write the YUV iaura to.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * @note Supported IauraFormat: NV12/NV21/YU12/YV12/P010/P016/I422/I444.
 */
AURA_EXPORTS Status WriteYuv(Context *ctx, const std::vector<Mat> &mats, IauraFormat fmt, const std::string &fname);

/**
 * @brief Get a multiple planes YUV iaura from a mat and iauraformat
 *
 * This function get a YUV iaura from the iauraformt mat and returns a vector of Mat objects representing the individual planes.
 *
 * @param ctx          The pointer to the Context object.
 * @param mat          A Mat objects containing the yuv iaura data, memory must continuous, channel must be 1
 * @param fmt          The iaura format describing the YUV format.
 * @param is_deep_copy identification of return mats deep copy, DT_TRUE means deep copy, DT_FALSE means shallow copy.
 *
 * @return vector of Mat objects which store yuv iaura different plane.
 *
 * @note Supported IauraFormat: NV12/NV21/YU12/YV12/P010/P016/I422/I444.
 */
AURA_EXPORTS std::vector<Mat> GetYuvIaura(Context *ctx, const Mat &mat, IauraFormat fmt, DT_BOOL is_deep_copy = DT_FALSE);

/**
 * @}
*/

} // namespace aura

#endif // AURA_RUNTIME_ARRAY_MAT_HPP__