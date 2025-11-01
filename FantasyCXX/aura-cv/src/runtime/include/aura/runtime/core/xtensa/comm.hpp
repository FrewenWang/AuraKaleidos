#ifndef AURA_RUNTIME_CORE_XTENSA_COMM_HPP__
#define AURA_RUNTIME_CORE_XTENSA_COMM_HPP__

#include "aura/runtime/core/xtensa/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"

#define AURA_XTENSA_LEN_128             (128)

#define AURA_VDSP_LOCAL_MEM_SIZE        (450)       // KB
#define AURA_VDSP_VLEN                  (128)       // B
#define AURA_VDSP_HVLEN                 (64)        // B

#define AURA_XTENSA_OK                  (0)         // everithing is ok
#define AURA_XTENSA_ERROR               (-1)        // unknown/unspecified error
#define AURA_XTENSA_IO_ERROR            (-2)        // io error
#define AURA_XTENSA_NO_MEM              (-3)        // insufficient memory
#define AURA_XTENSA_NULL_PTR            (-4)        // null pointer
#define AURA_XTENSA_BAD_ARG             (-5)        // arg/param is bad
#define AURA_XTENSA_BAD_OPT             (-6)        // bad operation

/**
 * @brief Invalid memory type.
*/
#define AURA_XTENSA_MEM_INVALID         (0)

/**
 * @brief Heap memory type.
*/
#define  AURA_XTENSA_MEM_HEAP           (1)

/**
 * @brief Returns with error handling, print an error log if the ret is ERROR.
 *
 * @param ret The ret to be checked.
 */
#if !defined(AURA_XTENSA_RETURN)
#  define AURA_XTENSA_RETURN(ret)                                                                      \
    do {                                                                                               \
        if (ret != AURA_XTENSA_OK)                                                                     \
        {                                                                                              \
            AURA_XTENSA_LOG("fail");                                                                   \
            return aura::xtensa::Status::ERROR;                                                        \
        }                                                                                              \
        return aura::xtensa::Status::OK;                                                               \
    } while (0)
#endif

/**
 * @brief Returns with error handling, print an error log if the status is ERROR.
 *
 * @param ret The status to be checked.
 */
#if !defined(AURA_STATUS_RETURN)
#  define AURA_STATUS_RETURN(ret)                                                                      \
    do {                                                                                               \
        if (aura::xtensa::Status::ERROR == ret)                                                        \
        {                                                                                              \
            AURA_XTENSA_LOG("fail");                                                                   \
        }                                                                                              \
        return ret;                                                                                    \
    } while (0)
#endif

#define AURA_XTENSA_LOG(format, ...) aura::xtensa::Print("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

namespace aura
{
namespace xtensa
{

using TileManager = AURA_VOID*;

/**
 * @brief Enumeration representing different border types.
 */
enum class BorderType
{
    CONSTANT    = 0, /*!< constant    border type(`iiiiii|abcdefgh|iiiiiii`) */
    REPLICATE,       /*!< replicate   border type(`aaaaaa|abcdefgh|hhhhhhh`) */
    REFLECT_101,     /*!< reflect_101 border type(`gfedcb|abcdefgh|gfedcba`) */
};

/**
 * @brief Enumeration representing different status.
 */
enum class Status
{
    OK    = 0,     /*!< Operation completed successfully. */
    ERROR = -1,    /*!< Operation encountered an error. */
    ABORT = -2,    /*!< Operation was aborted. */
};

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
 * @brief Enumeration representing different element types.
 */
enum class ElemType
{
    INVALID    = 0,
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    F32,
    F64,
    F16,
};

/**
 *
 * @brief Writes the C string pointed by format to the standard output (stdout).
 *        If format includes format specifiers (subsequences beginning with %),
 *        the additional arguments following format are formatted and inserted
 *        in the resulting string replacing their respective specifiers.
 *
 * @param format C string that contains the text to be written to stdout.
 *
 * @return On success, the total number of characters written is returned.
 *         If a writing error occurs, the error indicator (ferror) is set and a negative number is returned.
 *
 */
MI_S32 Print(const MI_CHAR *format, ...);

/**
 * @brief Invalidates the cache lines in the specified memory region.
 *
 * @param addr  Pointer to the start of the memory region.
 * @param size  The size of the memory region in bytes.
 */
AURA_VOID DCacheInvalidate(const AURA_VOID *addr, MI_U32 size);

/**
 * @brief Writes back the cache lines in the specified region.
 *
 * @param addr  Pointer to the start of the memory region.
 * @param size  The size of the memory region in bytes.
 */
AURA_VOID DCacheWriteback(const AURA_VOID *addr, MI_U32 size);

/**
 *
 * @brief Allocates a block of size bytes of memory, returning a pointer to the beginning of the block.
 *
 * @param size Size of the memory block, in bytes.
 *
 * @return On success, a pointer to the memory block allocated by the function. The type of this pointer
 *         is always void*, which can be cast to the desired type of data pointer in order to be dereferenceable.
 *         If the function failed to allocate the requested block of memory, a null pointer is returned.
 *
 */
AURA_VOID* Malloc(size_t size);

/**
 *
 * @brief A block of memory previously allocated by a call to malloc, calloc or realloc is deallocated,
 *        making it available again for further allocations.
 *
 * @param tile_num Pointer to a memory block previously allocated with malloc, calloc or realloc.
 *
 * @return AURA_VOID.
 *
 */
AURA_VOID Free(AURA_VOID *data);

/**
 *
 * @brief Copies the values of num bytes from the location pointed to by source directly
 *        to the memory block pointed to by destination.
 *
 * @param dst Pointer to the destination array where the content is to be copied, type-casted to a pointer of type void*.
 * @param src Pointer to the source of data to be copied, type-casted to a pointer of type const void*.
 * @param size  Number of bytes to copy.
 *
 * @return dst is returned.
 *
 */
AURA_VOID* Memcpy(AURA_VOID *dst, const AURA_VOID *src, size_t size);

/**
 *
 * @brief Sets the first num bytes of the block of memory pointed by ptr to the specified
 *        value (interpreted as an unsigned char).
 *
 * @param data   Pointer to the block of memory to fill.
 * @param value  Value to be set. The value is passed as an int, but the function fills
 *               the block of memory using the unsigned char conversion of this value.
 * @param size   Number of bytes to be set to the value.
 *
 * @return data is returned.
 *
 */
AURA_VOID* Memset(AURA_VOID *data, MI_S32 value, size_t size);

/**
 *
 * @brief Copies the values of num bytes from the location pointed by source to the memory block pointed by destination.
 *        Copying takes place as if an intermediate buffer were used, allowing the destination and source to overlap.
 *
 * @param dst  Pointer to the destination array where the content is to be copied, type-casted to a pointer of type void*.
 * @param src  Pointer to the source of data to be copied, type-casted to a pointer of type const void*.
 * @param size Number of bytes to copy.
 *
 * @return destination is returned.
 *
 */
AURA_VOID* Memmove(AURA_VOID *dst, const AURA_VOID *src, size_t size);

/**
 *
 * @brief Function to compares the string str1 to the string str2.
 *
 * @param str1 The string to be compared.
 * @param str2 The string to be compared.
 *
 * @return Returns an integral value indicating the relationship between the strings:
 *         |  <0  | the first character that does not match has a lower value in str1 than in str2      |
 *         |  =0  | the contents of both strings are equal                                              |
 *         |  >0  | the first character that does not match has a greater value in str1 than in str2    |
 *
 */
MI_S32 Strcmp(const MI_CHAR *str1, const MI_CHAR *str2);

/**
 *
 * @brief Function to returns the length of the C string str.
 *
 * @param str The pointer to string object.
 *
 * @return The length of string.
 *
 */
size_t Strlen(const MI_CHAR *str);

/**
 *
 * @brief Copies the C string pointed by source into the array pointed by destination,
 *        including the terminating null character (and stopping at that point).
 *
 * @param dst Pointer to the destination array where the content is to be copied.
 * @param src String to be copied.
 *
 * @return destination is returned.
 *
 */
MI_CHAR* Strcpy(MI_CHAR *dst, const MI_CHAR *src);


/**
 *
 * @brief Function to finds the first occurrence of the substring str2 in the string str1.
 *
 * @param str1 This is the main C string to be scanned.
 * @param str2 This is the small string to be searched with-in src1 string.
 *
 * @return This function returns a pointer to the first occurrence in src1
 *         of any of the entire sequence of characters specified in needle,
 *         or a null pointer if the sequence is not present in src1.
 *
 */
const MI_CHAR* Strstr(const MI_CHAR *str1, const MI_CHAR *str2);

} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_COMM_HPP__
