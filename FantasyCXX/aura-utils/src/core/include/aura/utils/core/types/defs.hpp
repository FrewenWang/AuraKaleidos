#pragma once

// c++ version check
// #if defined(AURA_CXX_STANDARD_11)
// #  if (__cplusplus < 201103L) && (defined(_MSC_VER) && _MSC_VER < 1900)
// #    error "C++ versions less than C++11 are not supported."
// #  endif // (__cplusplus < 201103L) && (defined(_MSC_VER) && _MSC_VER < 1900)
// #else
// #  error "undefined c++ standard."
// #endif // AURA_CXX_STANDARD_11

// visibility
// define AURA_EXPORTS __attribute__((dllexport)) 是一个宏定义，主要用于跨平台动态链接库（DLL/SO）开发中，
// 标记需要导出的函数、类或变量，使其可以被外部程序调用。具体作用如下：
// Windows（MSVC编译器）：通常使用 __declspec(dllexport)
// Linux/macOS（GCC/Clang编译器）: 使用 __attribute__((visibility("default")))
// __attribute__((dllexport))（某些工具链兼容此语法）
#if (defined(AURA_BUILD_WINDOWNS) || defined(__CYGWIN__))
#  if defined(AURA_API_EXPORTS)
#    if defined(__GNUC__)
#      define AURA_EXPORTS __attribute__((dllexport))
#    else
#      define AURA_EXPORTS __declspec(dllexport)
#    endif // AURA_API_EXPORTS
#  else
#    define AURA_EXPORTS
#  endif // AURA_API_EXPORTS
#else
#  if (__GNUC__ >= 4) && defined(AURA_API_EXPORTS)
#    define AURA_EXPORTS __attribute__((visibility("default")))
#  else
#    define AURA_EXPORTS
#  endif // (__GNUC__ >= 4) && defined(AURA_API_EXPORTS)
#endif // (defined(AURA_BUILD_WIN) || defined(__CYGWIN__))

// inline
#if !defined(AURA_INLINE)
#  if defined(__cplusplus)
#    define AURA_INLINE static inline
#  elif defined(_MSC_VER)
#    define AURA_INLINE __inline
#  else
#    define AURA_INLINE static
#  endif // __cplusplus
#endif // AURA_INLINE

// no static inline
#if !defined(AURA_NO_STATIC_INLINE)
#  if defined(__cplusplus)
#    define AURA_NO_STATIC_INLINE inline
#  elif defined(_MSC_VER)
#    define AURA_NO_STATIC_INLINE __inline
#  else
#    define AURA_NO_STATIC_INLINE
#  endif // __cplusplus
#endif // AURA_NO_STATIC_INLINE

// always inline
#if !defined(AURA_ALWAYS_INLINE)
#  if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define AURA_ALWAYS_INLINE inline __attribute__((always_inline))
#  elif defined(_MSC_VER)
#    define AURA_ALWAYS_INLINE __forceinline
#  else
#    define AURA_ALWAYS_INLINE inline
#  endif // _MSC_VER
#endif // AURA_ALWAYS_INLINE

// always inline
#if !defined(AURA_NO_INLINE)
#  define AURA_NO_INLINE __attribute__((noinline))
#endif // AURA_NO_INLINE

// noreturn, unused and unused result
#if defined(__clang__)
#  if __has_feature(cxx_attributes)
#    define AURA_NORETURN [[noreturn]]
#  else
#    define AURA_NORETURN __attribute__((noreturn))
#  endif // __has_feature(cxx_attributes)
#  define AURA_UNUSED_RESULT __attribute__((warn_unused_result))
#elif defined(__GNUC__)
#  define AURA_NORETURN __attribute__((noreturn))
#  define AURA_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#  define AURA_NORETURN
#  define AURA_UNUSED_RESULT
#endif // __clang__

/**
 * @brief 抑制类的拷贝构造函数。和赋值运算符
 */
#if !defined(AURA_DISABLE_COPY_ASSIGN)
#define AURA_DISABLE_COPY_ASSIGN(CLASSNAME)                                                                            \
    CLASSNAME(const CLASSNAME &)            = delete;                                                                  \
    CLASSNAME &operator=(const CLASSNAME &) = delete;                                                                  \
    CLASSNAME(CLASSNAME &&)                 = delete;                                                                  \
    CLASSNAME &operator=(CLASSNAME &&)      = delete;
#endif // AURA_DISABLE_COPY_ASSIGN

#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
#  define AURA_ENABLE_NEON
#  if defined(AURA_ARM82) && defined(__aarch64__)
#    define AURA_ENABLE_NEON_FP16
#  endif // AURA_ENABLE_NEON
#endif // (defined(__ARM_NEON) || defined(__ARM_NEON__))

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup defs Runtime Core Defs
 *      @}
 * @}
*/

/**
 * @addtogroup defs
 * @{
*/

namespace AURA_internal
{
/**
 * 用于忽略未使用的变量并禁止编译器警告的实用程序函数。
 * @brief Utility function to ignore unused variables and suppress compiler warnings.
 *
 * @tparam T Type of the variable.
 *
 * @param[in] var Variable to be ignored.
 *
 * @ingroup core
 */
template<typename T>
void ignore_unused_variable(const T &) {
}
}

/**
 * @brief Macro to mark a variable as unused.
 *
 * This macro uses the `ignore_unused_variable` utility function to suppress compiler warnings about unused variables.
 *
 * @param[in] var Variable to be marked as unused.
 */
#define AURA_UNUSED(var) AURA_internal::ignore_unused_variable(var)

/**
 * @brief Macro to align a value to a specified boundary.
 *
 * This macro performs alignment by rounding up to the nearest multiple of the specified alignment.
 *
 * @param[in] x Value to be aligned.
 * @param[in] a Alignment boundary.
 *
 * @return Aligned value.
 */
#define AURA_ALIGN(x, a)    (((x) + (a) - 1) / (a) * (a))

/**
 * @brief Macro to define a type alias for a function pointer.
 *
 * This macro defines a type alias for an OpenCL function pointer to simplify the declaration
 * and usage of function pointers.
 *
 * @param func The name of the function pointer.
 */
#if !defined(AURA_API_DEF)
#  define AURA_API_DEF(func)    using func##API
#endif // AURA_API_DEF

/**
 * @brief Macro to declare and initialize a function pointer variable.
 *
 * This macro is used to declare and initialize a function pointer variable.
 * It helps in the dynamic loading of OpenCL functions.
 *
 * @param func The name of the function pointer variable.
 */
#if !defined(AURA_API_PTR)
#  define AURA_API_PTR(func)    func##API func = MI_NULL
#endif // AURA_API_PTR

/**
 * @brief Macro to load a symbol from a dynamic library handle
 *
 * This macro is designed to be used within a `do-while(0)` loop in a constructor to
 * load a function symbol from a given dynamic library handle. If the symbol is found,
 * it is assigned to the provided function pointer variable. If not found, an error
 * message is printed.
 *
 * @param handle  The dynamic library handle from which to load the symbol
 * @param func    The name of the function symbol to load and the variable to assign it to
 */
#if !defined(AURA_DLSYM_API)
#  define AURA_DLSYM_API(handle, func)                                                  \
    {                                                                                   \
        AURA_VOID *ptr = dlsym(handle, #func);                                            \
        if (ptr)                                                                        \
        {                                                                               \
            func = reinterpret_cast<func##API>(ptr);                                    \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            std::string info = std::string("dlsym ") + #func + std::string(" failed");  \
            info += std::string("(") + std::string(dlerror()) + std::string(")");       \
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());                                \
            break;                                                                      \
        }                                                                               \
    }
#endif // AURA_DLSYM_API

/**
 * @brief Macro to load a symbol from a adreno gpu platform dynamic library handle
 *
 * This macro is designed to be used within a `do-while(0)` loop in a constructor to
 * load a function symbol from a given dynamic library handle. If the symbol is found,
 * it is assigned to the provided function pointer variable. If not found, an error
 * message is printed.
 *
 * @param handle  The dynamic library handle from which to load the symbol
 * @param func    The name of the function symbol to load and the variable to assign it to
 */
#if !defined(AURA_DLSYM_API_ADRENO)
#  define AURA_DLSYM_API_ADRENO(handle, func)                                           \
    {                                                                                   \
        AURA_VOID *ptr = dlsym(handle, #func);                                            \
        if (ptr)                                                                        \
        {                                                                               \
            func = reinterpret_cast<func##API>(ptr);                                    \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            std::string info = std::string("dlsym ") + #func + std::string(" failed");  \
            info += std::string("(") + std::string(dlerror()) + std::string(")");       \
            info += std::string(", Note: ignore this error message if not adreno gpu"); \
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());                                \
            break;                                                                      \
        }                                                                               \
    }
#endif // AURA_DLSYM_API_ADRENO

/**
 * @brief Macro to load a symbol from a mali gpu platform dynamic library handle
 *
 * This macro is designed to be used within a `do-while(0)` loop in a constructor to
 * load a function symbol from a given dynamic library handle. If the symbol is found,
 * it is assigned to the provided function pointer variable. If not found, an error
 * message is printed.
 *
 * @param handle  The dynamic library handle from which to load the symbol
 * @param func    The name of the function symbol to load and the variable to assign it to
 */
#if !defined(AURA_DLSYM_API_MALI)
#  define AURA_DLSYM_API_MALI(handle, func)                                             \
    {                                                                                   \
        AURA_VOID *ptr = dlsym(handle, #func);                                            \
        if (ptr)                                                                        \
        {                                                                               \
            func = reinterpret_cast<func##API>(ptr);                                    \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            std::string info = std::string("dlsym ") + #func + std::string(" failed");  \
            info += std::string("(") + std::string(dlerror()) + std::string(")");       \
            info += std::string(", Note: ignore this error message if not mali gpu");   \
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());                                \
            break;                                                                      \
        }                                                                               \
    }
#endif // AURA_DLSYM_API_MALI