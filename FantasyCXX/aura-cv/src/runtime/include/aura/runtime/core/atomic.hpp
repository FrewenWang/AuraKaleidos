#ifndef AURA_RUNTIME_CORE_ATOMIC_HPP__
#define AURA_RUNTIME_CORE_ATOMIC_HPP__

#include "aura/runtime/core/types/built-in.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup atomic Runtime Core Atomic
 *      @}
 * @}
*/

/**
 * @addtogroup atomic
 * @{
*/

/**
 * @brief Atomic addition operation across various platforms.
 * 
 * @param addr Pointer to the augend.
 * @param delta Addend.
*/
#if defined(AURA_BUILD_HEXAGON)
#  if defined(__ATOMIC_ACQ_REL)
#    define AURA_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#  else // __ATOMIC_ACQ_REL
#    define AURA_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#  endif // __ATOMIC_ACQ_REL
#elif defined(AURA_BUILD_HOST) // AURA_BUILD_HOST
#  if (defined __GNUC__ || defined __clang__) && !defined (AURA_BUILD_XPLORER)
#    if defined __clang__ && __clang_major__ >= 3 && !defined AURA_BUILD_ANDROID && !defined __INTEL_COMPILER
#      if defined(__ATOMIC_ACQ_REL)
#        define AURA_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#      else // __ATOMIC_ACQ_REL
#        define AURA_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#      endif // __ATOMIC_ACQ_REL
#    else
#      if defined __ATOMIC_ACQ_REL && !defined __clang__
#        define AURA_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#      else
#        define AURA_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#      endif
#    endif
#  elif defined _MSC_VER && !defined RC_INVOKED
#    include <intrin.h>
#    define AURA_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#  else
template <typename Tp> AURA_INLINE Tp AURA_XADD(Tp *addr, Tp delta) { Tp tmp = *addr; *addr += delta; return tmp; }
#  endif
#endif // AURA_BUILD_HEXAGON

/**
 * @}
*/

#endif // AURA_RUNTIME_CORE_ATOMIC_HPP__