#ifndef AURA_TOOLS_UNIT_TEST_TEST_PARAM_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_PARAM_HPP__

#include <tuple>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup unit_test Unit Test
 * @}
 */

/**
 * @addtogroup unit_test
 * @{
 */

/**
 * @brief Macro to count the number of arguments in a variadic macro.
 *
 * Usage: AURA_NARG(arg1, arg2, ..., argN)
 */
#if !defined(AURA_NARG)
#define AURA_NARG(...) _AURA_NARG_I(__VA_ARGS__, _AURA_RSEQ_N())
#define _AURA_NARG_I(...) _AURA_ARG_N(__VA_ARGS__)
#define _AURA_ARG_N(                                        \
    _1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9,  _10,       \
    _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,       \
    _21, _22, _23, _24, _25, _26, _27, _28, _29, _30,       \
    _31, _32, _33, _34, _35, _36, _37, _38, _39, _40,       \
    _41, _42, _43, _44, _45, _46, _47, _48, _49, _50,       \
    _51, _52, _53, _54, _55, _56, _57, _58, _59, _60,       \
    _61, _62, _63, MACRO_NAME, ...) MACRO_NAME

#define _AURA_RSEQ_N()                                      \
    63, 62, 61, 60, 59, 58, 57, 56,                         \
    55, 54, 53, 52, 51, 50, 49, 48,                         \
    47, 46, 45, 44, 43, 42, 41, 40,                         \
    39, 38, 37, 36, 35, 34, 33, 32,                         \
    31, 30, 29, 28, 27, 26, 25, 24,                         \
    23, 22, 21, 20, 19, 18, 17, 16,                         \
    15, 14, 13, 12, 11, 10, 9,  8,                          \
    7,  6,  5,  4,  3,  2,  1,  0
#endif // AURA_NARG

/******** function generator ********/


/**
 * @brief Macro to generate a variadic function name based on the number of arguments.
 *
 * Usage: AURA_TEST_PARAM_VFUNC(func, arg1, arg2, ..., argN)
 */
#define _AURA_TEST_PARAM_VFUNC_(name, n) name##_##n
#define _AURA_TEST_PARAM_VFUNC(name, n)  _AURA_TEST_PARAM_VFUNC_(name, n)
#define AURA_TEST_PARAM_VFUNC(func, ...) _AURA_TEST_PARAM_VFUNC(func, AURA_NARG(__VA_ARGS__))(__VA_ARGS__)

/******** test param ********/

#define AURA_MARCO_INVALID_PARAM static_assert(0, "struct member type and name must be paired");

/**
 * @brief Macro to raise a static assertion if the number of parameters is not even.
 *
 * Usage: AURA_TEST_PARAM_CHECK_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_TEST_PARAM_CHECK_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_CHECK_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_CHECK_SEQ_1(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_3(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_5(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_7(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_9(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_11(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_13(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_15(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_17(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_19(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_21(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_23(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_25(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_27(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_29(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_31(...) AURA_MARCO_INVALID_PARAM
#define AURA_TEST_PARAM_CHECK_SEQ_2(...)
#define AURA_TEST_PARAM_CHECK_SEQ_4(...)
#define AURA_TEST_PARAM_CHECK_SEQ_6(...)
#define AURA_TEST_PARAM_CHECK_SEQ_8(...)
#define AURA_TEST_PARAM_CHECK_SEQ_10(...)
#define AURA_TEST_PARAM_CHECK_SEQ_12(...)
#define AURA_TEST_PARAM_CHECK_SEQ_14(...)
#define AURA_TEST_PARAM_CHECK_SEQ_16(...)
#define AURA_TEST_PARAM_CHECK_SEQ_18(...)
#define AURA_TEST_PARAM_CHECK_SEQ_20(...)
#define AURA_TEST_PARAM_CHECK_SEQ_22(...)
#define AURA_TEST_PARAM_CHECK_SEQ_24(...)
#define AURA_TEST_PARAM_CHECK_SEQ_26(...)
#define AURA_TEST_PARAM_CHECK_SEQ_28(...)
#define AURA_TEST_PARAM_CHECK_SEQ_30(...)
#define AURA_TEST_PARAM_CHECK_SEQ_32(...)

/**
 * @brief Macro to generate a sequence of parameter types.
 *
 * Usage: AURA_TEST_PARAM_TYPE_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_TEST_PARAM_TYPE_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_TYPE_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_TYPE_SEQ_2(T0, N0) T0
#define AURA_TEST_PARAM_TYPE_SEQ_4(T0, N0, T1, N1) T0, T1
#define AURA_TEST_PARAM_TYPE_SEQ_6(T0, N0, T1, N1, T2, N2) T0, T1, T2
#define AURA_TEST_PARAM_TYPE_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) T0, T1, T2, T3
#define AURA_TEST_PARAM_TYPE_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) T0, T1, T2, T3, T4
#define AURA_TEST_PARAM_TYPE_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) T0, T1, T2, T3, T4, T5
#define AURA_TEST_PARAM_TYPE_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) T0, T1, T2, T3, T4, T5, T6
#define AURA_TEST_PARAM_TYPE_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) T0, T1, T2, T3, T4, T5, T6, T7
#define AURA_TEST_PARAM_TYPE_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) T0, T1, T2, T3, T4, T5, T6, T7, T8
#define AURA_TEST_PARAM_TYPE_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
#define AURA_TEST_PARAM_TYPE_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10
#define AURA_TEST_PARAM_TYPE_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11
#define AURA_TEST_PARAM_TYPE_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12
#define AURA_TEST_PARAM_TYPE_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13
#define AURA_TEST_PARAM_TYPE_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14
#define AURA_TEST_PARAM_TYPE_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15

/**
 * @brief Macro to generate a sequence of parameter names from a sequence of types and names.
 *
 * Usage: AURA_TEST_PARAM_NAME_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_TEST_PARAM_NAME_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_NAME_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_NAME_SEQ_2(T0, N0) N0
#define AURA_TEST_PARAM_NAME_SEQ_4(T0, N0, T1, N1) N0, N1
#define AURA_TEST_PARAM_NAME_SEQ_6(T0, N0, T1, N1, T2, N2) N0, N1, N2
#define AURA_TEST_PARAM_NAME_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) N0, N1, N2, N3
#define AURA_TEST_PARAM_NAME_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) N0, N1, N2, N3, N4
#define AURA_TEST_PARAM_NAME_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) N0, N1, N2, N3, N4, N5
#define AURA_TEST_PARAM_NAME_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) N0, N1, N2, N3, N4, N5, N6
#define AURA_TEST_PARAM_NAME_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) N0, N1, N2, N3, N4, N5, N6, N7
#define AURA_TEST_PARAM_NAME_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) N0, N1, N2, N3, N4, N5, N6, N7, N8
#define AURA_TEST_PARAM_NAME_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9
#define AURA_TEST_PARAM_NAME_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10
#define AURA_TEST_PARAM_NAME_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11
#define AURA_TEST_PARAM_NAME_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12
#define AURA_TEST_PARAM_NAME_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13
#define AURA_TEST_PARAM_NAME_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14
#define AURA_TEST_PARAM_NAME_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15

/**
 * @brief Macro to generate a sequence of type and name pairs with semicolons.
 *
 * Usage: AURA_TEST_PARAM_TYPE_AND_NAME_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_TYPE_AND_NAME_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_2(T0, N0) T0 N0;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_4(T0, N0, T1, N1) T0 N0; T1 N1;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_6(T0, N0, T1, N1, T2, N2) T0 N0; T1 N1; T2 N2;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) T0 N0; T1 N1; T2 N2; T3 N3;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10; T11 N11;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10; T11 N11; T12 N12;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10; T11 N11; T12 N12; T13 N13;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10; T11 N11; T12 N12; T13 N13; T14 N14;
#define AURA_TEST_PARAM_TYPE_AND_NAME_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) T0 N0; T1 N1; T2 N2; T3 N3; T4 N4; T5 N5; T6 N6; T7 N7; T8 N8; T9 N9; T10 N10; T11 N11; T12 N12; T13 N13; T14 N14; T15 N15;

/**
 * @brief Macro to generate a sequence of std::vector types.
 *
 * Usage: AURA_TEST_PARAM_VEC_TYPE_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_MARCO_VEC(TYPE) std::vector<TYPE>
#define AURA_TEST_PARAM_VEC_TYPE_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_VEC_TYPE_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_2(T0, N0) AURA_MARCO_VEC(T0)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_4(T0, N0, T1, N1) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_6(T0, N0, T1, N1, T2, N2) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10), AURA_MARCO_VEC(T11)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10), AURA_MARCO_VEC(T11), AURA_MARCO_VEC(T12)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10), AURA_MARCO_VEC(T11), AURA_MARCO_VEC(T12), AURA_MARCO_VEC(T13)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10), AURA_MARCO_VEC(T11), AURA_MARCO_VEC(T12), AURA_MARCO_VEC(T13), AURA_MARCO_VEC(T14)
#define AURA_TEST_PARAM_VEC_TYPE_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) AURA_MARCO_VEC(T0), AURA_MARCO_VEC(T1), AURA_MARCO_VEC(T2), AURA_MARCO_VEC(T3), AURA_MARCO_VEC(T4), AURA_MARCO_VEC(T5), AURA_MARCO_VEC(T6), AURA_MARCO_VEC(T7), AURA_MARCO_VEC(T8), AURA_MARCO_VEC(T9), AURA_MARCO_VEC(T10), AURA_MARCO_VEC(T11), AURA_MARCO_VEC(T12), AURA_MARCO_VEC(T13), AURA_MARCO_VEC(T14), AURA_MARCO_VEC(T15)

/**
 * @brief Macro to convert a variable name and its value to a string and output it.
 *
 * Usage: AURA_TEST_PARAM_NAME_TO_STR_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_MARCO_TO_STR(NAME) os << std::string(#NAME) << "(" << S.NAME << ")";
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_NAME_TO_STR_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_2(T0, N0) AURA_MARCO_TO_STR(N0)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_4(T0, N0, T1, N1) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_6(T0, N0, T1, N1, T2, N2) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10) AURA_MARCO_TO_STR(N11)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10) AURA_MARCO_TO_STR(N11) AURA_MARCO_TO_STR(N12)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10) AURA_MARCO_TO_STR(N11) AURA_MARCO_TO_STR(N12) AURA_MARCO_TO_STR(N13)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10) AURA_MARCO_TO_STR(N11) AURA_MARCO_TO_STR(N12) AURA_MARCO_TO_STR(N13) AURA_MARCO_TO_STR(N14)
#define AURA_TEST_PARAM_NAME_TO_STR_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) AURA_MARCO_TO_STR(N0) AURA_MARCO_TO_STR(N1) AURA_MARCO_TO_STR(N2) AURA_MARCO_TO_STR(N3) AURA_MARCO_TO_STR(N4) AURA_MARCO_TO_STR(N5) AURA_MARCO_TO_STR(N6) AURA_MARCO_TO_STR(N7) AURA_MARCO_TO_STR(N8) AURA_MARCO_TO_STR(N9) AURA_MARCO_TO_STR(N10) AURA_MARCO_TO_STR(N11) AURA_MARCO_TO_STR(N12) AURA_MARCO_TO_STR(N13) AURA_MARCO_TO_STR(N14) AURA_MARCO_TO_STR(N15)

/**
 * @brief Macro to retrieve a struct member by name.
 *
 * Usage: AURA_TEST_PARAM_MEMB_NAME_SEQ(T0, N0, T1, N1, ..., TN, NN)
 */
#define AURA_MARCO_MEMB(NAME) S.NAME
#define AURA_TEST_PARAM_MEMB_NAME_SEQ(...) AURA_TEST_PARAM_VFUNC(AURA_TEST_PARAM_MEMB_NAME_SEQ, __VA_ARGS__)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_2(T0, N0) AURA_MARCO_MEMB(N0)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_4(T0, N0, T1, N1) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_6(T0, N0, T1, N1, T2, N2) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_8(T0, N0, T1, N1, T2, N2, T3, N3) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_10(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_12(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_14(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_16(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_18(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_20(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_22(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_24(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10), AURA_MARCO_MEMB(N11)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_26(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10), AURA_MARCO_MEMB(N11), AURA_MARCO_MEMB(N12)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_28(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10), AURA_MARCO_MEMB(N11), AURA_MARCO_MEMB(N12), AURA_MARCO_MEMB(N13)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_30(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10), AURA_MARCO_MEMB(N11), AURA_MARCO_MEMB(N12), AURA_MARCO_MEMB(N13), AURA_MARCO_MEMB(N14)
#define AURA_TEST_PARAM_MEMB_NAME_SEQ_32(T0, N0, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5, T6, N6, T7, N7, T8, N8, T9, N9, T10, N10, T11, N11, T12, N12, T13, N13, T14, N14, T15, N15) AURA_MARCO_MEMB(N0), AURA_MARCO_MEMB(N1), AURA_MARCO_MEMB(N2), AURA_MARCO_MEMB(N3), AURA_MARCO_MEMB(N4), AURA_MARCO_MEMB(N5), AURA_MARCO_MEMB(N6), AURA_MARCO_MEMB(N7), AURA_MARCO_MEMB(N8), AURA_MARCO_MEMB(N9), AURA_MARCO_MEMB(N10), AURA_MARCO_MEMB(N11), AURA_MARCO_MEMB(N12), AURA_MARCO_MEMB(N13), AURA_MARCO_MEMB(N14), AURA_MARCO_MEMB(N15)

/**
 * @brief Overloaded output stream operator to print the elements of a vector.
 *
 * @tparam Tp The type of elements in the vector.
 *
 * @param os Output stream.
 * @param vec The vector to be printed.
 *
 * @return Output stream after printing.
 */
template<typename Tp>
AURA_INLINE std::ostream& operator<<(std::ostream &os, const std::vector<Tp> &vec)
{
    size_t vec_size = vec.size();
    os << "[";
    for (size_t i = 0; i < vec_size - 1; i++)
    {
        os << vec[i] << ", ";
    }

    if (vec_size > 0)
    {
        os << vec[vec_size - 1];
    }
    os << "]";

    return os;
}

/******** test param ********/

/**
 * @brief Macro for defining a test parameter structure.
 *
 * @param struct_name The name of the test parameter structure.
 * @param ... A list of parameter definitions in the form (type, name).
 * 
 * @code
 * Example:
 *     AURA_TEST_PARAM(MyTestParam, (int, value), (std::string, name))
 * @endcode
 */
#define AURA_TEST_PARAM(struct_name, ...)                                                                              \
    AURA_TEST_PARAM_CHECK_SEQ(__VA_ARGS__)                                                                             \
    struct struct_name                                                                                                 \
    {                                                                                                                  \
        using Tuple      = std::tuple<AURA_TEST_PARAM_TYPE_SEQ(__VA_ARGS__)>;                                          \
        using TupleTable = std::tuple<AURA_TEST_PARAM_VEC_TYPE_SEQ(__VA_ARGS__)>;                                      \
        AURA_TEST_PARAM_TYPE_AND_NAME_SEQ(__VA_ARGS__)                                                                 \
        friend std::ostream& operator<<(std::ostream &os, const struct_name &S)                                      \
        {                                                                                                              \
            AURA_TEST_PARAM_NAME_TO_STR_SEQ(__VA_ARGS__)                                                               \
            return os;                                                                                                 \
        }                                                                                                              \
        std::string ToString()                                                                                  \
        {                                                                                                              \
            std::ostringstream ss;                                                                                     \
            ss << *this;                                                                                               \
            return ss.str();                                                                                           \
        }                                                                                                              \
        struct_name()                                                                                                  \
        {                                                                                                              \
        }                                                                                                              \
        struct_name(Tuple &T)                                                                                          \
        {                                                                                                              \
            std::tie(AURA_TEST_PARAM_NAME_SEQ(__VA_ARGS__)) = T;                                                       \
        }                                                                                                              \
        struct_name Get(Tuple &T)                                                                                      \
        {                                                                                                              \
            return struct_name(T);                                                                                     \
        }                                                                                                              \
        Tuple Get(struct_name &S)                                                                                      \
        {                                                                                                              \
            return std::make_tuple(AURA_TEST_PARAM_MEMB_NAME_SEQ(__VA_ARGS__));                                        \
        }                                                                                                              \
    };

/**
 * @}
 */

#endif // AURA_TOOLS_UNIT_TEST_TEST_PARAM_HPP__
