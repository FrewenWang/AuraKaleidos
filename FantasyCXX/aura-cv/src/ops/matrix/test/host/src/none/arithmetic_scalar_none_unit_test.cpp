#include "host/include/arithmetic_scalar_unit_test.hpp"

static ArithmScalarTestParam::TupleTable g_arithmetic_scalar_table
{
    // element type src
    {
        ElemType::U8,  ElemType::S8,  ElemType::U16,
        ElemType::S16, ElemType::S32, ElemType::F32,
    },

    // element type dst
    {
        ElemType::U8,  ElemType::S8,  ElemType::U16,
        ElemType::S16, ElemType::S32, ElemType::F32,
    },

    // MatSize
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480,  2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480,  2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // scalar
    {
        0.5, 1.0, 128.2, 512.3
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Arithmetic_with_scalar, none)
{
    ArithmScalarTest arithmetic_scalar_test(UnitTest::GetInstance()->GetContext(), g_arithmetic_scalar_table);
    arithmetic_scalar_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}