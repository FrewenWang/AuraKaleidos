#include "host/include/minmax_unit_test.hpp"

static MinMaxTestParam::TupleTable g_minmax_table_none
{
    // element type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
        ElemType::F16,
        ElemType::F32,
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

    // operator type
    {
        BinaryOpType::MIN, BinaryOpType::MAX,
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, MinMax, none)
{
    MinMaxTest minmax_test(UnitTest::GetInstance()->GetContext(), g_minmax_table_none);
    minmax_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}