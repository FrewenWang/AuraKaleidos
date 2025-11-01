#include "host/include/fast_unit_test.hpp"

static FastParam::TupleTable g_fast_table_none
{
    // elem_type
    {
        ElemType::U8
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // threshold
    {
        40,
        60,
    },

    // nonmax suppresion
    {
        MI_TRUE,
    },

    // type
    {
        FastDetectorType::FAST_9_16,
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(feature2d, Fast, none)
{
    FastTest test(UnitTest::GetInstance()->GetContext(), g_fast_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}