#include "host/include/fast_unit_test.hpp"

static FastParam::TupleTable g_fast_table_neon
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
        DT_TRUE,
    },

    // type
    {
        FastDetectorType::FAST_9_16,
    },

    // target
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(feature2d, Fast, neon)
{
    FastTest test(UnitTest::GetInstance()->GetContext(), g_fast_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}