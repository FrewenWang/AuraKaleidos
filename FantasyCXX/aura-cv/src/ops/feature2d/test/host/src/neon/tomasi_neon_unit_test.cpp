#include "host/include/tomasi_unit_test.hpp"

static TomasiParam::TupleTable g_tomasi_table_neon
{
    // elem_type
    {
        ElemType::U8,
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // param
    {
        {0, 0.05, 3., 3, 3, MI_TRUE, 0.04},
        {0, 0.05, 3., 5, 3, MI_FALSE, 0.04},
        {10000, 0.05, 5., 5, 3, MI_FALSE, 0.04},
    },

    // target
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(feature2d, GoodFeaturesToTrack, neon)
{
    TomasiTest test(UnitTest::GetInstance()->GetContext(), g_tomasi_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}