#include "host/include/summean_unit_test.hpp"

static SumMeanTestParam::TupleTable g_summean_table_neon
{
    // element type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
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
        SumMeanOpType::SUM, SumMeanOpType::MEAN,
    },

    // target
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(matrix, SumMean, neon)
{
    SumMeanTest summean_test(UnitTest::GetInstance()->GetContext(), g_summean_table_neon);
    summean_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}