#include "host/include/summean_unit_test.hpp"

static SumMeanTestParam::TupleTable g_summean_table_cl
{
    // element type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
        ElemType::U32, ElemType::S32,
        ElemType::F16, ElemType::F32,
    },

    // MatSize
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // operator type
    {
        SumMeanOpType::SUM, SumMeanOpType::MEAN,
    },

    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(matrix, SumMean, opencl)
{
    SumMeanTest summean_test(UnitTest::GetInstance()->GetContext(), g_summean_table_cl);
    summean_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}