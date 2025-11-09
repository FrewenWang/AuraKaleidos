#include "host/include/summean_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static SumMeanTestParam::TupleTable g_summean_table_hvx
{
    // element type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
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
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(matrix, SumMean, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    SumMeanTest summean_test(UnitTest::GetInstance()->GetContext(), g_summean_table_hvx);
    summean_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}