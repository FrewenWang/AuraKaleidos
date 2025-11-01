#include "host/include/fast_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static FastParam::TupleTable g_fast_table_hvx
{
    // elem_type
    {
        ElemType::U8
    },

    // matrix size
    {
        {Sizes3(2161, 4095, 1), Sizes()},
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes(239,  384)},
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
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(feature2d, Fast, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    FastTest test(UnitTest::GetInstance()->GetContext(), g_fast_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}