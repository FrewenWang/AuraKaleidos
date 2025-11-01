#include "host/include/median_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static MedianParam::TupleTable g_median_table_hvx
{
    //elem type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
    },

    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // kernel size
    {
        3,
        5,
        7,
    },
    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(filter, Median, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    MedianTest test(UnitTest::GetInstance()->GetContext(), g_median_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}
