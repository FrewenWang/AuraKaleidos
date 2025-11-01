#include "host/include/calc_hist_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static CalcHistParam::TupleTable g_hist_table_hvx
{
    // MatSize
    {
        {Sizes3(2048, 4095, 1), Sizes()},
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(479, 2560)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // param : elem type | channel | hist size | range | accumulate | use mask
    {
        {ElemType::U8 , 0, 256, {0, 256}, MI_FALSE, MI_FALSE},
        {ElemType::U8 , 0, 242, {2, 244}, MI_FALSE, MI_TRUE},
    },

    // target
    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(hist, calchist, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    CalcHistTest test(UnitTest::GetInstance()->GetContext(), g_hist_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}
