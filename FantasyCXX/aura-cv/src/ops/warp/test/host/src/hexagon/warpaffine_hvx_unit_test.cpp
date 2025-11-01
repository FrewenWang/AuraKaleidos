#include "host/include/warpaffine_unit_test.hpp"

static WarpAffineParam::TupleTable g_warpaffine_table_hvx
{
    // elem_type
    {
        ElemType::U8,
    },
    // mat_size
    {
        {Sizes3(2048, 3072, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },
    // interp_type
    {
        InterpType::NEAREST,
        InterpType::LINEAR,
    },
    // border_type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
    },
    // target
    {
        OpTarget::Hvx(),
    },
};

NEW_TESTCASE(warp, Warpaffine, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    WarpAffineTest test(UnitTest::GetInstance()->GetContext(), g_warpaffine_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}