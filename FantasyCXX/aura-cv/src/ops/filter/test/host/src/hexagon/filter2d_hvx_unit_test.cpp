#include "host/include/filter2d_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static Filter2dParam::TupleTable g_filter2d_table_hvx
{
    //elem type
    {
        {ElemType::U8, ElemType::U8},
        {ElemType::U16, ElemType::U16},
        {ElemType::S16, ElemType::S16},
    },

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

    {
        3,
        5,
        7
    },

    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101,
    },

    {
        OpTarget::Hvx()
    }
};

NEW_TESTCASE(filter, Filter2d, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    Filter2dTest test(UnitTest::GetInstance()->GetContext(), g_filter2d_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}