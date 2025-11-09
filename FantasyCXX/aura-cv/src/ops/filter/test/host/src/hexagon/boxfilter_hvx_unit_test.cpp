#include "host/include/boxfilter_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static BoxFilterParam::TupleTable g_boxfilter_table_hvx
{
    // elem_type
    {
        ElemType::U8,
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
    // kernel_size
    {
        3,
        5,
        7,
        9,
        11,
        51,
    },
    // ElemType
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101,
    },
    // OpTarget
    {
        OpTarget::Hvx(),
    },
};

NEW_TESTCASE(filter, Boxfilter, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    BoxFilterTest test(UnitTest::GetInstance()->GetContext(), g_boxfilter_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}