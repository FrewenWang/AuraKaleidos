#include "host/include/pyrup_unit_test.hpp"
#include "aura/runtime/hexagon.h"


static PyrUpParam::TupleTable g_pyrup_table_hvx
{
    //elem type
    {
        ElemType::U8,
        ElemType::U16,
        ElemType::S16,
    },

    // input matrix size
    {
        {
            {Sizes3(1024, 2048, 1), Sizes()},
            {Sizes3(2048, 4096, 1), Sizes()},
        },

        {
            {Sizes3(239, 319, 1), Sizes(240, 1280 * 1)},
            {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
        },

        {
            {Sizes3(120, 160, 1), Sizes()},
            {Sizes3(240, 320, 1), Sizes()},
        },
    },

    {
        {5, 0.0f},
        {5, 3.3f},
    },

    {
        BorderType::REFLECT_101,
        BorderType::REPLICATE,
    },

    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(pyramid, PyrUp, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    PyrUpTest test(UnitTest::GetInstance()->GetContext(), g_pyrup_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}