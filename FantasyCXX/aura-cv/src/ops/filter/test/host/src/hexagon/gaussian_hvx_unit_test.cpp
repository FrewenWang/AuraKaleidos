#include "host/include/gaussian_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static GaussianParam::TupleTable g_gaussian_table_hvx {
    //elem type
    {
        ElemType::U8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32
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
        {3, 0.0f},
        {3, 1.3f},
        {5, 2.3f},
        {7, 3.3f},
        {9, 1.3f},
        {9, 3.3f},
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    {
        OpTarget::Hvx()
    }
};

NEW_TESTCASE(filter, Gaussian, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    GaussianTest test(UnitTest::GetInstance()->GetContext(), g_gaussian_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}