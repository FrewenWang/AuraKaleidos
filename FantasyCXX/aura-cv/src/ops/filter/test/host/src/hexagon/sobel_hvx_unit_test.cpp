#include "host/include/sobel_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static SobelParam::TupleTable g_sobel_table_hvx {
    //elem type
    {
        {ElemType::U8,  ElemType::S16},
    },

    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480, 2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480, 2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // sobel test param: {dx, dy, ksize, scale}
    {
        {1, 0, -1, 1.f},
        {1, 0, -1, 2.f},
        {0, 1, -1, 1.f},
        {0, 1, -1, 3.f},
        {1, 0,  0, 1.f},
        {1, 0,  0, 2.f},
        {0, 1,  0, 1.f},
        {0, 1,  0, 2.f},
        {1, 0,  1, 1.f},
        {1, 0,  1, 2.f},
        {0, 1,  1, 1.f},
        {0, 1,  1, 2.f},
        {2, 2,  1, 1.f},
        {2, 2,  1, 2.f},
        {1, 2,  3, 1.f},
        {2, 1,  3, 2.f},
        {1, 2,  5, 1.f},
        {2, 1,  5, 2.f}
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

NEW_TESTCASE(filter, Sobel, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    SobelTest test(UnitTest::GetInstance()->GetContext(), g_sobel_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}