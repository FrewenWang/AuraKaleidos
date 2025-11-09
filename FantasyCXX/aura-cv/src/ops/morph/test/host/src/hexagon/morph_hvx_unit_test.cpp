#include "host/include/morph_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static MorphParam::TupleTable g_morph_table_hvx {
    // element type
    {
        ElemType::U8,
        ElemType::U16,
        ElemType::S16,
    },

    // matrix size
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

    // morph type
    {
        MorphType::ERODE,
        MorphType::DILATE
    },

    // morph shape
    {
        MorphShape::RECT,
        MorphShape::CROSS,
        MorphShape::ELLIPSE
    },

    // morph test param: {ksize, iterations}
    {
        {3, 1},
        {5, 1},
        {7, 1},
        {3, 2},
        {5, 2},
        {7, 2},
        {3, 3},
        {5, 3},
        {7, 3}
    },

    // target
    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(morph, MorphologyEx, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    MorphTest test(UnitTest::GetInstance()->GetContext(), g_morph_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}
