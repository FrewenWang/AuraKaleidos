#include "host/include/resize_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static ResizeParam::TupleTable g_resize_up_table_hvx
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
    },

    // MatSize
    {
        {Sizes3(512, 512, 1), Sizes()},
        {Sizes3(479, 639, 1), Sizes(480, 2560 * 1)},
        {Sizes3(239, 319, 1), Sizes()},

        {Sizes3(512, 512, 2), Sizes()},
        {Sizes3(479, 639, 2), Sizes(480, 2560 * 2)},
        {Sizes3(239, 319, 2), Sizes()},

        {Sizes3(512, 512, 3), Sizes()},
        {Sizes3(479, 639, 3), Sizes(480, 2560 * 3)},
        {Sizes3(239, 319, 3), Sizes()},
    },

    // scale x and y
    {
        {2.0, 2.0},
        {4.0, 4.0},
        {2.0, 4.0},
        {1.3, 2.1},
    },

    // interp type
    {
        InterpType::NEAREST,
        InterpType::LINEAR,
        InterpType::CUBIC,
        InterpType::AREA
    },

    // target
    {
        OpTarget::Hvx()
    },
};

static ResizeParam::TupleTable g_resize_down_table_hvx
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
    },

    // MatSize
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

    // scale x and y
    {
        {0.5,  0.5},
        {0.25, 0.25},
        {0.5,  0.25},
        {0.71, 0.53},
    },

    // interp type
    {
        InterpType::NEAREST,
        InterpType::LINEAR,
        InterpType::CUBIC,
        InterpType::AREA
    },

    // target
    {
        OpTarget::Hvx(MI_TRUE)
    },
};

NEW_TESTCASE(resize, Resize, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(HexagonPowerLevel::TURBO, MI_FALSE);

    ResizeTest test_up(UnitTest::GetInstance()->GetContext(), g_resize_up_table_hvx);
    test_up.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    ResizeTest test_down(UnitTest::GetInstance()->GetContext(), g_resize_down_table_hvx);
    test_down.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(HexagonPowerLevel::STANDBY, MI_FALSE);
}