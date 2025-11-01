#include "host/include/resize_unit_test.hpp"

static ResizeParam::TupleTable g_resize_up_table_neon
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
        ElemType::F32,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16
#endif
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
        OpTarget::Neon()
    },
};

static ResizeParam::TupleTable g_resize_down_table_neon
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
        ElemType::F32,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16
#endif
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
        OpTarget::Neon()
    },
};

NEW_TESTCASE(resize, Resize, neon)
{
    ResizeTest test_up(UnitTest::GetInstance()->GetContext(), g_resize_up_table_neon);
    test_up.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    ResizeTest test_down(UnitTest::GetInstance()->GetContext(), g_resize_down_table_neon);
    test_down.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}