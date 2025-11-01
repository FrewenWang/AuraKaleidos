#include "host/include/morph_unit_test.hpp"

static MorphParam::TupleTable g_morph_table_neon {
    // element type
    {
        ElemType::U8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32
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
        OpTarget::Neon()
    },
};

NEW_TESTCASE(morph, MorphologyEx, neon)
{
    MorphTest test(UnitTest::GetInstance()->GetContext(), g_morph_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
