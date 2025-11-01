#include "host/include/guidefilter_unit_test.hpp"

static GuideFilterParam::TupleTable g_guidefilter_table_none {
    //elem type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32
    },

    // mat size
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

    // kernel size
    {
        3,
        5,
        7,
        9,
        11,
    },

    // eps
    {
        0.25,
        1.2
    },

    // method
    {
        GuideFilterType::NORMAL,
        GuideFilterType::FAST,
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(filter, GuideFilter, none)
{
    GuideFilterTest test(UnitTest::GetInstance()->GetContext(), g_guidefilter_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
