#include "host/include/filter2d_unit_test.hpp"

static Filter2dParam::TupleTable g_filter2d_table_neon {
    {
        {ElemType::U8, ElemType::U8},
        {ElemType::U16, ElemType::U16},
        {ElemType::S16, ElemType::S16},
#if defined(AURA_ENABLE_NEON_FP16)
        {ElemType::F16, ElemType::F16},
#endif
        {ElemType::F32, ElemType::F32},
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
        7,
    },

    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101,
    },

    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(filter, Filter2d, neon)
{
    Filter2dTest test(UnitTest::GetInstance()->GetContext(), g_filter2d_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
