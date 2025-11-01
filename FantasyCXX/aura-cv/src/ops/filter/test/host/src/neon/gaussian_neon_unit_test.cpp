#include "host/include/gaussian_unit_test.hpp"

static GaussianParam::TupleTable g_gaussian_table_neon {
    //elem type
    {
        ElemType::U8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32
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
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(filter, Gaussian, neon)
{
    GaussianTest test(UnitTest::GetInstance()->GetContext(), g_gaussian_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
