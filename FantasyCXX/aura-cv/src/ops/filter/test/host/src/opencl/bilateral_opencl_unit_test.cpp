#include "host/include/bilateral_unit_test.hpp"

static BilateralParam::TupleTable g_bilateral_table_cl {
    // element
    {
        ElemType::U8,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32,
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480,  2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // bilateral test param: {sigma_color, sigma_space, ksize}
    {
        {-1.f, -1.f, 3},
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(filter, Bilateral, opencl)
{
    BilateralTest test(UnitTest::GetInstance()->GetContext(), g_bilateral_table_cl);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}