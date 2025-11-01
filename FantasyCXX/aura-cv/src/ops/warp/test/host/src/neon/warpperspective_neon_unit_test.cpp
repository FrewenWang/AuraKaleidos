#include "host/include/warpperspective_unit_test.hpp"

static WarpPerspectiveParam::TupleTable g_warpperspective_table_neon
{
    // elem_type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32,
    },
    // mat_size
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
    // interp_type
    {
        InterpType::NEAREST,
        InterpType::LINEAR,
        InterpType::CUBIC,
    },
    // border_type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101,
    },
    // target
    {
        OpTarget::Neon(),
    },
};

NEW_TESTCASE(warp, Warpperspective, neon)
{
    WarpPerspectiveTest test(UnitTest::GetInstance()->GetContext(), g_warpperspective_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
