#include "host/include/rotate_unit_test.hpp"

static RotateParam::TupleTable g_rotate_table_neon
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
    // MatSize
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
    // rotate type
    {
        RotateType::ROTATE_90,
        RotateType::ROTATE_180,
        RotateType::ROTATE_270
    },
    // target
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(matrix, Rotate, neon)
{
    MatrixRotate matrix_rotate_test(UnitTest::GetInstance()->GetContext(), g_rotate_table_neon);
    matrix_rotate_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}