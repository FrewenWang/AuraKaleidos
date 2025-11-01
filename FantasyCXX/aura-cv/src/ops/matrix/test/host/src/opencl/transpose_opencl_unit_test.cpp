#include "host/include/transpose_unit_test.hpp"

static TransposeParam::TupleTable g_transpose_table_cl
{
    // elem_type
    {
        aura::ElemType::U8,  aura::ElemType::S8,
        aura::ElemType::U16, aura::ElemType::S16,
        aura::ElemType::U32, aura::ElemType::S32,
        aura::ElemType::F16, aura::ElemType::F32,
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

        {Sizes3(1024, 2048, 4), Sizes()},
        {Sizes3(479,  639,  4), Sizes(480,  2560 * 4)},
        {Sizes3(239,  319,  4), Sizes()},
    },
    // target
    {
        aura::OpTarget::Opencl()
    },
};

NEW_TESTCASE(matrix, Transpose, opencl)
{
    MatrixTranspose matrix_transpose_test(aura::UnitTest::GetInstance()->GetContext(), g_transpose_table_cl);
    matrix_transpose_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}