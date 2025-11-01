#include "host/include/transpose_unit_test.hpp"

static TransposeParam::TupleTable g_transpose_table_none
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
        ElemType::U32, ElemType::S32,
        ElemType::F16, ElemType::F32,
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
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Transpose, none)
{
    MatrixTranspose matrix_transpose_test(UnitTest::GetInstance()->GetContext(), g_transpose_table_none);
    matrix_transpose_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}