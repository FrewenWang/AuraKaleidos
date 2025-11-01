#include "host/include/dft_idft_unit_test.hpp"

static DftParam::TupleTable g_dft_table_none
{
    // ElemType
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
        ElemType::F16,
        ElemType::F32,
    },
    // MatSizePair
    {
        {
            {Sizes3(2048, 2048, 1), Sizes()},
            {Sizes3(2048, 2048, 2), Sizes()},
        },
        {
            {Sizes3(512,  639,  1), Sizes(512, 2560 * 2)},
            {Sizes3(512,  639,  2), Sizes(512, 2560 * 2)},
        },
        {
            {Sizes3(256,  319,  1), Sizes()},
            {Sizes3(256,  319,  2), Sizes()},
        },
        {
            {Sizes3(239,  256,  1), Sizes()},
            {Sizes3(239,  256,  2), Sizes()},
        },
        {
            {Sizes3(1, 64, 1), Sizes()},
            {Sizes3(1, 64, 2), Sizes()},
        },
        {
            {Sizes3(1, 63, 1), Sizes()},
            {Sizes3(1, 63, 2), Sizes()},
        },
        {
            {Sizes3(2, 64, 1), Sizes()},
            {Sizes3(2, 64, 2), Sizes()},
        },
        {
            {Sizes3(2, 63, 1), Sizes()},
            {Sizes3(2, 63, 2), Sizes()},
        },

        {
            {Sizes3(64, 1, 1), Sizes()},
            {Sizes3(64, 1, 2), Sizes()},
        },
        {
            {Sizes3(63, 1, 1), Sizes()},
            {Sizes3(63, 1, 2), Sizes()},
        },
        {
            {Sizes3(64, 2, 1), Sizes()},
            {Sizes3(64, 2, 2), Sizes()},
        },
        {
            {Sizes3(63, 2, 1), Sizes()},
            {Sizes3(63, 2, 2), Sizes()},
        },
    },
    // impl
    {
        OpTarget::None()
    },
};

static IDftParam::TupleTable g_idft_table_none
{
    // ElemType
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
        ElemType::F16,
        ElemType::F32,
    },
    // MatSize
    {
        {Sizes3(2048, 2048, 2), Sizes()},
        {Sizes3(1021, 1031, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480,  2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},
        {Sizes3(1, 64, 2), Sizes()},
        {Sizes3(1, 63, 2), Sizes()},
        {Sizes3(2, 64, 2), Sizes()},
        {Sizes3(2, 63, 2), Sizes()},
        {Sizes3(64, 1, 2), Sizes()},
        {Sizes3(63, 1, 2), Sizes()},
        {Sizes3(64, 2, 2), Sizes()},
        {Sizes3(63, 2, 2), Sizes()},
    },
    // dst channels
    {
        1,
        2,
    },
    // impl
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Dft, none)
{
    MatrixDftTest matrix_dft_test(UnitTest::GetInstance()->GetContext(), g_dft_table_none);
    matrix_dft_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

NEW_TESTCASE(matrix, IDft, none)
{
    MatrixIDftTest matrix_idft_test(UnitTest::GetInstance()->GetContext(), g_idft_table_none);
    matrix_idft_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}