#include "host/include/dft_idft_unit_test.hpp"

static DftParam::TupleTable g_dft_table_opencl
{
    // MatElemPair
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
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
            {Sizes3(256, 512, 1), Sizes(256, 2052 * 1)},
            {Sizes3(256, 512, 2), Sizes(256, 2052 * 2)},
        },
        {
            {Sizes3(16, 16, 1), Sizes()},
            {Sizes3(16, 16, 2), Sizes()},
        },
    },
    // impl
    {
        OpTarget::Opencl(),
    },
};

static IDftParam::TupleTable g_idft_table_opencl
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
        {Sizes3(256,  512,  2), Sizes(256, 2052 * 2)},
        {Sizes3(16,   16,   2), Sizes()},
    },
    //dst channels
    {
        1,
        2,
    },
    // impl
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(matrix, Dft, opencl)
{
    MatrixDftTest matrix_dft_test(UnitTest::GetInstance()->GetContext(), g_dft_table_opencl);
    matrix_dft_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

NEW_TESTCASE(matrix, IDft, opencl)
{
    MatrixIDftTest matrix_dft_test(UnitTest::GetInstance()->GetContext(), g_idft_table_opencl);
    matrix_dft_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
