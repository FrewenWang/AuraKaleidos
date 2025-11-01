#include "host/include/grid_dft_idft_unit_test.hpp"

static GridDftParam::TupleTable g_grid_dft_table_cl
{
    // ElemType
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
            {Sizes3(512,  512,  1), Sizes(512, 2560 * 2)},
            {Sizes3(512,  512,  2), Sizes(512, 2560 * 2)},
        },
        {
            {Sizes3(544, 544, 1), Sizes(544, 3000 * 2)},
            {Sizes3(544, 544, 2), Sizes(544, 3000 * 2)},
        },
        {
            {Sizes3(1024,  1024,  1), Sizes()},
            {Sizes3(1024,  1024,  2), Sizes()},
        },
        {
            {Sizes3(2048, 2048, 1), Sizes()},
            {Sizes3(2048, 2048, 2), Sizes()},
        },
    },
    // grid length
    {
        4,
        8,
        16,
        32,
    },
    // impl
    {
        OpTarget::Opencl(),
    },
    //array type
    {
        ArrayType::MAT
    },
};

static GridIDftParam::TupleTable g_grid_idft_table_opencl
{
    // MatSize
    {
        {Sizes3(512, 512, 2), Sizes(512, 2560 * 2)},
        {Sizes3(544, 544, 2), Sizes(544, 3000 * 2)},
        {Sizes3(1024, 1024, 2), Sizes()},
        {Sizes3(2048, 2048, 2), Sizes()},
    },
    //grid length
    {
        4, 
        8, 
        16,
        32, 
    },
    // impl
    {
        OpTarget::Opencl()
    },
    //array type
    {
        ArrayType::MAT
    },
};

static GridIDftRealParam::TupleTable g_grid_idft_real_table_opencl
{
    // ElemType
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
            {Sizes3(512, 512, 2), Sizes(512, 2560 * 2)},
            {Sizes3(512, 512, 1), Sizes(512, 2560 * 2)},
        },
        {
            {Sizes3(544, 544, 2), Sizes(544, 3000 * 2)},
            {Sizes3(544, 544, 1), Sizes(544, 3000 * 2)},
        },
        {
            {Sizes3(1024, 1024, 2), Sizes()},
            {Sizes3(1024, 1024, 1), Sizes()},
        },
        {
            {Sizes3(2048, 2048, 2), Sizes()},
            {Sizes3(2048, 2048, 1), Sizes()},
        },
    },
    //grid length
    {
        4, 
        8, 
        16,
        32, 
    },
    // impl
    {
        OpTarget::Opencl()
    },
    //array type
    {
        ArrayType::MAT
    },
};

NEW_TESTCASE(matrix, GridDft, opencl)
{
    MatrixGridDftTest matrix_grid_dft_test(UnitTest::GetInstance()->GetContext(), g_grid_dft_table_cl);
    matrix_grid_dft_test.RunTest(this);
}

NEW_TESTCASE(matrix, GridIDft, opencl)
{
    MatrixGridIDftTest matrix_grid_idft_test(UnitTest::GetInstance()->GetContext(), g_grid_idft_table_opencl);
    matrix_grid_idft_test.RunTest(this);
}

NEW_TESTCASE(matrix, GridIDftReal, opencl)
{
    MatrixGridIDftRealTest matrix_grid_idft_real_test(UnitTest::GetInstance()->GetContext(), g_grid_idft_real_table_opencl);
    matrix_grid_idft_real_test.RunTest(this);
}
