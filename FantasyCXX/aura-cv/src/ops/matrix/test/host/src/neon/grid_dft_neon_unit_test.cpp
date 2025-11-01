#include "host/include/grid_dft_idft_unit_test.hpp"

static GridDftParam::TupleTable g_grid_dft_table_neon
{
    // ElemType
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
        ElemType::F32,
    },
    // MatSizePair
    {
        {
            {Sizes3(2048, 2048, 1), Sizes()},
            {Sizes3(2048, 2048, 2), Sizes()},
        },
        {
            {Sizes3(512, 512, 1), Sizes()},
            {Sizes3(512, 512, 2), Sizes()},
        },
        {
            {Sizes3(576, 576, 1), Sizes(577, 2368 * 1)},
            {Sizes3(576, 576, 2), Sizes(577, 2368 * 2)},
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
        OpTarget::Neon()
    },
    //array type
    {
        ArrayType::MAT
    },
};

static GridIDftParam::TupleTable g_grid_idft_table_neon
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
        OpTarget::Neon()
    },
    //array type
    {
        ArrayType::MAT
    },
};

static GridIDftRealParam::TupleTable g_grid_idft_real_table_neon
{
    // ElemType
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
#if defined(AURA_ENABLE_NEON_FP16)
        ElemType::F16,
#endif
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
        8, 
        16,
        32, 
    },
    // impl
    {
        OpTarget::Neon()
    },
    //array type
    {
        ArrayType::MAT
    },
};

NEW_TESTCASE(matrix, GridDft, neon)
{
    MatrixGridDftTest matrix_grid_dft_test(UnitTest::GetInstance()->GetContext(), g_grid_dft_table_neon);
    matrix_grid_dft_test.RunTest(this);
}

NEW_TESTCASE(matrix, GridIDft, neon)
{
    MatrixGridIDftTest matrix_grid_idft_test(UnitTest::GetInstance()->GetContext(), g_grid_idft_table_neon);
    matrix_grid_idft_test.RunTest(this);
}

NEW_TESTCASE(matrix, GridIDftReal, neon)
{
    MatrixGridIDftRealTest matrix_grid_idft_real_test(UnitTest::GetInstance()->GetContext(), g_grid_idft_real_table_neon);
    matrix_grid_idft_real_test.RunTest(this);
}