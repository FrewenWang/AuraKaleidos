#include "host/include/gemm_unit_test.hpp"

static GemmParam::TupleTable g_gemm_table_cl{
    // elem_type
    {
        ElemType::F32,
    },
    // size_param
    {
        {
            {Sizes3(575, 2055, 1), Sizes(575, 8448)},
            {Sizes3(2055, 575, 1), Sizes(2055, 2304)},
            {Sizes3(575, 575, 1),  Sizes(575,  2304)},
        },

        {
            {Sizes3(64, 1024, 1),  Sizes(64,   4096)},
            {Sizes3(1024, 512, 1), Sizes(1024, 2048)},
            {Sizes3(64, 512, 1),   Sizes(64,   2048)},
        },

        {
            {Sizes3(512, 1024, 1), Sizes(512,  4096)},
            {Sizes3(1024, 512, 1), Sizes(1024, 2048)},
            {Sizes3(512, 512, 1),  Sizes(512,  2048)},
        },

        {
            {Sizes3(1024, 1024, 1), Sizes(1024, 4096)},
            {Sizes3(1024, 1024, 1), Sizes(1024, 4096)},
            {Sizes3(1024, 1024, 1), Sizes(1024, 4096)},
        },

        {
            {Sizes3(479, 639, 1), Sizes(480, 2560)},
            {Sizes3(639, 127, 1), Sizes(640,  512)},
            {Sizes3(479, 127, 1), Sizes(480,  512)},
        },
    },
    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(matrix, Gemm, opencl)
{
    MatrixGemmTest matrix_gemm_test(UnitTest::GetInstance()->GetContext(), g_gemm_table_cl);
    matrix_gemm_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}