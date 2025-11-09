#include "host/include/mul_spectrums_unit_test.hpp"

static MulSpectrumsParam::TupleTable g_mul_spectrums_table_cl
{
    // MatSize
    {
        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480,  2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},
    },
    // conj_src1
    {
        DT_TRUE,
        DT_FALSE
    },
    // impl
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(matrix, MulSpectrums, opencl)
{
    MatrixMulSpectrumsTest matrix_mul_spectrums_test(UnitTest::GetInstance()->GetContext(), g_mul_spectrums_table_cl);
    matrix_mul_spectrums_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}