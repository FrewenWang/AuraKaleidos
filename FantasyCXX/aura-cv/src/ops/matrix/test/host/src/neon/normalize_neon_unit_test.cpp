#include "host/include/normalize_unit_test.hpp"

static NormalizeParam::TupleTable g_normalize_table_neon
{
    // element type
    {
        ElemType::U8,  ElemType::S8,  ElemType::U16,
        ElemType::S16, ElemType::F32,
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
    // Extra param
    {
        {0.0, 1.0, NormType::NORM_MINMAX},
        {3.4321, 0.0, NormType::NORM_L1},
        {12.322, 0.0, NormType::NORM_L2},
        {3.1415, 0.0, NormType::NORM_INF},
    },
    // Impl
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(matrix, Normalize, neon)
{
    MatrixNormalizeTest matrix_normalize_test(UnitTest::GetInstance()->GetContext(), g_normalize_table_neon);
    matrix_normalize_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
