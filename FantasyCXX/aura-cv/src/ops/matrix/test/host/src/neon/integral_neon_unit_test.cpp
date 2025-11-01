#include "host/include/integral_unit_test.hpp"

static IntegralParam::TupleTable g_integral_table_neon
{
    // element pair: src and dst elem type
    {
        // sum only
        {ElemType::U8, ElemType::U32, ElemType::INVALID, ElemType::U32, ElemType::INVALID},
        {ElemType::U8, ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::S8, ElemType::S32, ElemType::INVALID, ElemType::S32, ElemType::INVALID},
        {ElemType::S8, ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},

        // sq sum only
        {ElemType::U8, ElemType::INVALID, ElemType::U32, ElemType::INVALID, ElemType::U32},
        {ElemType::S8, ElemType::INVALID, ElemType::U32, ElemType::INVALID, ElemType::U32},

        // sum and sqsum
        {ElemType::U8, ElemType::U32, ElemType::U32, ElemType::U32, ElemType::U32},
        {ElemType::S8, ElemType::S32, ElemType::U32, ElemType::U32, ElemType::U32},
        {ElemType::U8, ElemType::F32, ElemType::U32, ElemType::F32, ElemType::U32},
        {ElemType::S8, ElemType::F32, ElemType::U32, ElemType::F32, ElemType::U32},
    },

    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560)},
        {Sizes3(239,  319,  1), Sizes()},
    },
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(matrix, Integral, neon)
{
    MatrixIntegralTest matrix_integral_test(UnitTest::GetInstance()->GetContext(), g_integral_table_neon);
    matrix_integral_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}