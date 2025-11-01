#include "host/include/integral_unit_test.hpp"

static IntegralParam::TupleTable g_integral_table_none
{
    // element: src and dst, dst_sq, cv_ref, cv_ref_sq elem type
    {
        // sum only
        {ElemType::U8,  ElemType::U32, ElemType::INVALID, ElemType::U32, ElemType::INVALID},
        {ElemType::U8,  ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::U8,  ElemType::F64, ElemType::INVALID, ElemType::F64, ElemType::INVALID},
        {ElemType::S8,  ElemType::S32, ElemType::INVALID, ElemType::S32, ElemType::INVALID},
        {ElemType::S8,  ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::S8,  ElemType::F64, ElemType::INVALID, ElemType::F64, ElemType::INVALID},
        {ElemType::U16, ElemType::U32, ElemType::INVALID, ElemType::U32, ElemType::INVALID},
        {ElemType::U16, ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::U16, ElemType::F64, ElemType::INVALID, ElemType::F64, ElemType::INVALID},
        {ElemType::S16, ElemType::S32, ElemType::INVALID, ElemType::S32, ElemType::INVALID},
        {ElemType::S16, ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::S16, ElemType::F64, ElemType::INVALID, ElemType::F64, ElemType::INVALID},
        {ElemType::F32, ElemType::F32, ElemType::INVALID, ElemType::F32, ElemType::INVALID},
        {ElemType::F32, ElemType::F64, ElemType::INVALID, ElemType::F64, ElemType::INVALID},

        // sqsum only
        {ElemType::U8,  ElemType::INVALID, ElemType::U32, ElemType::INVALID, ElemType::F64},
        {ElemType::U8,  ElemType::INVALID, ElemType::F64, ElemType::INVALID, ElemType::F64},
        {ElemType::S8,  ElemType::INVALID, ElemType::U32, ElemType::INVALID, ElemType::F64},
        {ElemType::S8,  ElemType::INVALID, ElemType::F64, ElemType::INVALID, ElemType::F64},
        {ElemType::U16, ElemType::INVALID, ElemType::F64, ElemType::INVALID, ElemType::F64},
        {ElemType::S16, ElemType::INVALID, ElemType::F64, ElemType::INVALID, ElemType::F64},
        {ElemType::F32, ElemType::INVALID, ElemType::F64, ElemType::INVALID, ElemType::F64},

        // sum and sqsum
        {ElemType::U8,  ElemType::U32, ElemType::F64, ElemType::U32, ElemType::F64},
        {ElemType::S8,  ElemType::S32, ElemType::F64, ElemType::S32, ElemType::F64},
        {ElemType::U16, ElemType::F32, ElemType::F64, ElemType::F32, ElemType::F64},
        {ElemType::S16, ElemType::F32, ElemType::F64, ElemType::F32, ElemType::F64},
        {ElemType::F32, ElemType::F64, ElemType::F64, ElemType::F64, ElemType::F64},
    },

    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480, 2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480, 2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // optarget
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Integral, none)
{
    MatrixIntegralTest matrix_integral_test(UnitTest::GetInstance()->GetContext(), g_integral_table_none);
    matrix_integral_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}