#include "host/include/harris_unit_test.hpp"

static HarrisParam::TupleTable g_harris_table_none
{
    // elem_type
    {
        ElemType::U8, ElemType::F32,
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // param
    {
        {3, 3, 0.04},
        {5, 3, 0.04},
        {5, 3, 0.0},
    },

    // border_type
    {
        BorderType::CONSTANT, 
        BorderType::REPLICATE,
        BorderType::REFLECT_101, 
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(feature2d, CornerHarris, none)
{
    HarrisTest test(UnitTest::GetInstance()->GetContext(), g_harris_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}