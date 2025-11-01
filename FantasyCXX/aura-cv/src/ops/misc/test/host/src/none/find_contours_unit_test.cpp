#include "host/include/find_contours_unit_test.hpp"

static FindContoursParam::TupleTable g_find_contours_table_none {
    // element type, only support U8 type
    {
        ElemType::U8
    },

    // matrix size, only support single channel
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // find contours mode
    {
        ContoursMode::RETR_EXTERNAL,
        ContoursMode::RETR_LIST
    },

    // find contours method
    {
        ContoursMethod::CHAIN_APPROX_NONE,
        ContoursMethod::CHAIN_APPROX_SIMPLE
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(misc, FindContours, none)
{
    FindContoursTest test(UnitTest::GetInstance()->GetContext(), g_find_contours_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}