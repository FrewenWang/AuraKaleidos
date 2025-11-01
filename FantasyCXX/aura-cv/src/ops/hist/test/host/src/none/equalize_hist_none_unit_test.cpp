#include "host/include/equalize_hist_unit_test.hpp"

static EqualizeHistParam::TupleTable g_hist_table_none
{
    // elem_type
    {
        ElemType::U8,
    },

    // MatSize
    {
        {Sizes3(1024, 2048, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(hist, Equalizehist, none)
{
    EqualizeHistTest test(UnitTest::GetInstance()->GetContext(), g_hist_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}