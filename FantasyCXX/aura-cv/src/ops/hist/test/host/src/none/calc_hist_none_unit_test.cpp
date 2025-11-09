#include "host/include/calc_hist_unit_test.hpp"

static CalcHistParam::TupleTable g_hist_table_none
{
    // MatSize
    {
        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480, 2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // param : elem type | channel | hist size | range | accumulate | use mask
    {
        {ElemType::U8 , 0, 256, {0, 256}, DT_FALSE, DT_FALSE},
        {ElemType::U8 , 1, 242, {2, 244}, DT_FALSE, DT_TRUE},
        {ElemType::U16, 0, 65536, {0, 65536}, DT_FALSE, DT_FALSE},
        {ElemType::U16, 2, 65520, {10, 65530}, DT_FALSE, DT_TRUE}
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(hist, calchist, none)
{
    CalcHistTest test(UnitTest::GetInstance()->GetContext(), g_hist_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}