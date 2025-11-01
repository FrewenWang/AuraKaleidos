#include "host/include/creat_clahe_unit_test.hpp"

static CreatCLAHEParam::TupleTable g_clahe_table_none
{
    // elem_type
    {
        ElemType::U8,
        ElemType::U16
    },

    // MatSize
    {
        {Sizes3(1024, 2048, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // Param
    {
        {40.0, {8, 10}},
        {67.0, {11, 15}}
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(hist, CreatCLAHE, none)
{
    CreatCLAHETest test(UnitTest::GetInstance()->GetContext(), g_clahe_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}