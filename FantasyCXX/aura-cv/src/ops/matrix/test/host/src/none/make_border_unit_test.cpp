#include "host/include/make_border_unit_test.hpp"

static MakeBorderRunParam::TupleTable g_make_border_table_none
{
    // element type
    {
        ElemType::U8,  ElemType::S8,  ElemType::U16, ElemType::S16,
        ElemType::U32, ElemType::S32, ElemType::F16, ElemType::F32,
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

    // border typer
    {BorderType::CONSTANT, BorderType::REPLICATE, BorderType::REFLECT_101},

    // border size
    {BorderSize(5, 5, 5, 5), BorderSize(2, 2, 0, 0), BorderSize(0, 0, 1, 1)},

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, MakeBorder, none)
{
    MakeBorderTest make_border_test(UnitTest::GetInstance()->GetContext(), g_make_border_table_none);
    make_border_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}