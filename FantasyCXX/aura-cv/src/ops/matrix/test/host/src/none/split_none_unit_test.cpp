#include "host/include/split_unit_test.hpp"

static SplitParam::TupleTable g_split_table_none
{
    // elem_type
    {
        ElemType::U8,  ElemType::S8,
        ElemType::U16, ElemType::S16,
        ElemType::U32, ElemType::S32,
        ElemType::F32,
    },
    // MatSize
    {
        {
            {Sizes3(2048, 4096, 1), Sizes()},
            {Sizes3(2048, 4096, 1), Sizes()},
        },
        {
            {Sizes3(479, 639, 1), Sizes(480, 2560 * 1)},
            {Sizes3(479, 639, 1), Sizes(480, 2560 * 1)},
            {Sizes3(479, 639, 1), Sizes(480, 2560 * 1)},
        },
    },
    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Split, none)
{
    SplitMultiMatTest split_multi_mat_test(UnitTest::GetInstance()->GetContext(), g_split_table_none);
    split_multi_mat_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
