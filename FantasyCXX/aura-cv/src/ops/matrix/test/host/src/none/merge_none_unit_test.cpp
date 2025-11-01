#include "host/include/merge_unit_test.hpp"

static MergeParam::TupleTable g_merge_table_none
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
            {Sizes3(1024, 2048, 1), Sizes()},
            {Sizes3(1024, 2048, 2), Sizes()},
        },
        {
            {Sizes3(479, 639, 2), Sizes(479, 2560 * 2)},
            {Sizes3(479, 639, 3), Sizes(479, 2560 * 3)},
            {Sizes3(479, 639, 4), Sizes(479, 2560 * 4)},
        },
    },
    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Merge, none)
{
    MergeMultiMatTest matrix_merge_test(UnitTest::GetInstance()->GetContext(), g_merge_table_none);
    matrix_merge_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

