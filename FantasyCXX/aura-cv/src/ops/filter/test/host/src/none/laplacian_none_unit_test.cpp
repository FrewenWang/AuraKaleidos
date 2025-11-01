#include "host/include/laplacian_unit_test.hpp"

static LaplacianParam::TupleTable g_laplacian_table_none {
    // element type
    {
        {ElemType::U8,  ElemType::S16},
        {ElemType::U16, ElemType::U16},
        {ElemType::S16, ElemType::S16},
        {ElemType::F16, ElemType::F16},
        {ElemType::F32, ElemType::F32}
    },

    // matrix size
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

    // kernel size
    {
        1,
        3,
        5,
        7
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    // target
    {
        OpTarget::None()
    },
};

NEW_TESTCASE(filter, Laplacian, none)
{
    LaplacianTest test(UnitTest::GetInstance()->GetContext(), g_laplacian_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
