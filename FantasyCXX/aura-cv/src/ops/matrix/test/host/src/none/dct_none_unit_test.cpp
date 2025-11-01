#include "host/include/dct_idct_unit_test.hpp"

static DctParam::TupleTable g_dct_table_none
{
    // ElemType
    {
        aura::ElemType::U8,
        aura::ElemType::S8,
        aura::ElemType::U16,
        aura::ElemType::S16,
#if defined(AURA_BUILD_HOST)
        aura::ElemType::F16,
#endif // AURA_BUILD_HOST
        aura::ElemType::F32,
    },
    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(1024, 2048, 1), aura::Sizes(1025, 2050 * 2)},
        {aura::Sizes3(512,  1024, 1), aura::Sizes()},
        {aura::Sizes3(64,   512,  1), aura::Sizes()},
        {aura::Sizes3(2,    64,   1), aura::Sizes()},
        {aura::Sizes3(64,   2,    1), aura::Sizes()},
        {aura::Sizes3(1,    64,   1), aura::Sizes()},
        {aura::Sizes3(64,   1,    1), aura::Sizes()},
    },
    // impl
    {
        aura::OpTarget::None()
    },
};


static IDctParam::TupleTable g_idct_table_none
{
    // ElemType
    {
        aura::ElemType::U8,
        aura::ElemType::S8,
        aura::ElemType::U16,
        aura::ElemType::S16,
#if defined(AURA_BUILD_HOST)
        aura::ElemType::F16,
#endif // AURA_BUILD_HOST
        aura::ElemType::F32,
    },
    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(1024, 2048, 1), aura::Sizes(1025, 2050 * 2)},
        {aura::Sizes3(512,  1024, 1), aura::Sizes()},
        {aura::Sizes3(64,   512,  1), aura::Sizes()},
        {aura::Sizes3(2,    64,   1), aura::Sizes()},
        {aura::Sizes3(64,   2,    1), aura::Sizes()},
        {aura::Sizes3(1,    64,   1), aura::Sizes()},
        {aura::Sizes3(64,   1,    1), aura::Sizes()},
    },
    // impl
    {
        aura::OpTarget::None()
    },
};

NEW_TESTCASE(matrix, Dct, none)
{
    MatrixDctTest matrix_dct_test(UnitTest::GetInstance()->GetContext(), g_dct_table_none);
    matrix_dct_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

NEW_TESTCASE(matrix, IDct, none)
{
    MatrixIDctTest matrix_idct_test(UnitTest::GetInstance()->GetContext(), g_idct_table_none);
    matrix_idct_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}