#include "host/include/houghlines_unit_test.hpp"

static HoughLinesParam::TupleTable g_houghlines_table_none
{
    // elem_type
    {
        aura::ElemType::U8
    },

    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},
    },

    {
        {aura::LinesType::VEC2F, 1, AURA_PI/180, 100,  0, 0, 0, AURA_PI},
        {aura::LinesType::VEC2F, 1, AURA_PI/180, 240,  0, 0, 0, AURA_PI},
        {aura::LinesType::VEC3F, 1, AURA_PI/180, 100,  0, 0, 0, AURA_PI},
        {aura::LinesType::VEC3F, 1, AURA_PI/180, 240,  0, 0, 0, AURA_PI},
        {aura::LinesType::VEC2F, 1, AURA_PI/180, 100,  2, 30, 0, AURA_PI},
        {aura::LinesType::VEC2F, 1, AURA_PI/180, 200,  2, 30, 0, AURA_PI},
        {aura::LinesType::VEC3F, 1, AURA_PI/180, 100,  2, 30, 0, AURA_PI},
        {aura::LinesType::VEC3F, 1, AURA_PI/180, 200,  2, 30, 0, AURA_PI},
    },

    // target
    {
        aura::OpTarget::None()
    },
};

NEW_TESTCASE(misc, HoughLines, none)
{
    HoughLinesTest houghlines_test(aura::UnitTest::GetInstance()->GetContext(), g_houghlines_table_none);
    houghlines_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

static HoughLinesPParam::TupleTable g_houghlinesp_table_none
{
    // elem_type
    {
        aura::ElemType::U8
    },

    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},
    },

    {
        {1, AURA_PI/180, 100, 100, 10},
        {1, AURA_PI/180, 240, 240, 10},
        {1, AURA_PI/180, 100, 100, 20},
        {1, AURA_PI/180, 240, 240, 20},
    },

    // target
    {
        aura::OpTarget::None()
    },
};

NEW_TESTCASE(misc, HoughLinesP, none)
{
    HoughLinesPTest houghlinesp_test(aura::UnitTest::GetInstance()->GetContext(), g_houghlinesp_table_none);
    houghlinesp_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}