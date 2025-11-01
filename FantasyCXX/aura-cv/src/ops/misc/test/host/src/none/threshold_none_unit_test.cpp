#include "host/include/threshold_unit_test.hpp"

static MiscThresholdParam::TupleTable g_threshold_table_none
{
    // datatype
    {
        aura::ElemType::U8,
        aura::ElemType::U16, aura::ElemType::S16,
        aura::ElemType::F32, aura::ElemType::F16
    },

    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},

        {aura::Sizes3(1024, 2048, 2), aura::Sizes()},
        {aura::Sizes3(479,  639,  2), aura::Sizes(480,  2560 * 2)},
        {aura::Sizes3(239,  319,  2), aura::Sizes()},

        {aura::Sizes3(1024, 2048, 3), aura::Sizes()},
        {aura::Sizes3(479,  639,  3), aura::Sizes(480,  2560 * 3)},
        {aura::Sizes3(239,  319,  3), aura::Sizes()},
    },

    //param
    {
        {15.1f, 2300.0f, AURA_THRESH_BINARY},
        {28.5f, 255.0f, AURA_THRESH_BINARY_INV},
        {127.6f, 35.0f, AURA_THRESH_TRUNC},
        {-8.2f, 127.2f, AURA_THRESH_TOZERO},
        {300.1f, 100.0f,AURA_THRESH_TOZERO_INV},
    },

    // target
    {
        aura::OpTarget::None()
    },
};

static MiscThresholdParam::TupleTable g_combine_threshold_table_none
{
    // datatype
    {
        aura::ElemType::U8
    },

    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},
    },

    // param
    {
        {0.9f, 122.0f, AURA_THRESH_OTSU | AURA_THRESH_BINARY},
        {10.9f, 255.0f, AURA_THRESH_OTSU | AURA_THRESH_BINARY_INV},
        {20.9f, 255.0f, AURA_THRESH_OTSU | AURA_THRESH_TRUNC},
        {30.9f, 122.0f, AURA_THRESH_OTSU | AURA_THRESH_TOZERO},
        {40.9f, 255.0f, AURA_THRESH_OTSU | AURA_THRESH_TOZERO_INV},
        {50.9f, 122.0f, AURA_THRESH_TRIANGLE | AURA_THRESH_BINARY},
        {60.9f, 255.0f, AURA_THRESH_TRIANGLE | AURA_THRESH_BINARY_INV},
        {70.9f, 122.0f, AURA_THRESH_TRIANGLE | AURA_THRESH_TRUNC},
        {80.9f, 255.0f, AURA_THRESH_TRIANGLE | AURA_THRESH_TOZERO},
        {90.9f, 122.0f, AURA_THRESH_TRIANGLE | AURA_THRESH_TOZERO_INV}
    },

    // target
    {
        aura::OpTarget::None()
    },
};

NEW_TESTCASE(misc, Threshold, none)
{
    ThresholdTest test_threshold(aura::UnitTest::GetInstance()->GetContext(), g_threshold_table_none);
    test_threshold.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    ThresholdTest test_combine_threshold(aura::UnitTest::GetInstance()->GetContext(), g_combine_threshold_table_none);
    test_combine_threshold.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}