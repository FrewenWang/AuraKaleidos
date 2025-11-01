#include "host/include/adaptive_threshold_unit_test.hpp"

static MiscAdaptiveThresholdParam::TupleTable g_adaptive_threshold_table_neon
{
    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},
    },

    // param
    {
        {255.0f, aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C, AURA_THRESH_BINARY, 3, 5.0},
        {127.5f, aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C, AURA_THRESH_BINARY_INV, 3, 8.0},
        {255.0f, aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C, AURA_THRESH_BINARY, 3, 5.0},
        {127.5f, aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C, AURA_THRESH_BINARY_INV, 3, 8.0},
    },

    // target
    {
        aura::OpTarget::Neon()
    },
};

NEW_TESTCASE(misc, AdaptiveThreshold, neon)
{
    AdaptiveThresholdTest test_adaptive_threshold(aura::UnitTest::GetInstance()->GetContext(), g_adaptive_threshold_table_neon);
    test_adaptive_threshold.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}