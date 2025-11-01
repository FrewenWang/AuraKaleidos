#include "host/include/houghcircles_unit_test.hpp"

static HoughCirclesParam::TupleTable g_hough_circles_table_neon
{
    // datatype
    {
        aura::ElemType::U8,
    },

    // MatSize
    {
        {aura::Sizes3(2048, 4096, 1), aura::Sizes()},
        {aura::Sizes3(479,  639,  1), aura::Sizes(480,  2560 * 1)},
        {aura::Sizes3(239,  319,  1), aura::Sizes()},
    },

    // Parameters
    {
        {aura::HoughCirclesMethod::HOUGH_GRADIENT, 1, 115, 100, 30, 1, 130},
        {aura::HoughCirclesMethod::HOUGH_GRADIENT, 2, 115, 100, 30, 1, 130},
        {aura::HoughCirclesMethod::HOUGH_GRADIENT, 1, 115, 100, 30, 1, -1},
        {aura::HoughCirclesMethod::HOUGH_GRADIENT, 2, 115, 100, 30, 1, -1},
    },

    // target
    {
        aura::OpTarget::Neon()
    },
};

NEW_TESTCASE(misc, HoughCircles, neon)
{
    HoughCirclesTest test(aura::UnitTest::GetInstance()->GetContext(), g_hough_circles_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}