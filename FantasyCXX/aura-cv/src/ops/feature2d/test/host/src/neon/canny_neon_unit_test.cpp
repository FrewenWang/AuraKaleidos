#include "host/include/canny_unit_test.hpp"

static CannyPixelParam::TupleTable g_canny_pixel_table_neon
{
    // elem_type
    {
        ElemType::U8
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480,  2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // low_thresh
    {
        80, 60,
    },

    // high_thresh
    {
        100, 150,
    },

    // aperture_size
    {
        3, 5, //7,
    },

    // l2_gradient
    {
        MI_FALSE, MI_TRUE,
    },

    // target
    {
        OpTarget::Neon()
    },
};

static CannyGradientParam::TupleTable g_canny_gradient_table_neon
{
    // imat_elem_type
    {
        ElemType::S16
    },

    // omat_elem_type
    {
        ElemType::U8
    },

    // ImatSize
    {
        {Sizes3(2048, 4096, 1), Sizes(2048, 4096 * 1)},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},

        {Sizes3(2048, 4096, 3), Sizes(2048, 4096 * 3)},
        {Sizes3(479,  639,  3), Sizes(480,  2560 * 3)},
    },

    // low_thresh
    {
        80, 60,
    },

    // high_thresh
    {
        100, 150,
    },

    // l2_gradient
    {
        MI_FALSE, MI_TRUE,
    },

    // target
    {
        OpTarget::Neon()
    },
};

NEW_TESTCASE(feature2d, Canny, neon)
{
    CannyPixelTest pixel_test(UnitTest::GetInstance()->GetContext(), g_canny_pixel_table_neon);
    pixel_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    CannyGradientTest Gradient_test(UnitTest::GetInstance()->GetContext(), g_canny_gradient_table_neon);
    Gradient_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}