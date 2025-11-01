#include "host/include/find_homography_unit_test.hpp"

static FindHomographyParam::TupleTable g_find_homography_table_none
{
    // iaura size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(1024, 2048, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes()},
        {Sizes3(239,  319,  1), Sizes()}
    },

    // point number
    {
        4,
        50,
        303
    }
};

NEW_TESTCASE(warp, FindHomography, none)
{
    FindHomographyTest test(UnitTest::GetInstance()->GetContext(), g_find_homography_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}