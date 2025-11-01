#include "host/include/pyrdown_unit_test.hpp"

static PyrDownParam::TupleTable g_pyrdown_table_none
{
    //elem type
    {
        ElemType::U8,
        ElemType::S16,
        ElemType::U16,
    },

    // input matrix size
    {
        {
            {Sizes3(2048, 4096, 1), Sizes()},
            {Sizes3(1024, 2048, 1), Sizes()},
        },

        {
            {Sizes3(479, 639, 1), Sizes(480, 2560 * 1)},
            {Sizes3(240, 320, 1), Sizes(240, 1280 * 1)},
        },

        {
            {Sizes3(239, 319, 1), Sizes()},
            {Sizes3(120, 160, 1), Sizes()},
        },
    },

    {
        {5, 0.0f},
        {5, 3.3f},
    },

    {
        BorderType::REFLECT_101,
        BorderType::REPLICATE,
    },

    {
        OpTarget::None()
    },
};

NEW_TESTCASE(pyramid, PyrDown, none)
{
    PyrDownTest test(UnitTest::GetInstance()->GetContext(), g_pyrdown_table_none);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}