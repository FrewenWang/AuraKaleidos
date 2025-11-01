#include "host/include/mipi_unit_test.hpp"

static MiscMipiParam::TupleTable g_mipi_unpack_table_neon
{
    // MatSize
    {
        {
            {aura::Sizes3(2048, 4096 * 5 / 4, 1)}, {aura::Sizes3(2048, 4096 * 5 / 2, 1)}
        },
    },

    // datatype
    {
        aura::ElemType::U8, aura::ElemType::U16
    },

    // target
    {
        aura::OpTarget::Neon()
    },
};

NEW_TESTCASE(misc, MipiUnpack, neon)
{
    MipiUnpackTest test(aura::UnitTest::GetInstance()->GetContext(), g_mipi_unpack_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

static MiscMipiParam::TupleTable g_mipi_pack_table_neon
{
    // MatSize
    {
        {
            {aura::Sizes3(2048, 4096, 1)}, {aura::Sizes3(2048, 4096 * 2, 1)}
        },
    },

    // datatype
    {
        aura::ElemType::U8
    },

    // target
    {
        aura::OpTarget::Neon()
    },
};

NEW_TESTCASE(misc, MipiPack, neon)
{
    MipiPackTest test(aura::UnitTest::GetInstance()->GetContext(), g_mipi_pack_table_neon);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}