#include "host/include/mipi_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static MiscMipiParam::TupleTable g_mipi_unpack_table_hvx
{
    // MatSize
    {
        {
            {aura::Sizes3(2048, 4096 * 5 / 4, 1)}, {aura::Sizes3(2048, 4096 * 5 / 2, 1)}
        },
    },

    // datatype
    {
        aura::ElemType::U8,
        aura::ElemType::U16,
    },

    // target
    {
        aura::OpTarget::Hvx()
    },
};

NEW_TESTCASE(misc, MipiUnpack, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    MipiUnpackTest test(aura::UnitTest::GetInstance()->GetContext(), g_mipi_unpack_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}

static MiscMipiParam::TupleTable g_mipi_pack_table_hvx
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
        aura::OpTarget::Hvx()
    },
};

NEW_TESTCASE(misc, MipiPack, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    MipiPackTest test(aura::UnitTest::GetInstance()->GetContext(), g_mipi_pack_table_hvx);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}