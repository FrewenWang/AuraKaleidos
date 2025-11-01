#include "host/include/laplacian_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static LaplacianParam::TupleTable g_laplacian_table_hvx
{
    // element type
    {
        {ElemType::U8,  ElemType::S16},
        {ElemType::U16, ElemType::U16},
        {ElemType::S16, ElemType::S16},
    },

    // matrix size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480, 2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480, 2560 * 3)}
    },

    // kernel size
    {
        1,
        3,
        5,
        7
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    // target
    {
        OpTarget::Hvx()
    }
};

NEW_TESTCASE(filter, Laplacian, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    LaplacianTest test(UnitTest::GetInstance()->GetContext(), g_laplacian_table_hvx);
    test.RunTest(this);

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}
