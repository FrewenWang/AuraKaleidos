#include "host/include/integral_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static IntegralParam::TupleTable g_integral_table_hvx
{
    // element pair: src and dst elem type
    {
        {ElemType::U8, ElemType::U32, ElemType::INVALID, ElemType::U32, ElemType::INVALID},
        {ElemType::S8, ElemType::S32, ElemType::INVALID, ElemType::S32, ElemType::INVALID},
    },
    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480, 2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},
    },
    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(matrix, Integral, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, DT_FALSE);

    MatrixIntegralTest matrix_integral_test(UnitTest::GetInstance()->GetContext(), g_integral_table_hvx);
    matrix_integral_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, DT_FALSE);
}