#include "host/include/arithmetic_unit_test.hpp"
#include "aura/runtime/hexagon.h"

static ArithmeticParam::TupleTable g_arithmetic_table
{
    // element type src0
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
    },

    // element type dst
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
    },

    // MatSize
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480,  2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480,  2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // operator type
    {
        ArithmOpType::ADD,
        ArithmOpType::SUB,
        ArithmOpType::MUL,
    },

    // target
    {
        OpTarget::Hvx()
    },
};

NEW_TESTCASE(matrix, Arithmetic, hvx)
{
    HexagonEngine *engine = UnitTest::GetInstance()->GetContext()->GetHexagonEngine();
    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    ArithmeticTest test(UnitTest::GetInstance()->GetContext(), g_arithmetic_table);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
}