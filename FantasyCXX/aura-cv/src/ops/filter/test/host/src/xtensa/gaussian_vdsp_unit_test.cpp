#include "host/include/gaussian_unit_test.hpp"
#include "aura/runtime/xtensa.h"

static GaussianParam::TupleTable g_gaussian_table_vdsp{
    // elem_type
    {
        ElemType::U8,
    },

    // MatSize
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    {
        {3, 0.0f},
        {3, 1.3f},
    },
    //ElemType
    {
        BorderType::REPLICATE,
        BorderType::CONSTANT,
        BorderType::REFLECT_101,
    },
    //OpTarget
    {
        OpTarget::Vdsp()
    },
};

NEW_TESTCASE(filter, Gaussian, vdsp)
{
    XtensaEngine *engine = UnitTest::GetInstance()->GetContext()->GetXtensaEngine();
    engine->SetPower(aura::XtensaPowerLevel::TURBO);

    GaussianTest test(UnitTest::GetInstance()->GetContext(), g_gaussian_table_vdsp);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());

    engine->SetPower(aura::XtensaPowerLevel::STANDBY);
}
