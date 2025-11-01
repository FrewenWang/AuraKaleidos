#include "host/include/cvtcolor_unit_test.hpp"
#include "aura/runtime/xtensa.h"

static CvtColorParam::TupleTable g_cvtcolor_bgr2gray_table_vdsp
{
    // elem_type
    {
        ElemType::U8,
    },

    // mat size pair
    {
        {
            {
                {Sizes3(2048, 4096, 3), Sizes(2048, 4096 * 3)}
            },
            {
                {Sizes3(2048, 4096, 1), Sizes(2048, 4096)}
            }
        },
        {
            {
                {Sizes3(479, 639, 3), Sizes(479, 639 * 3)}
            },
            {
                {Sizes3(479, 639, 1), Sizes(479, 639)}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::BGR2GRAY,
        CvtColorType::RGB2GRAY,
    },

    // target
    {
        OpTarget::Vdsp()
    },
};

NEW_TESTCASE(cvtcolor, cvtcolor, vdsp)
{
    XtensaEngine *engine = UnitTest::GetInstance()->GetContext()->GetXtensaEngine();
    engine->SetPower(aura::XtensaPowerLevel::TURBO);

    {
        CvtColorTest test_bgr2gray(UnitTest::GetInstance()->GetContext(), g_cvtcolor_bgr2gray_table_vdsp);
        test_bgr2gray.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    engine->SetPower(aura::XtensaPowerLevel::STANDBY);
}