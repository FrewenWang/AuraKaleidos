#include "host/include/sobel_unit_test.hpp"

static SobelParam::TupleTable g_sobel_table_cl {
    // element type
    {
        {ElemType::U8,  ElemType::S16},
        {ElemType::U8,  ElemType::F32},
        {ElemType::U16, ElemType::U16},
        {ElemType::S16, ElemType::S16},
        {ElemType::F16, ElemType::F16},
        {ElemType::F32, ElemType::F32}
    },

    // mat size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480, 2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480, 2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},

        {Sizes3(1024, 2048, 3), Sizes()},
        {Sizes3(479,  639,  3), Sizes(480, 2560 * 3)},
        {Sizes3(239,  319,  3), Sizes()},
    },

    // sobel test param: {dx, dy, ksize, scale}
    {
        {1, 0, -1, 1.f},
        {1, 0, -1, 2.f},
        {0, 1, -1, 1.f},
        {0, 1, -1, 2.f},
        {1, 0,  0, 1.f},
        {1, 0,  0, 2.f},
        {0, 1,  0, 1.f},
        {0, 1,  0, 2.f},
        {1, 0,  1, 1.f},
        {1, 0,  1, 2.f},
        {0, 1,  1, 1.f},
        {0, 1,  1, 2.f},
        {2, 2,  1, 1.f},
        {2, 2,  1, 2.f},
        {1, 2,  3, 1.f},
        {2, 1,  3, 2.f},
        {1, 2,  5, 1.f},
        {2, 1,  5, 2.f}
    },

    // border type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101
    },

    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(filter, Sobel, opencl)
{
    SobelTest test(UnitTest::GetInstance()->GetContext(), g_sobel_table_cl);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}
