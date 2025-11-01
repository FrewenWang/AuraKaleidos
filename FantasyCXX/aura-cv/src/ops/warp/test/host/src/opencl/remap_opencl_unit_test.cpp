#include "host/include/remap_unit_test.hpp"

static RemapParam::TupleTable g_remap_table_cl
{
    // elem_type
    {
        ElemType::U8,
        ElemType::S8,
        ElemType::U16,
        ElemType::S16,
        ElemType::U32,
        ElemType::S32,
        ElemType::F16,
        ElemType::F32,
    },
    // map_elem_type
    {
        ElemType::F32,
        ElemType::S16,
    },
    // mat_size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},

        {Sizes3(1024, 2048, 2), Sizes()},
        {Sizes3(479,  639,  2), Sizes(480,  2560 * 2)},
        {Sizes3(239,  319,  2), Sizes()},
    },
    // interp_type
    {
        InterpType::NEAREST,
        InterpType::LINEAR,
        InterpType::CUBIC,
    },
    // border_type
    {
        BorderType::CONSTANT,
        BorderType::REPLICATE,
        BorderType::REFLECT_101,
    },
    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(warp, Remap, opencl)
{
    RemapTest test(UnitTest::GetInstance()->GetContext(), g_remap_table_cl);
    test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}