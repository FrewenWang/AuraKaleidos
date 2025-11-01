#include "host/include/connect_component_label_unit_test.hpp"

static MiscConnectComponentLabelParam::TupleTable g_connect_component_table_opencl
{
    // Mat Size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
    },

    // label mat datatype
    {
        aura::ElemType::S32,
        aura::ElemType::U32,
    },

    // mask density
    {
        0.0f,
        0.1f,
        0.3f,
        0.5f,
        0.7f,
        0.9f,
        1.0f,
    },

    // CCL algo type
    {
        {aura::CCLAlgo::HA_GPU, aura::ConnectivityType::CROSS,  aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::HA_GPU, aura::ConnectivityType::SQUARE, aura::EquivalenceSolver::UNION_FIND,},
    },

    // taret
    {
        aura::OpTarget::Opencl(),
    },
};

NEW_TESTCASE(misc, ConnectComponentLabel, opencl)
{
    ConnectComponentLabelTest connectcomponentlabel_test(aura::UnitTest::GetInstance()->GetContext(), g_connect_component_table_opencl);
    connectcomponentlabel_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}