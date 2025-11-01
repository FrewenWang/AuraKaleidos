#include "host/include/connect_component_label_unit_test.hpp"

static MiscConnectComponentLabelParam::TupleTable g_connect_component_table_none_tta
{
    // Mat Size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
        {Sizes3(240,  319,  1), Sizes()},
        {Sizes3(239,  320,  1), Sizes()},
    },

    // label mat datatype
    {
        aura::ElemType::U32,
        aura::ElemType::S32,
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
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::THREE_TABLE_ARRAYS,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::THREE_TABLE_ARRAYS,},
        {aura::CCLAlgo::BBDT,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::THREE_TABLE_ARRAYS,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::THREE_TABLE_ARRAYS,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::THREE_TABLE_ARRAYS,},
    },

    // taret
    {
        aura::OpTarget::None(MI_TRUE),
        aura::OpTarget::None(MI_FALSE),
    },
};

static MiscConnectComponentLabelParam::TupleTable g_connect_component_table_none_others
{
    // Mat Size
    {
        {Sizes3(2048, 4096, 1), Sizes()},
        {Sizes3(479,  639,  1), Sizes(480,  2560 * 1)},
        {Sizes3(239,  319,  1), Sizes()},
        {Sizes3(240,  319,  1), Sizes()},
        {Sizes3(239,  320,  1), Sizes()},
    },

    // label mat datatype
    {
        aura::ElemType::U8,
        aura::ElemType::U16,
        aura::ElemType::U32,
        aura::ElemType::S32,
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
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::UNION_FIND_PATH_COMPRESS,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::REM_SPLICING,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND_PATH_COMPRESS,},
        {aura::CCLAlgo::SAUF,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::REM_SPLICING,},
        {aura::CCLAlgo::BBDT,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::BBDT,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND_PATH_COMPRESS,},
        {aura::CCLAlgo::BBDT,      aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::REM_SPLICING,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::UNION_FIND_PATH_COMPRESS,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::CROSS,   aura::EquivalenceSolver::REM_SPLICING,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::UNION_FIND_PATH_COMPRESS,},
        {aura::CCLAlgo::SPAGHETTI, aura::ConnectivityType::SQUARE,  aura::EquivalenceSolver::REM_SPLICING,},
    },

    // taret
    {
        aura::OpTarget::None(MI_TRUE),
        aura::OpTarget::None(MI_FALSE),
    },
};

NEW_TESTCASE(misc, ConnectComponentLabel, none)
{
    // TTASolver, only 32bit integer
    {
        ConnectComponentLabelTest connectcomponentlabel_test(aura::UnitTest::GetInstance()->GetContext(), g_connect_component_table_none_tta);
        connectcomponentlabel_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    // other solvers
    {
        ConnectComponentLabelTest connectcomponentlabel_test(aura::UnitTest::GetInstance()->GetContext(), g_connect_component_table_none_others);
        connectcomponentlabel_test.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }
}