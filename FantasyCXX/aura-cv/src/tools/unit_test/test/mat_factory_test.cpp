#include "aura/tools/unit_test.h"


using namespace aura;

AURA_TEST_PARAM(FactoryParam,
                ElemType, elem_type,
                MI_S32,   channel);

static FactoryParam::TupleTable g_factory_table_none
{
    {
        ElemType::U8,  ElemType::S8,  ElemType::U16, ElemType::S16,
        ElemType::U32, ElemType::S32, ElemType::F32, ElemType::F64
    },
    {
        1, 3
    },
};

class MatFactoryTest : public TestBase<FactoryParam::TupleTable, FactoryParam::Tuple>
{
public:
    MatFactoryTest(Context *ctx, FactoryParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(m_ctx)
    {
        Status status = Status::OK;
        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGE(this->m_ctx, AURA_TAG, "LoadBaseMat failed");
        }

        m_factory.PrintInfo();
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32) override
    {
        FactoryParam param(GetParam((index)));

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        std::string param_str = "param : " + param.ToString();
        AURA_LOGD(m_ctx, AURA_TAG, param_str.c_str());

        ElemType elem_type = param.elem_type;
        MI_S32 nchannel     = param.channel;

        Sizes3 size0{64, 64, nchannel};

        Mat mat0 = m_factory.GetFileMat(data_file + "baboon_512x512.rgb", elem_type, size0);
        AURA_CHECK_EQ(m_ctx, mat0.IsValid(), MI_TRUE, "mat is invalid");

        Sizes3 size1{512, 512, nchannel};
        Mat mat1 = m_factory.GetRandomMat(0, 1000, elem_type, size1);
        AURA_CHECK_EQ(m_ctx, mat1.IsValid(), MI_TRUE, "mat is invalid");

        Sizes3 size2{512, 512, nchannel};
        Mat mat2 = m_factory.GetDerivedMat(2, 1, elem_type, size2);
        AURA_CHECK_EQ(m_ctx, mat2.IsValid(), MI_TRUE, "mat is invalid");

        Sizes3 size3{2048, 3060, nchannel};
        Mat mat3 = m_factory.GetEmptyMat(elem_type, size3);
        AURA_CHECK_EQ(m_ctx, mat3.IsValid(), MI_TRUE, "mat is invalid");

        m_factory.PutMats(mat0, mat1, mat2, mat3);
        m_factory.PrintInfo();

        test_case->AddTestResult(TestStatus::PASSED);

        return 0;
    }

private:
    Context *m_ctx;
    MatFactory m_factory;
};

NEW_TESTCASE(unit_test, mat_factory, none)
{
    MatFactoryTest test_generator(UnitTest::GetInstance()->GetContext(), g_factory_table_none);
    test_generator.RunTest(this);
}

// ============================
//   Test Overflow Situation
// ============================
NEW_TESTCASE(unit_test, mat_factory, overflow)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    MatFactory mf(ctx, 512); // set max memory as 312MB
    mf.PrintInfo();

    AURA_LOGD(ctx, AURA_TAG, "############################## ADD SOME FILE ##############################\n");

    Status status = Status::OK;
    std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";
    status |= mf.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
    status |= mf.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

    if (status != Status::OK)
    {
        AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed at line %d\n", __LINE__);
        return;
    }

    AURA_LOGD(ctx, AURA_TAG, "############################## ADD MAT ##############################\n");
    // Add mat, but not to exceed the memory limit
    for (MI_S32 i = 0; i < 7; i++)
    {
        AURA_LOGD(ctx, AURA_TAG, "====================> %d\n", i);
        Mat mat0 = mf.GetRandomMat(i, 2 * i, ElemType::U8, Sizes(720, 1080));
        Mat mat1 = mf.GetEmptyMat(ElemType::U8, Sizes(720 + i, 1080 + i));
        Mat mat2 = mf.GetDerivedMat(1.0f, 1.0f, ElemType::S8, Sizes(720 + i, 1080 + i));
        Mat mat3 = mf.GetFileMat(data_file + "cameraman_487x487.gray", ElemType::U8, {480 + i, 480 + i});

        mf.PutMats(mat0, mat1, mat2, mat3);
        mf.PrintInfo();
    }

    AURA_LOGD(ctx, AURA_TAG, "############################## REUSE MAT ##############################\n");
    // test reusage
    for (MI_S32 i = 0; i < 7; i++)
    {
        AURA_LOGD(ctx, AURA_TAG, "====================> %d\n", i);
        Mat mat0 = mf.GetRandomMat(i, 2 * i, ElemType::U8, Sizes(720, 1080));
        Mat mat1 = mf.GetEmptyMat(ElemType::U8, Sizes(720 + i, 1080 + i));
        Mat mat2 = mf.GetDerivedMat(1.0f, 1.0f, ElemType::S8, Sizes(720 + i, 1080 + i));
        Mat mat3 = mf.GetFileMat(data_file + "cameraman_487x487.gray", ElemType::U8, {480 + i, 480 + i});
        mf.PrintInfo();

        Mat mat4 = mf.GetRandomMat(i, 2 * i, ElemType::U8, Sizes(720, 1080));
        Mat mat5 = mf.GetEmptyMat(ElemType::U8, Sizes(720 + i, 1080 + i));
        Mat mat6 = mf.GetDerivedMat(1.0f, 1.0f, ElemType::S8, Sizes(720 + i, 1080 + i));
        Mat mat7 = mf.GetFileMat(data_file + "cameraman_487x487.gray", ElemType::U8, {480 + i, 480 + i});

        mf.PutMats(mat0, mat1, mat2, mat3, mat4, mat5, mat6, mat7);
        mf.PrintInfo();
    }

    AURA_LOGD(ctx, AURA_TAG, "############################## OVERFLOW TEST ##############################\n");
    // test overlfow
    for (MI_S32 i = 7; i < 10; i++)
    {
        AURA_LOGD(ctx, AURA_TAG, "====================> %d\n", i);
        Mat mat0 = mf.GetRandomMat(i, 2 * i, ElemType::U8, Sizes(720, 1080));
        Mat mat1 = mf.GetEmptyMat(ElemType::U8, Sizes(720 + i, 1080 + i));
        Sizes3 sizes{720 + i, 1080 + i};
        Mat mat2 = mf.GetDerivedMat(1.0f, 1.0f, ElemType::U16, sizes);
        Mat mat3 = mf.GetFileMat(data_file + "cameraman_487x487.gray", ElemType::U8, {400 + i, 400 + i});

        mf.PutMats(mat0, mat1, mat2, mat3);
        mf.PrintInfo();
    }

    AURA_LOGD(ctx, AURA_TAG, "############################## PutAllMats TEST ##############################\n");
    for (MI_S32 i = 7; i < 10; i++)
    {
        AURA_LOGD(ctx, AURA_TAG, "====================> %d\n", i);
        Mat mat0 = mf.GetRandomMat(i, 2 * i, ElemType::U8, Sizes(720, 1080));
        Mat mat1 = mf.GetEmptyMat(ElemType::U8, Sizes(720 + i, 1080 + i));
        Sizes3 sizes{720 + i, 1080 + i};
        Mat mat2 = mf.GetDerivedMat(1.0f, 1.0f, ElemType::U16, sizes);
        Mat mat3 = mf.GetFileMat(data_file + "cameraman_487x487.gray", ElemType::U8, {400 + i, 400 + i});

        mf.PrintInfo();
    }

    mf.PutAllMats();
    mf.PrintInfo();

    AddTestResult(TestStatus::PASSED);
}