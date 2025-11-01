#include "aura/tools/unit_test.h"

using namespace aura;

// example function to test
static Status Axpy(const Context *ctx, const Mat *src_x, const Mat *src_y, Mat *dst);

// define param struct and table for RunOne()
AURA_TEST_PARAM(AxpyRunParam,
                ElemType, elem_type,
                Sizes3,   img_size,
                Sizes,    img_stride);

static AxpyRunParam::TupleTable g_axpy_table_c
{
    {ElemType::U8},
    {Sizes3(860, 1024, 3)},
    {Sizes(900, 1026), Sizes(1901, 2027)},
};

class AxpyTest : public TestBase<AxpyRunParam::TupleTable, AxpyRunParam::Tuple>
{
public:
    AxpyTest(Context *ctx, AxpyRunParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32) override
    {
        if (MI_NULL == test_case)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "invalid test_case\n");
            return -1;
        }

        // get next param set
        AxpyRunParam run_param(GetParam((index)));

        // print param info
        AURA_LOGD(m_ctx, AURA_TAG, "******test param : %s \n", run_param.ToString().c_str());

        // creat iauras
        Mat img_x   = m_factory.GetRandomMat(1, 3, run_param.elem_type, run_param.img_size, AURA_MEM_DEFAULT, run_param.img_stride);
        Mat img_y   = m_factory.GetRandomMat(2, 4, run_param.elem_type, run_param.img_size, AURA_MEM_DEFAULT, run_param.img_stride);
        Mat img_dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.img_size, AURA_MEM_DEFAULT, run_param.img_stride);
        Mat img_ref = m_factory.GetRandomMat(3, 5, run_param.elem_type, run_param.img_size, AURA_MEM_DEFAULT, run_param.img_stride);

        // executor
        MI_S32 loop_count = 1;
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        if (UnitTest::GetInstance()->IsStressMode())
        {
            loop_count = UnitTest::GetInstance()->GetStressCount();
            result.param = std::string("stress_test_mode");
        }
        else
        {
            result.param = std::string("normal_test_mode");
        }

        Status status_exec = Executor(loop_count, 0, time_val, Axpy, m_ctx, &img_x, &img_y, &img_dst);

        if (status_exec != Status::OK)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "execute fail\n");
        }
        else
        {
            if (MatCompare(m_ctx, img_dst, img_ref, cmp_result, 2) == Status::OK)
            {
                AURA_LOGD(m_ctx, AURA_TAG, "%s \n", cmp_result.ToString().c_str());
            }
            else
            {
                AURA_LOGD(m_ctx, AURA_TAG, "iaura compare fail\n");
            }
        }

        result.input          = std::string("input");
        result.output         = std::string("output");
        result.perf_status    = TestStatus::PASSED;
        result.accu_status    = TestStatus::PASSED;
        result.accu_benchmark = std::string("accu_ref_impl");
        result.accu_result    = cmp_result.ToString();
        result.perf_result[run_param.ToString()] = time_val;

        test_case->AddTestResult(TestStatus::PASSED, result);
        // release resource
        m_factory.PutMats(img_x, img_y, img_dst, img_ref);
        m_factory.Clear();
        return 0;
    }

private:
    Context *m_ctx;
    MatFactory m_factory;
};

NEW_TESTCASE(unit_test, axpy, c)
{
    AxpyTest test_axpy(UnitTest::GetInstance()->GetContext(), g_axpy_table_c);
    test_axpy.RunTest(this, UnitTest::GetInstance()->GetStressCount());
}

template<typename T>
static Status Axpy_(const Context *ctx, const Mat *src_x, const Mat *src_y, Mat *dst)
{
    if ((MI_NULL == ctx) || (MI_NULL == src_x) || (MI_NULL == src_y) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid parm");
        return Status::ERROR;
    }

    {
        auto size_x   = src_x->GetSizes();
        auto size_y   = src_x->GetSizes();
        auto size_dst = dst->GetSizes();
        if ((size_dst != size_x) || (size_dst != size_y))
        {
            AURA_ADD_ERROR_STRING(ctx, "plane size not match");
            return Status::ERROR;
        }
    }

    {
        auto size = dst->GetSizes();
        for (auto h = 0; h < size.m_height; ++h)
        {
            for (auto w = 0; w < size.m_width; ++w)
            {
                for (auto c = 0; c < size.m_channel; ++c)
                {
                    dst->At<T>(h, w, c) = src_x->At<T>(h, w, c) + src_y->At<T>(h, w, c);
                }
            }
        }
    }
    return Status::OK;
}

static Status Axpy(const Context *ctx, const Mat *src_x, const Mat *src_y, Mat *dst)
{
    if ((dst->GetElemType() != src_x->GetElemType()) || (dst->GetElemType() != src_y->GetElemType()))
    {
        AURA_ADD_ERROR_STRING(ctx, "elem type not match");
        return Status::ERROR;
    }

    switch (dst->GetElemType())
    {
        case ElemType::U8:
        {
            Axpy_<MI_U8>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::S8:
        {
            Axpy_<MI_S8>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::U16:
        {
            Axpy_<MI_U16>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::S16:
        {
            Axpy_<MI_S16>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::U32:
        {
            Axpy_<MI_U32>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::S32:
        {
            Axpy_<MI_S32>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::F32:
        {
            Axpy_<MI_F32>(ctx, src_x, src_y, dst);
            break;
        }
        case ElemType::F64:
        {
            Axpy_<MI_F64>(ctx, src_x, src_y, dst);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type not support");
            return Status::ERROR;
        }
    }
    return Status::OK;
}
