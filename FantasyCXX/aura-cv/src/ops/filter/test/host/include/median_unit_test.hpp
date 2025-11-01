#ifndef AURA_OPS_MEDIAN_UINT_TEST_HPP__
#define AURA_OPS_MEDIAN_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MedianParam,
                ElemType, elem_type,
                MatSize,  mat_size,
                MI_S32,   ksize,
                OpTarget, target);

static Status CvMedian(Context *ctx, Mat &src, Mat &dst, MI_S32 ksize)
{
    if ((src.GetSizes().m_channel == 2) ||
        (src.GetElemType() != ElemType::U8 && ksize >= 7) ||
        (src.GetElemType() == ElemType::S8) ||
        (src.GetElemType() == ElemType::U32) ||
        (src.GetElemType() == ElemType::S32) ||
        (src.GetElemType() == ElemType::F16))
    {
        AURA_LOGE(ctx, AURA_TAG, "Input formats not supported by CvMedian\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::medianBlur(cv_src, cv_dst, ksize);
#else
    AURA_UNUSED(dst);
    AURA_UNUSED(ksize);
#endif

    return Status::OK;
}

class MedianTest : public TestBase<MedianParam::TupleTable, MedianParam::Tuple>
{
public:
    MedianTest(Context *ctx, MedianParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in MedianTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        MedianParam run_param(GetParam((index)));

        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (2 == run_param.mat_size.m_sizes.m_channel)
                {
                    return Status::OK;
                }
                else
                {
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        MedianParam run_param(GetParam((index)));

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_size.m_sizes);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes);
        Mat ref = m_factory.GetEmptyMat(((ElemType::F16 == run_param.elem_type) &&
                                         (TargetType::NONE == run_param.target.m_type))
                                         ? ElemType::F32 : run_param.elem_type, run_param.mat_size.m_sizes);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "ksize:" + std::to_string(run_param.ksize);
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IMedian, m_ctx, src, dst,
                                      run_param.ksize, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            std::cout << m_ctx->GetLogger()->GetErrorString() << std::endl;
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail\n");
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            Mat src_cv;
            src_cv = m_factory.GetDerivedMat(1.0f, 0.0f, (ElemType::F16 == run_param.elem_type ?
                                                          ElemType::F32 : run_param.elem_type),
                                                          run_param.mat_size.m_sizes);
            status_exec = Executor(10, 2, time_val, CvMedian, m_ctx, src_cv, ref, run_param.ksize);
            m_factory.PutMats(src_cv);
            result.accu_benchmark = "OpenCV::Median";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvMedian execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IMedian(m_ctx, src, ref, run_param.ksize, TargetType::NONE);
            result.accu_benchmark = "Median(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 1) == Status::OK)
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MEDIAN_UINT_TEST_HPP__
