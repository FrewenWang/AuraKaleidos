#ifndef AURA_OPS_MATRIX_DCT_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_DCT_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(DctParam,
                ElemType,    elem_type,
                MatSize,     mat_size,
                OpTarget,    target);

static Status OpenCVDct(Mat &src, Mat &dst)
{
    Sizes3 src_sz = src.GetSizes();
    if((src_sz.m_height & 1) || (src_sz.m_width & 1))
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::dct(cv_src, cv_dst);
#else
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

static Status OpenCVIDct(Mat &src, Mat &dst)
{
    Sizes3 src_sz = src.GetSizes();
    if((src_sz.m_height & 1) || (src_sz.m_width & 1))
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::idct(cv_src, cv_dst);
#else
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class MatrixDctTest : public TestBase<DctParam::TupleTable, DctParam::Tuple>
{
public:
    MatrixDctTest(Context *ctx, DctParam::TupleTable &table):TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        DctParam run_param(GetParam(index));
        AURA_LOGD(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src   = m_factory.GetRandomMat(-1000.f, 1000.f, run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // OpenCV need float input
        Mat cv_src = m_factory.GetEmptyMat(aura::ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // Create dst/ref mat
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        TestTime time_val;
        MatCmpResult cmp_res;
        TestResult res;
        res.param  = "";
        res.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);
        res.output = run_param.mat_size.ToString() + " " + "F32";

        // run interface
        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);

        Status status_exec = Executor(loop_count, 2, time_val, IDct, m_ctx, src, dst, run_param.target);

        if (Status::OK == status_exec)
        {
            res.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            res.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            res.perf_status = TestStatus::FAILED;
            res.accu_status = TestStatus::FAILED;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            res.accu_benchmark = "OpenCV::DCT";
            status_exec = IConvertTo(m_ctx, src, cv_src);
            if (status_exec != aura::Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "aura::ConvertTo for OpenCVDct execute fail\n");
                res.accu_status = aura::TestStatus::UNTESTED;
                goto EXIT;
            }

            status_exec = Executor(loop_count, 2, time_val, OpenCVDct, cv_src, ref);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVDct just only support even size mat\n");
                res.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            res.perf_result["OpenCV"] = time_val;
            res.accu_benchmark = "OpenCV";
        }
        else
        {
            res.accu_benchmark = "aura::Dct(impl::none)";
            status_exec = IDct(m_ctx, src, ref, TargetType::NONE);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none just only support even size mat\n");
                res.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        if (Status::OK == MatCompare(m_ctx, dst, ref, cmp_res, 0.1f))
        {
            res.accu_status = cmp_res.status ? TestStatus::PASSED : TestStatus::FAILED;
            res.accu_result = cmp_res.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }
EXIT:
        test_case->AddTestResult(res.accu_status && res.perf_status, res);
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

AURA_TEST_PARAM(IDctParam,
                ElemType, elem_type,
                MatSize,  mat_size,
                OpTarget, target);

class MatrixIDctTest : public TestBase<IDctParam::TupleTable, IDctParam::Tuple>
{
public:
    MatrixIDctTest(Context *ctx, IDctParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        IDctParam run_param(GetParam(index));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src = m_factory.GetRandomMat(-1000.f, 1000.f, ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        TestTime time_val;
        MatCmpResult cmp_res;
        TestResult res;
        MI_F32 tolerance = (ElemType::F32 == run_param.elem_type) ? 0.1f : (ElemType::F16 == run_param.elem_type) ? 2.f : 1.f;

        res.param  = "";
        res.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(ElemType::F32);
        res.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);

        Status status_exec = Executor(loop_count, 2, time_val, IInverseDct, m_ctx, src, dst, run_param.target);
        if (Status::OK == status_exec)
        {
            res.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            res.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            res.perf_status = TestStatus::FAILED;
            res.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            ElemType dst_elem_type = dst.GetElemType();
            res.accu_benchmark = "OpenCV::IDct";
            if (ElemType::F32 == dst_elem_type)
            {
                // Idct in 0pencv only supports F32 and F64 in dst mat
                status_exec = Executor(10, 2, time_val, OpenCVIDct, src, ref);
                if (status_exec != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVIDct just only support even size mat\n");
                    res.accu_status = TestStatus::UNTESTED;
                    goto EXIT;
                }
                res.perf_result["OpenCV"] = time_val;
            }
            else
            {
                res.accu_benchmark = "OpenCV::IDct";
                // Idct in 0pencv only supports F32 and F64 in dst mat
                Mat ref_f32 = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
                status_exec = Executor(10, 2, time_val, OpenCVIDct, src, ref_f32);
                if (status_exec != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVIDct just only support even size mat\n");
                    res.accu_status = TestStatus::UNTESTED;
                    goto EXIT;
                }
                res.perf_result["OpenCV"] = time_val;
                status_exec = IConvertTo(m_ctx, ref_f32, ref);
                if (status_exec != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "benchmark ConvertTot execute fail\n");
                    res.accu_status = TestStatus::UNTESTED;
                    goto EXIT;
                }
            }
        }
        else
        {
            res.accu_benchmark = "aura::IDct(impl::none)";
            status_exec = IInverseDct(m_ctx, src, ref, aura::OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                res.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        MatCompare(m_ctx, dst, ref, cmp_res, tolerance);
        res.accu_status = cmp_res.status ? TestStatus::PASSED : TestStatus::FAILED;
        res.accu_result = cmp_res.ToString();
EXIT:
        test_case->AddTestResult(res.accu_status && res.perf_status, res);
        m_factory.PutAllMats();

        return 0;
    }
private:
    aura::Context   *m_ctx;
    aura::MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_DCT_UNIT_TEST_HPP__