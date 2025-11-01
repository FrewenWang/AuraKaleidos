#ifndef AURA_OPS_MATRIX_DFT_IDFT_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_DFT_IDFT_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

#if !defined(MATSIZEPAIR)
#define MATSIZEPAIR
using MatSizePair = std::pair<MatSize, MatSize>;
static std::ostream& operator << (std::ostream &os, MatSizePair size_pair)
{
    os << "src mat size : " << size_pair.first << " dst mat size : " << size_pair.second << std::endl;
    return os;
}
#endif // MATSIZEPAIR

struct MixedDiff
{
    MI_F64 operator()(const MI_F64 val0, const MI_F64 val1) const
    {
        RelativeDiff relative_diff;
        AbsDiff abs_diff;

        return Min(abs_diff(val0, val1), relative_diff(val0, val1) * 100);
    }

    static std::string ToString()
    {
        return "MixedDiff";
    }
};

AURA_TEST_PARAM(DftParam,
                ElemType,     elem_type,
                MatSizePair,  mat_size,
                OpTarget,     target);

AURA_TEST_PARAM(IDftParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                MI_S32,     dst_channels,
                OpTarget,   target);

AURA_INLINE Status OpenCVDft(Mat &src, Mat &dst)
{
    if (ElemType::F16 == src.GetElemType() || ElemType::U32 == src.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::dft(cv_src, cv_dst, cv::DFT_COMPLEX_OUTPUT);
#else
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

AURA_INLINE Status OpenCVIDft(Mat &src, Mat &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    Sizes3 dst_size = dst.GetSizes();
    MI_S32 height   = dst_size.m_height;

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst;

    if (1 == dst_size.m_channel)
    {
        cv::Mat channels[2];
        cv::idft(cv_src, cv_dst, cv::DFT_SCALE);
        cv::split(cv_dst, channels);

        for(MI_S32 h = 0; h < height; h++)
        {
            MI_F32 *dst_row = dst.Ptr<MI_F32>(h);
            MI_F32 *cv_row  = (MI_F32 *)(channels[0].data + channels[0].step * h);

            memcpy(dst_row, cv_row, channels[0].step);
        }
    }
    else
    {
        cv_dst = MatToOpencv(dst);
        cv::idft(cv_src, cv_dst, cv::DFT_SCALE);
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class MatrixDftTest : public TestBase<DftParam::TupleTable, DftParam::Tuple>
{
public:
    MatrixDftTest(Context *ctx, DftParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        DftParam run_param(GetParam((index)));
        AURA_LOGD(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src    = m_factory.GetRandomMat(-1000, 1000, run_param.elem_type, run_param.mat_size.first.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.first.m_strides);
        Mat cv_src = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.first.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.first.m_strides); // opencv need float input
        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = "";
        result.input  = run_param.mat_size.first.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.second.ToString() + " " + "F32";

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IDft, m_ctx, src, dst, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::Dft";
            status_exec           = IConvertTo(m_ctx, src, cv_src, 1, 0, OpTarget::None());
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "ConvertTo for OpenCVDft execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            status_exec = Executor(10, 2, time_val, OpenCVDft, cv_src, ref);
            result.perf_result["OpenCV"] = time_val;
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVDft execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
        }
        else
        {
            result.accu_benchmark = "Dft(target::none)";
            status_exec = IDft(m_ctx, src, ref, TargetType::NONE);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare<MixedDiff>(m_ctx, dst, ref, cmp_result, 2, 1) == Status::OK)
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
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context   *m_ctx;
    MatFactory m_factory;
};

class MatrixIDftTest : public TestBase<IDftParam::TupleTable, IDftParam::Tuple>
{
public:
    MatrixIDftTest(Context *ctx, IDftParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    Status CheckParam(MI_S32 index) override
    {
        IDftParam run_param(GetParam((index)));
        if (2 == run_param.dst_channels && run_param.elem_type != ElemType::F32)
        {
            return Status::ERROR;
        }

        return Status::OK;
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        IDftParam run_param(GetParam((index)));
        AURA_LOGD(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src = m_factory.GetRandomMat(-1000, 1000, ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // Create dst mat
        Sizes dst_strides;
        dst_strides.m_height                 = run_param.mat_size.m_strides.m_height;
        dst_strides.m_width                  = run_param.mat_size.m_strides.m_width / ((1 == run_param.dst_channels) ? 2 : 1);
        run_param.mat_size.m_sizes.m_channel = run_param.dst_channels;

        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, dst_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, dst_strides);

        MI_S32 loop_count = stress_count ? stress_count : 
                            (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = "";
        result.input  = src.GetSizes().ToString() + " " + ElemTypesToString(ElemType::F32);
        result.output = dst.GetSizes().ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IInverseDft, m_ctx, src, dst, MI_TRUE, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::IDft";

            Mat cv_ref;
            if (ElemType::F32 == run_param.elem_type)
            {
                cv_ref = ref;
            }
            else
            {
                cv_ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, dst_strides);
            }

            status_exec = Executor(10, 2, time_val, OpenCVIDft, src, cv_ref);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVIDft execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            status_exec = IConvertTo(m_ctx, cv_ref, ref, 1, 0, OpTarget::None());
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "ConvertTo for OpenCVIDft execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "IDft(target::none)";
            status_exec = IInverseDft(m_ctx, src, ref, MI_TRUE, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 1.0) == Status::OK)
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
        m_factory.PutAllMats();

        return 0;
    }
private:
    Context   *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_DFT_IDFT_UNIT_TEST_HPP__
