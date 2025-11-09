#ifndef AURA_OPS_BILATERAL_UINT_TEST_HPP__
#define AURA_OPS_BILATERAL_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct BilateralTestParam
{
    BilateralTestParam()
    {}

    BilateralTestParam(DT_F32 sigma_color, DT_F32 sigma_space, DT_S32 ksize) : sigma_color(sigma_color),
                       sigma_space(sigma_space), ksize(ksize)
    {}

    friend std::ostream& operator<<(std::ostream &os, const BilateralTestParam &bilateral_test_param)
    {
        os << "ksize:" << bilateral_test_param.ksize << " | color_sigma:" << bilateral_test_param.sigma_color
           << " | space_sigma:" << bilateral_test_param.sigma_space;
        return os;
    }

    std::string ToString() const
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    DT_F32 sigma_color;
    DT_F32 sigma_space;
    DT_S32 ksize;
};

AURA_TEST_PARAM(BilateralParam,
                ElemType,           elem_type,
                MatSize,            mat_size,
                BilateralTestParam, bilateral_test_param,
                BorderType,         border_type,
                OpTarget,           target);

static Status CvBilateralFilter(Context *ctx, Mat &src, Mat &dst, BilateralTestParam bilateral_test_param,
                                      BorderType border_type)
{
    Status status = Status::OK;

    if (src.GetElemType() != ElemType::U8 && src.GetElemType() != ElemType::F32)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV BilateralFilter not support\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    DT_S32 cv_border_type = BorderTypeToOpencv(border_type);

    DT_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    DT_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);

    if (src_cv_type != -1 && dst_cv_type != -1 && cv_border_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::bilateralFilter(cv_src, cv_dst, bilateral_test_param.ksize, bilateral_test_param.sigma_color,
                            bilateral_test_param.sigma_space, cv_border_type);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV BilateralFilter not support\n");
        status = Status::ERROR;
    }
#else
    AURA_UNUSED(dst);
    AURA_UNUSED(bilateral_test_param);
    AURA_UNUSED(border_type);
#endif

    return status;
}

class BilateralTest : public TestBase<BilateralParam::TupleTable, BilateralParam::Tuple>
{
public:
    BilateralTest(Context *ctx, BilateralParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in BilateralTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        BilateralParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type)
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        BilateralParam run_param(GetParam((index)));
        ElemType src_elem_type = run_param.elem_type;
        ElemType dst_elem_type = run_param.elem_type;
        MatSize mat_sizes      = run_param.mat_size;

        BilateralTestParam bilateral_test_param = run_param.bilateral_test_param;
        AURA_LOGD(m_ctx, AURA_TAG, "bilateral param detail: elem_type(%s, %s), mat_size(%d, %d, %d), bordertype(%s), test_param(%s) \n",
                  ElemTypesToString(src_elem_type).c_str(), ElemTypesToString(dst_elem_type).c_str(),
                  mat_sizes.m_sizes.m_channel, mat_sizes.m_sizes.m_height, mat_sizes.m_sizes.m_width,
                  BorderTypeToString(run_param.border_type).c_str(), bilateral_test_param.ToString().c_str());

        Mat src = m_factory.GetDerivedMat(0.77f, 0.0f, src_elem_type, mat_sizes.m_sizes, AURA_MEM_DEFAULT, mat_sizes.m_strides);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, mat_sizes.m_sizes, AURA_MEM_DEFAULT, mat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(dst_elem_type, mat_sizes.m_sizes, AURA_MEM_DEFAULT, mat_sizes.m_strides);

        Scalar border_val = Scalar(0, 0, 0, 0);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        DT_F32 tolerance = 1.0f;

        result.param  = BorderTypeToString(run_param.border_type) + " | "+ bilateral_test_param.ToString();
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        Status status_exec = Executor(loop_count, 2, time_val, IBilateral, m_ctx, src, dst,
                                      bilateral_test_param.sigma_color, bilateral_test_param.sigma_space,
                                      bilateral_test_param.ksize, run_param.border_type, border_val, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail\n %s \n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;

            goto EXIT;
        }

        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec = Executor(10, 2, time_val, CvBilateralFilter, m_ctx, src, ref, bilateral_test_param, run_param.border_type);
            result.accu_benchmark = "OpenCV::BilateralFilter";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvBilateralFilter execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IBilateral(m_ctx, src, ref, bilateral_test_param.sigma_color, bilateral_test_param.sigma_space,
                                     bilateral_test_param.ksize, run_param.border_type, border_val, TargetType::NONE);

            result.accu_benchmark = "BilateralFilter(target::none)";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        if (MatCompare(m_ctx, dst, ref, cmp_result, tolerance) == Status::OK)
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

#endif // AURA_OPS_BILATERAL_UINT_TEST_HPP__