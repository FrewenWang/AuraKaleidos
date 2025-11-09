#ifndef AURA_OPS_GAUSSIAN_UINT_TEST_HPP__
#define AURA_OPS_GAUSSIAN_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct GaussianTestParam
{
    GaussianTestParam()
    {}

    GaussianTestParam(DT_S32 ksize, DT_F32 sigma) : ksize(ksize), sigma(sigma)
    {}

    friend std::ostream& operator<<(std::ostream &os, const GaussianTestParam &gaussian_test_param)
    {
        os << "ksize:" << gaussian_test_param.ksize << " | sigma:" << gaussian_test_param.sigma;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    DT_S32 ksize;
    DT_F32 sigma;
};

AURA_TEST_PARAM(GaussianParam,
                ElemType,          elem_type,
                MatSize,           mat_size,
                GaussianTestParam, kernel_param,
                BorderType,        border_type,
                OpTarget,          target);

static Status CvGaussian(Context *ctx, Mat &src, Mat &dst, GaussianTestParam &kernel_param, BorderType &border_type)
{
#if defined(__arm__)
    if (BorderType::CONSTANT == border_type)
    {
        AURA_LOGD(ctx, AURA_TAG, "CvGaussian unsupported border_type in arm32\n");
        return Status::ERROR;
    }
#endif // __arm__

#if !defined(AURA_BUILD_XPLORER)
    if (src.GetElemType() != ElemType::S8 && src.GetElemType() != ElemType::F16 && src.GetElemType() != ElemType::S32 && src.GetElemType() != ElemType::U32)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_ref = MatToOpencv(dst);
        cv::GaussianBlur(cv_src, cv_ref, cv::Size(kernel_param.ksize, kernel_param.ksize), kernel_param.sigma, kernel_param.sigma, BorderTypeToOpencv(border_type));
    }
    else
    {
        AURA_LOGD(ctx, AURA_TAG, "CvGaussian unsupported elem_type\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(kernel_param);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

class GaussianTest : public TestBase<GaussianParam::TupleTable, GaussianParam::Tuple>
{
public:
    GaussianTest(Context *ctx, GaussianParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in GaussianTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        GaussianParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    2 == run_param.mat_size.m_sizes.m_channel)
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
        // get next param set
        GaussianParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize mat_size = run_param.mat_size;
        GaussianTestParam gaussian_ker_param = run_param.kernel_param;
        AURA_LOGD(m_ctx, AURA_TAG, "gaussian param detail: elem_type(%s), mat_size(%s), kernel_param(%s), border_type(%s)\n",
                  ElemTypesToString(elem_type).c_str(), mat_size.ToString().c_str(),
                  gaussian_ker_param.ToString().c_str(), BorderTypeToString(run_param.border_type).c_str());

        if (TargetType::OPENCL == run_param.target.m_type && run_param.kernel_param.ksize >= 7 && mat_size.m_sizes.m_channel > 1)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "opencl target cannot support channel > 1 when ksize >= 7! \n");
            return 0;
        }

        // creat iauras
        Mat src;
        if ((ElemType::F32 == elem_type) || (ElemType::F16 == elem_type))
        {
            src = m_factory.GetRandomMat(0.f, 1024.f, elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        }
        else if ((ElemType::U16 == elem_type) || (ElemType::S16 == elem_type))
        {
            src = m_factory.GetRandomMat(0.f, 8191.f, elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        }
        else
        {
            src = m_factory.GetRandomMat(0.f, 4294967295.f, elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        }

        Mat dst = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(((ElemType::F16 == elem_type) && (TargetType::NONE == run_param.target.m_type))
                                        ? ElemType::F32 : elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;
        Scalar border_value = Scalar(0, 0, 0, 0);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        DT_F32 tolerance = (ElemType::F32 == elem_type) ? 0.5f : 1.0f;

        result.param  = BorderTypeToString(run_param.border_type) + " | " + gaussian_ker_param.ToString();
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IGaussian, m_ctx, src, dst,
                                      run_param.kernel_param.ksize, run_param.kernel_param.sigma,
                                      run_param.border_type, border_value, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec = Executor(10, 2, time_val, CvGaussian, m_ctx, src, ref, gaussian_ker_param, run_param.border_type);
            result.perf_result["OpenCV"] = time_val;
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvGaussian execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.accu_benchmark = "OpenCV::Gaussian";
        }
        else
        {
            status_exec = IGaussian(m_ctx, src, ref, gaussian_ker_param.ksize,
                                    gaussian_ker_param.sigma, run_param.border_type, border_value, TargetType::NONE);
            result.accu_benchmark = "Gaussian(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
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
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_GAUSSIAN_UINT_TEST_HPP__
