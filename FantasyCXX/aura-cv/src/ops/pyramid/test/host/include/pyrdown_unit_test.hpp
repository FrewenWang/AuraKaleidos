/** @brief      : pyrdown unit test head for aura
 *  @file       : pyrdown_unit_test.hpp
 *  @author     : lvxia@xiaomi.com, lizhiyu@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Jun. 29, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_PYRDOWN_UINT_TEST_HPP__
#define AURA_OPS_PYRDOWN_UINT_TEST_HPP__

#include "aura/ops/pyramid.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct PrydownTestParam
{
    PrydownTestParam()
    {}

    PrydownTestParam(DT_S32 ksize, DT_F32 sigma) : ksize(ksize), sigma(sigma)
    {}

    friend std::ostream& operator<<(std::ostream &os, const PrydownTestParam &pyrdown_test_param)
    {
        os << "ksize:" << pyrdown_test_param.ksize << " | sigma:" << pyrdown_test_param.sigma;
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

AURA_TEST_PARAM(PyrDownParam,
                ElemType,             elem_type,
                std::vector<MatSize>, mat_size,
                PrydownTestParam,     kernel_param,
                BorderType,           border_type,
                OpTarget,             target);

static Status CvPyrDown(Context *ctx, Mat &src, Mat &dst,
                              PrydownTestParam &kernel_param, BorderType &border_type)
{
#if !defined(AURA_BUILD_XPLORER)
    if (kernel_param.ksize != 5 || Abs(kernel_param.sigma - 0.f) > 1e-10)
    {
        AURA_LOGE(ctx, AURA_TAG, "OpenCV unsupport the ksize\n");
        return Status::ERROR;
    }
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_ref = MatToOpencv(dst);
    pyrDown(cv_src, cv_ref, cv::Size(cv_ref.cols, cv_ref.rows), BorderTypeToOpencv(border_type));
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(kernel_param);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

class PyrDownTest : public TestBase<PyrDownParam::TupleTable, PyrDownParam::Tuple>
{
public:
    PyrDownTest(Context *ctx, PyrDownParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487, 1});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in PyrDownTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        PyrDownParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    1 == run_param.mat_size[0].m_sizes.m_channel &&
                    run_param.mat_size[0].m_sizes.m_width < 800 &&
                    run_param.mat_size[0].m_sizes.m_height < 600)
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
        PyrDownParam run_param(GetParam(index));
        ElemType elem_type = run_param.elem_type;
        MatSize imat_size = run_param.mat_size[0];
        MatSize omat_size = run_param.mat_size[1];
        PrydownTestParam kernel_param = run_param.kernel_param;
        AURA_LOGD(m_ctx, AURA_TAG, "pyrdown param detail: elem_type(%s), imat_size(%s), omat_size(%s), kernel_param(%s) border_type(%s)\n",
                  ElemTypesToString(elem_type).c_str(), imat_size.ToString().c_str(), omat_size.ToString().c_str(),
                  kernel_param.ToString().c_str(), BorderTypeToString(run_param.border_type).c_str());

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, elem_type, imat_size.m_sizes, AURA_MEM_DEFAULT, imat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(elem_type, omat_size.m_sizes, AURA_MEM_DEFAULT, omat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(elem_type, omat_size.m_sizes, AURA_MEM_DEFAULT, omat_size.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        DT_F32 tolerance = 1.0f;

        result.param  = BorderTypeToString(run_param.border_type) + " | " + kernel_param.ToString().c_str();
        result.input  = imat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = omat_size.ToString() + " " + ElemTypesToString(elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IPyrDown, m_ctx, src, dst,
                                                  kernel_param.ksize, kernel_param.sigma, run_param.border_type, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
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
            status_exec = Executor(10, 2, time_val, CvPyrDown, m_ctx, src, ref, kernel_param, run_param.border_type);
            result.accu_benchmark = "OpenCV::PyrDown";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvPyrDown execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IPyrDown(m_ctx, src, ref, kernel_param.ksize, kernel_param.sigma, run_param.border_type, TargetType::NONE);
            result.accu_benchmark = "PyrDown(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, tolerance);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();
EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutMats(src, dst, ref);
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_PYRDOWN_UINT_TEST_HPP__