#ifndef AURA_OPS_BOXFILTER_UNIT_TEST_HPP__
#define AURA_OPS_BOXFILTER_UNIT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(BoxFilterParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                DT_S32,     ksize,
                BorderType, border_type,
                OpTarget,   target);

static Status CvBoxFilter(Context *ctx, Mat &src, Mat &ref, DT_S32 ksize, BorderType border_type)
{
    if (src.GetElemType() == ElemType::F16)
    {
        AURA_LOGE(ctx, AURA_TAG, "CvBoxFilter does not support f16\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat src_mat = MatToOpencv(src);
    cv::Mat ref_mat = MatToOpencv(ref);
    cv::boxFilter(src_mat, ref_mat, ref_mat.depth(), cv::Size(ksize, ksize), cv::Point(-1, -1), true, BorderTypeToOpencv(border_type));
#else
    AURA_UNUSED(ref);
    AURA_UNUSED(ksize);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

class BoxFilterTest : public TestBase<BoxFilterParam::TupleTable, BoxFilterParam::Tuple>
{
public:
    BoxFilterTest(Context *ctx, BoxFilterParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in BoxFilterTest\n");
        }

    }

    Status CheckParam(DT_S32 index) override
    {
        BoxFilterParam run_param(GetParam((index)));

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
        BoxFilterParam run_param(GetParam((index)));
        AURA_LOGD(m_ctx, AURA_TAG, "boxfilter param: %s\n",  run_param.ToString().c_str());

        ElemType elem_type = run_param.elem_type;
        MatSize mat_size = run_param.mat_size;

        AURA_LOGD(m_ctx, AURA_TAG, "boxfilter param detail: elem_type(%s), mat_size(%s), kernel_size(%d), border_type(%s)\n",
                  ElemTypesToString(elem_type).c_str(), mat_size.ToString().c_str(),
                  run_param.ksize, BorderTypeToString(run_param.border_type).c_str());

        if (TargetType::OPENCL == run_param.target.m_type && run_param.ksize >= 7 && run_param.mat_size.m_sizes.m_channel > 1)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "opencl target cannot support channel > 1 when ksize >= 7! \n");
            return 0;
        }

        if (run_param.ksize >= run_param.mat_size.m_sizes.m_height || run_param.ksize >= run_param.mat_size.m_sizes.m_width)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "kernel size must less than height or width! \n");
            return 0;
        }

        if ((run_param.ksize > 128) && ((run_param.mat_size.m_sizes.m_height >= 1024) ||
            (run_param.mat_size.m_sizes.m_width >= 2048)))
        {
            AURA_LOGD(m_ctx, AURA_TAG, "only test small size testcases when kernel size is greater than 128 \n");
            return 0;
        }

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0, 0, run_param.elem_type, run_param.mat_size.m_sizes);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes);
        Mat ref = m_factory.GetEmptyMat(((ElemType::F16 == run_param.elem_type) && (TargetType::NONE == run_param.target.m_type))
                                        ? ElemType::F32 : run_param.elem_type, run_param.mat_size.m_sizes);

        DT_S32 loop_count = stress_count ? stress_count : 10;
        Scalar border_value = Scalar(0, 0, 0, 0);

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = BorderTypeToString(run_param.border_type) + " | ksize:" + std::to_string(run_param.ksize);
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IBoxfilter, m_ctx, src, dst,
                                      run_param.ksize, run_param.border_type, border_value, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, err info: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
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

            status_exec = Executor(10, 2, time_val, CvBoxFilter, m_ctx, src_cv, ref, run_param.ksize, run_param.border_type);

            m_factory.PutMats(src_cv);
            result.accu_benchmark = "OpenCV::boxFilter";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvBoxFilter execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IBoxfilter(m_ctx, src, ref, run_param.ksize, run_param.border_type, border_value, TargetType::NONE);
            result.accu_benchmark = "Boxfilter(target::none)";

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

#endif // AURA_OPS_BOXFILTER_UNIT_TEST_HPP__
