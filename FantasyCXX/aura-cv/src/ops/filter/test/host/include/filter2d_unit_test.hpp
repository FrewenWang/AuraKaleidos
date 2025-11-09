#ifndef AURA_OPS_FILTER2D_UINT_TEST_HPP__
#define AURA_OPS_FILTER2D_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(Filter2dParam,
                MatElemPair, elem_type,
                MatSize,     mat_size,
                DT_S32,      ksize,
                BorderType,  border_type,
                OpTarget,    target);

static Status CvFilter2D(Context *ctx, Mat &src, Mat &dst, Mat &kernel, DT_F64 delta, BorderType border_type)
{
    AURA_UNUSED(ctx);

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat src_mat = MatToOpencv(src);
    cv::Mat ref_mat = MatToOpencv(dst);
    cv::Mat kernel_mat = MatToOpencv(kernel);
    cv::filter2D(src_mat, ref_mat, ref_mat.depth(), kernel_mat, cv::Point(-1, -1), delta, BorderTypeToOpencv(border_type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(kernel);
    AURA_UNUSED(delta);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

class Filter2dTest : public TestBase<Filter2dParam::TupleTable, Filter2dParam::Tuple>
{
public:
    Filter2dTest(Context *ctx, Filter2dParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in Filter2dTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        Filter2dParam run_param(GetParam((index)));
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
        Filter2dParam run_param(GetParam((index)));
        ElemType src_elem_type = run_param.elem_type.first;
        ElemType dst_elem_type = run_param.elem_type.second;
        MatSize  mat_size = run_param.mat_size;
        AURA_LOGD(m_ctx, AURA_TAG, "filter2d param detail: elem_type(%s, %s), mat_size(%d, %d, %d), ksize(%d), bordertype(%s) \n",
                  ElemTypesToString(src_elem_type).c_str(), ElemTypesToString(dst_elem_type).c_str(),
                  mat_size.m_sizes.m_channel, mat_size.m_sizes.m_height, mat_size.m_sizes.m_width,
                  run_param.ksize, BorderTypeToString(run_param.border_type).c_str());

        if ((TargetType::OPENCL == run_param.target.m_type) && (run_param.ksize >= 7) && (mat_size.m_sizes.m_channel > 1))
        {
            AURA_LOGD(m_ctx, AURA_TAG, "opencl impl cannot support channel > 1 when ksize >= 7! \n");
            return 0;
        }

        if ((TargetType::HVX == run_param.target.m_type) && (7 == run_param.ksize) && (3 == mat_size.m_sizes.m_channel))
        {
            AURA_LOGD(m_ctx, AURA_TAG, "hvx impl cannot support channel = 3 when ksize = 7! \n");
            return 0;
        }

        // creat iauras
        DT_BOOL promote_fp16 = ElemType::F16 == src_elem_type && TargetType::NONE == run_param.target.m_type;
        ElemType ref_elem_type = promote_fp16 ? ElemType::F32 : dst_elem_type;

        const Sizes3 ker_sizes(run_param.ksize, run_param.ksize);
        Mat kernel = m_factory.GetRandomMat(0.f, 1.0f, ElemType::F32, ker_sizes);

        Mat src = m_factory.GetDerivedMat(.2f, 0.f, src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(ref_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Scalar border_value{0, 0, 0, 0};

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        DT_F32 delta = 0.0f;

        result.param  = BorderTypeToString(run_param.border_type) + " | ksize:" + std::to_string(run_param.ksize);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IFilter2d, m_ctx, src, dst, kernel,
                                      run_param.border_type, border_value, run_param.target);
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
            ElemType cv_src_elem_type = promote_fp16 ? ElemType::F32 : src_elem_type;
            Mat src_cv = m_factory.GetDerivedMat(.2f, 0.f, cv_src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

            status_exec = Executor(10, 2, time_val, CvFilter2D, m_ctx, src_cv, ref, kernel, delta, run_param.border_type);
            result.accu_benchmark = "OpenCV::filter2D";
            m_factory.PutMats(src_cv);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvFilter2D execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IFilter2d(m_ctx, src, ref, kernel, run_param.border_type,
                                    border_value, TargetType::NONE);
            result.accu_benchmark = "Filter2D(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 1, 0.5) == Status::OK)
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
        m_factory.PutMats(src, dst, ref, kernel);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_FILTER2D_UINT_TEST_HPP__
