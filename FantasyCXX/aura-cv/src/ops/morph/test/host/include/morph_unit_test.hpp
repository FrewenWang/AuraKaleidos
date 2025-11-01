/** @brief      : morph uint test head for aura
 *  @file       : morph_unit_test.hpp
 *  @author     : liuguangxin1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 27, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MORPH_UINT_TEST_HPP__
#define AURA_OPS_MORPH_UINT_TEST_HPP__

#include "aura/ops/morph.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct MorphTestParam
{
    MorphTestParam()
    {}

    MorphTestParam(MI_S32 ksize, MI_S32 iterations) : ksize(ksize), iterations(iterations)
    {}

    friend std::ostream& operator<<(std::ostream &os, const MorphTestParam &morph_test_param)
    {
        os << "ksize:" << std::to_string(morph_test_param.ksize)
           << " | iterations:" << std::to_string(morph_test_param.iterations);
        return os;
    }

    std::string ToString()
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    MI_S32 ksize;
    MI_S32 iterations;
};

AURA_TEST_PARAM(MorphParam,
                ElemType,       elem_type,
                MatSize,        mat_size,
                MorphType,      type,
                MorphShape,     shape,
                MorphTestParam, morph_test_param,
                OpTarget,       target);

static Status CvMorphologyEx(Context *ctx, Mat &src, Mat &dst, MorphType type, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
{
    AURA_UNUSED(ctx);

#if !defined(AURA_BUILD_XPLORER)
    BorderType border_type = BorderType::REPLICATE;

    cv::MorphShapes cvshape    = static_cast<cv::MorphShapes>(shape);
    cv::Mat kernel             = getStructuringElement(cvshape, cv::Size(ksize, ksize), cv::Point(-1, -1));
    cv::Scalar cv_border_value = {0, 0, 0, 0};

    cv::Mat src_mat       = MatToOpencv(src);
    cv::Mat ref_mat       = MatToOpencv(dst);
    cv::MorphTypes cvtype = static_cast<cv::MorphTypes>(type);
    cv::morphologyEx(src_mat, ref_mat, cvtype, kernel, cv::Point(-1, -1),
                     iterations, BorderTypeToOpencv(border_type), cv_border_value);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
    AURA_UNUSED(ksize);
    AURA_UNUSED(shape);
    AURA_UNUSED(iterations);
#endif

    return Status::OK;
}

class MorphTest : public TestBase<MorphParam::TupleTable, MorphParam::Tuple>
{
public:
    MorphTest(Context *ctx, MorphParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in MorphTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        MorphParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (1 == run_param.mat_size.m_sizes.m_channel &&
                    run_param.mat_size.m_sizes.m_width  < 800 &&
                    run_param.mat_size.m_sizes.m_height < 600)
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
        MorphParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize &mat_size  = run_param.mat_size;
        MorphType type     = run_param.type;
        MorphShape shape   = run_param.shape;
        MI_S32 ksize       = run_param.morph_test_param.ksize;
        MI_S32 iterations  = run_param.morph_test_param.iterations;
        AURA_LOGD(m_ctx, AURA_TAG, "morph param detail: elem_type(%s), mat_size(%d, %d, %d), type(%s), shape(%s), morph_test_param(%s)\n",
                  ElemTypesToString(elem_type).c_str(), mat_size.m_sizes.m_channel, mat_size.m_sizes.m_height, mat_size.m_sizes.m_width,
                  MorphTypeToString(type).c_str(), MorphShapeToString(shape).c_str(), run_param.morph_test_param.ToString().c_str());

        // create iauras
        MI_BOOL promote = ElemType::F16 == elem_type && TargetType::NONE == run_param.target.m_type;
        ElemType cv_elem_type = promote ? ElemType::F32 : elem_type;
        Mat src = m_factory.GetDerivedMat(1.f, 0.f, elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(cv_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        // run interface
        result.param  = MorphTypeToString(type) + " | " + MorphShapeToString(shape) + " | " + run_param.morph_test_param.ToString();
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        Status status_exec = Executor(loop_count, 2, time_val, IMorphologyEx, m_ctx, src, dst, type, ksize, shape, iterations, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            Mat src_cv = m_factory.GetDerivedMat(1.f, 0.f, cv_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

            status_exec        = Executor(10, 2, time_val, CvMorphologyEx, m_ctx, src_cv, ref, type, ksize, shape, iterations);
            result.accu_benchmark = "OpenCV::MorphologyEx";
            m_factory.PutMats(src_cv);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvMorphologyEx execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec        = IMorphologyEx(m_ctx, src, ref, type, ksize, shape, iterations, TargetType::NONE);
            result.accu_benchmark = "MorphologyEx(target::none)";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (Status::OK == MatCompare(m_ctx, dst, ref, cmp_result, 1.f))
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

#endif // AURA_OPS_MORPH_UINT_TEST_HPP__
