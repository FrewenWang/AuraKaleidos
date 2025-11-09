/** @brief      : resize uint test head for aura
 *  @file       : resize_unit_test.hpp
 *  @author     : xianan@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 11, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_RESIZE_UINT_TEST_HPP__
#define AURA_OPS_RESIZE_UINT_TEST_HPP__

#include "aura/ops/resize.h"
#include "aura/tools/unit_test.h"

#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/cl_mem.h"
#endif // AURA_ENABLE_OPENCL

using namespace aura;

AURA_TEST_PARAM(ResizeParam,
                ElemType,   elem_type,
                MatSize,    imat_sizes,
                Scalar,     scale,
                InterpType, interp_type,
                OpTarget,   target);
#if !defined(AURA_BUILD_XPLORER)
static DT_S32 ResizeInterpTypeToOpencv(InterpType method)
{
    DT_S32 cv_type = -1;

    switch (method)
    {
        case InterpType::NEAREST:
        {
            cv_type = cv::INTER_NEAREST;
            break;
        }

        case InterpType::LINEAR:
        {
            cv_type = cv::INTER_LINEAR;
            break;
        }

        case InterpType::CUBIC:
        {
            cv_type = cv::INTER_CUBIC;
            break;
        }

        case InterpType::AREA:
        {
            cv_type = cv::INTER_AREA;
            break;
        }

        default:
        {
            cv_type = -1;
            break;
        }
    }

    return cv_type;
}
#endif
static Status CvResize(Context *ctx, Mat &src, Mat &dst, InterpType type)
{
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    DT_S32 cv_method = ResizeInterpTypeToOpencv(type);

    DT_S32 cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    DT_S32 cn = src.GetSizes().m_channel;

    if ((((CV_32SC(cn) == cv_type) || (CV_8SC(cn)  == cv_type)) && (InterpType::AREA   == type)) ||
        (((CV_8SC(cn)  == cv_type) || (CV_32SC(cn) == cv_type)) && (InterpType::CUBIC  == type)) ||
        (((CV_8SC(cn)  == cv_type) || (CV_32SC(cn) == cv_type)) && (InterpType::LINEAR == type)))
    {
        cv_type = -1;
        cv_method = -1;
    }

    if (cv_type != -1 && cv_method != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        DT_S32 owidth  = dst.GetSizes().m_width;
        DT_S32 oheight = dst.GetSizes().m_height;
        cv::resize(cv_src, cv_dst, cv::Size(owidth, oheight), 0, 0, cv_method);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV resize not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
#endif

    return ret;
}

AURA_INLINE Sizes DefaultStride(Sizes3 mat_size, ElemType type)
{
    DT_S32 elem_size = ElemTypeSize(type);
    DT_S32 width   = mat_size.m_width;
    DT_S32 height  = mat_size.m_height;
    DT_S32 channel = mat_size.m_channel;

    return Sizes(height, width * channel * elem_size);
}

class ResizeTest : public TestBase<ResizeParam::TupleTable, ResizeParam::Tuple>
{
public:
    ResizeTest(Context *ctx, ResizeParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 1024)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv",      ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",     ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        ResizeParam run_param(GetParam((index)));

        Status ret = Status::OK;
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (2 == run_param.imat_sizes.m_sizes.m_channel)
                {
                    ret |= Status::OK;
                }
                else
                {
                    ret |= Status::ERROR;
                }
            }
            else
            {
                ret |= Status::OK;
            }
        }

        const Sizes3 &src_sz = run_param.imat_sizes.m_sizes;
        const Sizes3 &dst_sz = MatSize(run_param.imat_sizes, run_param.scale).m_sizes;

        if ((InterpType::AREA   == run_param.interp_type)   &&
            (TargetType::OPENCL == run_param.target.m_type) &&
            (src_sz.m_width > dst_sz.m_width && src_sz.m_height > dst_sz.m_height)) // when CL AREA mode, down sample only support 2x 4x
        {
            if ((src_sz.m_width == 2 * dst_sz.m_width) && (src_sz.m_height == 2 * dst_sz.m_height))
            {
                ret |= Status::OK;
            }
            else if ((src_sz.m_width == 4 * dst_sz.m_width) && (src_sz.m_height == 4 * dst_sz.m_height))
            {
                ret |= Status::OK;
            }
            else
            {
                AURA_LOGI(m_ctx, AURA_TAG, "ResizeAerea current only support 2x 4x down sample.\n");
                ret |= Status::ERROR;
            }
        }
        else
        {
            ret |= Status::OK;
        }

        return ret;
    }

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        ResizeParam run_param(GetParam((index)));
        AURA_LOGD(m_ctx, AURA_TAG, "Resize param: %s\n", run_param.ToString().c_str());

        ElemType elem_type = run_param.elem_type;
        MatSize imat_sizes = run_param.imat_sizes;
        MatSize omat_sizes(imat_sizes, run_param.scale);
        Sizes omat_strides = DefaultStride(omat_sizes.m_sizes, elem_type) + (imat_sizes.m_strides.m_width > 0 ? Sizes(1, 4): Sizes());

        // creat iauras
        DT_BOOL promote_fp16 = (ElemType::F16 == elem_type) && (TargetType::NONE == run_param.target.m_type);
        ElemType ref_elem_type = promote_fp16 ? ElemType::F32 : elem_type;

        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, elem_type, imat_sizes.m_sizes, AURA_MEM_DEFAULT, imat_sizes.m_strides);
        Mat dst = m_factory.GetEmptyMat(elem_type,     omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_strides);
        Mat ref = m_factory.GetEmptyMat(ref_elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        std::ostringstream oss;
        oss << run_param.scale.m_val[0] << "," << run_param.scale.m_val[1];
        result.param  = InterpTypeToString(run_param.interp_type) + ":" + oss.str();
        result.input  = imat_sizes.ToString() + " " + ElemTypesToString(elem_type);
        result.output = omat_sizes.ToString();
        // run interface
        Status ret = Executor(loop_count, 2, time_val, IResize, m_ctx, src, dst, run_param.interp_type, run_param.target);

        if (Status::OK == ret)
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
            Mat src_cv;
            ElemType cv_elem_type = promote_fp16 ? ElemType::F32 : elem_type;
            src_cv = m_factory.GetDerivedMat(1.0f, 0.0f, cv_elem_type, imat_sizes.m_sizes, AURA_MEM_DEFAULT, imat_sizes.m_strides);

            ret = Executor(10, 2, time_val, CvResize, m_ctx, src_cv, ref, run_param.interp_type);
            result.accu_benchmark = "OpenCV::Resize";

            m_factory.PutMats(src_cv);

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvResize execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IResize(m_ctx, src, ref, run_param.interp_type, TargetType::NONE);
            result.accu_benchmark = "Resize(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        MatCompare(m_ctx, dst, ref, cmp_result, 2);

        if (TargetType::OPENCL == run_param.target.m_type && InterpType::NEAREST == run_param.interp_type)
        {
            // float compute error because none impl use -Ofast flag
            if (!cmp_result.status)
            {
                DT_F32 ratio = cmp_result.hist.back().second * 1.0f / cmp_result.total;

                if (ratio > 0.995f)
                {
                    result.accu_status = TestStatus::PASSED;
                }
                else
                {
                    result.accu_status = TestStatus::FAILED;
                }
            }
        }
        else
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        }
        result.accu_result = cmp_result.ToString();
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

#endif // AURA_OPS_RESIZE_UINT_TEST_HPP__