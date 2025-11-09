/** @brief      : remap uint test head for aura
 *  @file       : remap_unit_test.hpp
 *  @author     : chenchunyu@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 30, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_REMAP_UINT_TEST_HPP__
#define AURA_OPS_REMAP_UINT_TEST_HPP__

#include "aura/ops/warp.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(RemapParam,
                ElemType,   src_elem_type,
                ElemType,   map_elem_type,
                MatSize,    mat_size,
                InterpType, interp_type,
                BorderType, border_type,
                OpTarget,   target);

static Status CvRemap(Mat &src, Mat &dst, Mat &map,
                      InterpType interp_type, BorderType border_type, Scalar &border_value)
{
#if !defined(AURA_BUILD_XPLORER)
    if (ElemType::S8  == src.GetElemType() || ElemType::F16 == src.GetElemType() ||
        ElemType::S32 == src.GetElemType() || ElemType::U32 == src.GetElemType())
    {
        return Status::ERROR;
    }

#if defined(ANDROID)
#  if !defined(__aarch64__)
    // there exists differences on rounding operation of convertTo Function in OPENCV and Aura projects,
    // differences usually occur when rounding data whose decimal is .5 (**.5)
    return Status::ERROR;
#  endif // __aarch64__
#endif // ANDROID

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::Mat cv_m0  = MatToOpencv(map);
    cv::Mat cv_m1  = cv::Mat();
    cv::Scalar cv_border_value(border_value.m_val[0], border_value.m_val[1], border_value.m_val[2], border_value.m_val[3]);

    cv::remap(cv_src, cv_dst, cv_m0, cv_m1, static_cast<DT_S32>(interp_type), BorderTypeToOpencv(border_type), cv_border_value);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(map);
    AURA_UNUSED(interp_type);
    AURA_UNUSED(border_type);
    AURA_UNUSED(border_value);
#endif

    return Status::OK;
}

static Status CreateMap(MatFactory &factory, Sizes3 src_sizes, ElemType map_elem_type, Mat &map)
{
    Sizes3 map_c2_sizes;
    map_c2_sizes           = src_sizes;
    map_c2_sizes.m_channel = 2;
    DT_F32 max_val_f32     = Min(Max(src_sizes.m_width, src_sizes.m_height) * 15.2f + 200.f, 32767.f);

    if (map_elem_type == ElemType::F32)
    {
        map = factory.GetRandomMat(0, max_val_f32, ElemType::F32, map_c2_sizes, AURA_MEM_DEFAULT);

        for (DT_S32 h = 0; h < src_sizes.m_height; h++)
        {
            DT_S32 h_group = h / 13;
            for (DT_S32 w = 0; w < src_sizes.m_width; w++)
            {
                DT_S32 w_group = w / 253;
                DT_F32 w_val   = (rand()% (int)max_val_f32) / 253.f;
                DT_F32 w_index = (w_val + w_group - ((DT_S32)w_val)) * 253.f;
                map.Ptr<DT_F32>(h)[2 * w] = w_index;
                DT_F32  h_val   = (rand()% (int)max_val_f32) / 13.f;
                DT_F32  h_index = (h_val + h_group - ((DT_S32)h_val)) * 13.f;
                h_index = pow(-1, rand()) * h_index;
                map.Ptr<DT_F32>(h)[2 * w + 1] = h_index;
            }
            map.Ptr<DT_F32>(h)[2 * src_sizes.m_width - 2] = pow(-1, rand()) * (rand()% (int)max_val_f32);
            map.Ptr<DT_F32>(h)[2 * src_sizes.m_width - 1] = pow(-1, rand()) * (rand()% (int)max_val_f32);
        }
    }
    else if (map_elem_type == ElemType::S16)
    {
        map = factory.GetRandomMat(0, max_val_f32, ElemType::S16, map_c2_sizes, AURA_MEM_DEFAULT);
        for(DT_S32 h = 0; h < src_sizes.m_height; h++)
        {
            DT_S32 h_group = h / 13;
            for(DT_S32 w = 0; w < src_sizes.m_width; w++)
            {
                DT_S32 w_group = w / 253;
                DT_F32 w_val   = (rand() % (int)max_val_f32) / 253.f;
                DT_S16 w_index = SaturateCast<DT_S16>((w_val + w_group - ((DT_S32)w_val)) * 253.f);
                map.Ptr<DT_S16>(h)[2 * w] = w_index;
                DT_S16  h_val   = (rand() % (int)max_val_f32) / 13;
                DT_S16  h_index = SaturateCast<DT_S16>((h_val + h_group - ((DT_S32)h_val)) * 13);
                h_index = pow(-1, rand()) * h_index;
                map.Ptr<DT_S16>(h)[2 * w + 1] = h_index;
            }
            map.Ptr<DT_S16>(h)[2 * src_sizes.m_width - 2] = pow(-1, rand()) * (rand() % (int)max_val_f32);
            map.Ptr<DT_S16>(h)[2 * src_sizes.m_width - 1] = pow(-1, rand()) * (rand() % (int)max_val_f32);
        }
    }
    else
    {
        return Status::ERROR;
    }

    return Status::OK;
}

class RemapTest : public TestBase<RemapParam::TupleTable, RemapParam::Tuple>
{
public:
    RemapTest(Context *ctx, RemapParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in RemapTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        RemapParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    run_param.mat_size.m_sizes.m_width < 800 &&
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        RemapParam run_param(GetParam((index)));
        if (run_param.map_elem_type == ElemType::S16 && run_param.interp_type != InterpType::NEAREST)
        {
            return 0;
        }

        // get next param set
        ElemType src_elem_type  = run_param.src_elem_type;
        ElemType dst_elem_type  = run_param.src_elem_type;
        MatSize  mat_size = run_param.mat_size;
        Scalar scale(1.0, 1.0);
        MatSize omat_sizes(run_param.mat_size, scale);

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1, 0, src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat map;
        CreateMap(m_factory, omat_sizes.m_sizes, run_param.map_elem_type, map);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(((ElemType::F16 == dst_elem_type) &&
                                         (TargetType::NONE == run_param.target.m_type))
                                        ? ElemType::F32 : dst_elem_type,
                                        omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);
        Scalar border_value = {10.2, 20.2, 13.2, 14.2};

        DT_S32 loop_count = stress_count ? stress_count : 10;
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        Status status_exec = Status::OK;

        result.param = InterpTypeToString(run_param.interp_type) + " | " +
                       BorderTypeToString(run_param.border_type) + " | map:" +
                       ElemTypesToString(map.GetElemType());
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        AURA_LOGD(m_ctx, AURA_TAG, "src is %s, map type is %s, Inter_type is %s, border_type is %s \n", result.input.c_str(), ElemTypesToString(run_param.map_elem_type).c_str(),
                  InterpTypeToString(run_param.interp_type).c_str(), BorderTypeToString(run_param.border_type).c_str());

        // run interface
        status_exec = Executor(loop_count, 2, time_val, IRemap, m_ctx, src, map, dst, run_param.interp_type,
                               run_param.border_type, border_value, run_param.target);

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
            status_exec = Executor(loop_count, 2, time_val, CvRemap, src, ref, map, run_param.interp_type, run_param.border_type, border_value);

            result.accu_benchmark = "OpenCV::remap";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark remap execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IRemap(m_ctx, src, map, ref, run_param.interp_type, run_param.border_type,
                                 border_value, TargetType::NONE);
            result.accu_benchmark = "Remap(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (Status::OK == MatCompare(m_ctx, dst, ref, cmp_result, 1.0, 0.5))
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
        m_factory.PutMats(src, dst, ref, map);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

#endif // AURA_OPS_WARP_UINT_TEST_HPP__