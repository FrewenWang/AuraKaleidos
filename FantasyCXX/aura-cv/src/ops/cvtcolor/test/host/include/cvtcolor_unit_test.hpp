#ifndef AURA_OPS_CVTCOLOR_UINT_TEST_HPP__
#define AURA_OPS_CVTCOLOR_UINT_TEST_HPP__

#include "aura/ops/cvtcolor/cvtcolor.hpp"
#include "cvtcolor_impl.hpp"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/cl_mem.h"
#endif

using namespace aura;

using VecMatSizePair = std::pair<std::vector<MatSize>, std::vector<MatSize>>;

AURA_INLINE std::ostream& operator<<(std::ostream &os, std::vector<MatSize> mat_sizes)
{
    MI_S32 n = mat_sizes.size();
    if (n > 0)
    {
        auto iter = mat_sizes.begin();

        for (MI_S32 i = 0; i < (n - 1); i++)
        {
            os << *iter << " ";
            iter++;
        }

        os << *iter;
    }
    return os;
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, VecMatSizePair size_pair)
{
    os << "src mat size : " << size_pair.first << " dst mat size : " << size_pair.second << std::endl;
    return os;
}

AURA_TEST_PARAM(CvtColorParam,
                ElemType,       elem_type,
                VecMatSizePair, mat_sizes_pair,
                CvtColorType,   cvt_type,
                OpTarget,       target);

#if !defined(AURA_BUILD_XPLORER)
static MI_S32 CvtColorTypeToOpencv(CvtColorType cvt_type)
{
    switch (cvt_type)
    {
        // RGB <-> BGRA
        case CvtColorType::BGR2BGRA:
        {
            return cv::COLOR_BGR2BGRA;
        }

        case CvtColorType::BGRA2BGR:
        {
            return cv::COLOR_BGRA2BGR;
        }

        case CvtColorType::BGR2RGB:
        {
            return cv::COLOR_BGR2RGB;
        }

        case CvtColorType::BGR2GRAY:
        {
            return cv::COLOR_BGR2GRAY;
        }

        case CvtColorType::RGB2GRAY:
        {
            return cv::COLOR_RGB2GRAY;
        }

        case CvtColorType::GRAY2BGR:
        {
            return cv::COLOR_GRAY2BGR;
        }

        case CvtColorType::GRAY2BGRA:
        {
            return cv::COLOR_GRAY2BGRA;
        }

        case CvtColorType::BGRA2GRAY:
        {
            return cv::COLOR_BGRA2GRAY;
        }

        case CvtColorType::RGBA2GRAY:
        {
            return cv::COLOR_RGBA2GRAY;
        }

        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        {
            return -1;
        }
        case CvtColorType::RGB2YUV_NV21:
        {
            return -1;
        }
        case CvtColorType::RGB2YUV_YU12:
        {
            return cv::COLOR_RGB2YUV_I420;
        }
        case CvtColorType::RGB2YUV_YV12:
        {
            return cv::COLOR_RGB2YUV_YV12;
        }
        case CvtColorType::RGB2YUV_Y444:
        {
            return -1;
        }

        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        {
            return cv::COLOR_YUV2RGB_NV12;
        }
        case CvtColorType::YUV2RGB_NV21:
        {
            return cv::COLOR_YUV2RGB_NV21;
        }
        case CvtColorType::YUV2RGB_YU12:
        {
            return cv::COLOR_YUV2RGB_I420;
        }
        case CvtColorType::YUV2RGB_Y422:
        {
            return cv::COLOR_YUV2RGB_Y422;
        }
        case CvtColorType::YUV2RGB_YUYV:
        {
            return cv::COLOR_YUV2RGB_YUYV;
        }
        case CvtColorType::YUV2RGB_YVYU:
        {
            return cv::COLOR_YUV2RGB_YVYU;
        }
        case CvtColorType::YUV2RGB_YV12:
        {
            return cv::COLOR_YUV2RGB_YV12;
        }
        case CvtColorType::YUV2RGB_Y444:
        {
            return -1;
        }

        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        {
            return cv::COLOR_BayerBG2BGR;
        }
        case CvtColorType::BAYERGB2BGR:
        {
            return cv::COLOR_BayerGB2BGR;
        }
        case CvtColorType::BAYERRG2BGR:
        {
            return cv::COLOR_BayerRG2BGR;
        }
        case CvtColorType::BAYERGR2BGR:
        {
            return cv::COLOR_BayerGR2BGR;
        }

        default:
        {
            return -1;
        }
    }
}

static cv::Mat AuraMatToOpencvMat(std::vector<Mat> &mats)
{
    Mat    mat    = mats[0];
    MI_S32 width  = mat.GetSizes().m_width;
    MI_S32 height = 0;

    switch (mats.size())
    {
        case 1:
        {
            height = mat.GetSizes().m_height;
            break;
        }
        case 2:
        {
            height = mats[0].GetSizes().m_height + mats[1].GetSizes().m_height;
            break;
        }
        case 3:
        {
            height = mats[0].GetSizes().m_height + (mats[1].GetSizes().m_height + mats[2].GetSizes().m_height) * mats[1].GetSizes().m_width / width;
            break;
        }
        default:
        {
            return cv::Mat();
        }
    }

    MI_S32 cv_type = ElemTypeToOpencv(mat.GetElemType(), mat.GetSizes().m_channel);
    cv::Mat result(height, width, cv_type);

    MI_S32 offset = 0;
    MI_U8 *data   = (MI_U8 *)result.data;
    for (MI_S32 i = 0; i < (MI_S32)mats.size(); i++)
    {
        MI_S32 row_bytes = mats[i].GetSizes().m_width * mats[i].GetSizes().m_channel * ElemTypeSize(mats[i].GetElemType());
        for (MI_S32 h = 0; h < mats[i].GetSizes().m_height; h++)
        {
            memcpy(data + offset, mats[i].Ptr<MI_U8>(h), row_bytes);
            offset += row_bytes;
        }
    }

    return result;
}
#endif

static std::string VecMatSizeToString(std::vector<MatSize> mat_sizes)
{
    std::ostringstream ss;
    ss << mat_sizes;
    return ss.str();
}

/**
 * @brief the formula of BT601 RGB -> YUV
 *  Y =  0.299  * r + 0.587  * g + 0.114  * b
 * Cb = -0.1687 * r - 0.3313 * g + 0.5    * b + 128
 * Cr =  0.5    * r - 0.4187 * g - 0.0813 * b + 128
 *
 * @brief the formula of P010 RGB -> YUV
 *  Y = ( 0.299  * r + 0.587  * g + 0.114  * b)       * (1 << 6)
 * Cb = (-0.1687 * r - 0.3313 * g + 0.5    * b + 512) * (1 << 6)
 * Cr = ( 0.5    * r - 0.4187 * g - 0.0813 * b + 512) * (1 << 6)
 */
#define CVTCOLOR_601_RGB2Y(Tp, R, G, B, SHIFT)      SaturateCast<Tp>(( 0.299f  * R + 0.587f  * G + 0.114f  * B)      * (1 << SHIFT))
#define CVTCOLOR_601_RGB2U(Tp, R, G, B, SHIFT, UC)  SaturateCast<Tp>((-0.1687f * R - 0.3313f * G + 0.5f    * B + UC) * (1 << SHIFT))
#define CVTCOLOR_601_RGB2V(Tp, R, G, B, SHIFT, UC)  SaturateCast<Tp>(( 0.5f    * R - 0.4187f * G - 0.0813f * B + UC) * (1 << SHIFT))

template <typename Tp, MI_S32 SHIFT>
static Status Cvt601Rgb2Yuv420Benchmark(std::vector<Mat> &src, std::vector<Mat> &dst, CvtColorType type)
{
    MI_S32 height = src[0].GetSizes().m_height;
    MI_S32 width  = src[0].GetSizes().m_width;
    MI_S32 u_idx  = SwapUv(type);
    MI_S32 v_idx  = 1 - u_idx;

    for (MI_S32 h = 0; h < height; h++)
    {
        for (MI_S32 w = 0; w < width; w++)
        {
            Tp R = src[0].At<Tp>(h, w, 0);
            Tp G = src[0].At<Tp>(h, w, 1);
            Tp B = src[0].At<Tp>(h, w, 2);

            dst[0].At<Tp>(h, w) = CVTCOLOR_601_RGB2Y(Tp, R, G, B, SHIFT);
            if (0 == (h & 1) && 0 == (w & 1))
            {
                switch (type)
                {
                    case CvtColorType::RGB2YUV_NV12_601:
                    case CvtColorType::RGB2YUV_NV21_601:
                    {
                        dst[1].At<Tp>(h / 2, w / 2, u_idx) = CVTCOLOR_601_RGB2U(Tp, R, G, B, SHIFT, 128);
                        dst[1].At<Tp>(h / 2, w / 2, v_idx) = CVTCOLOR_601_RGB2V(Tp, R, G, B, SHIFT, 128);
                        break;
                    }
                    case CvtColorType::RGB2YUV_YU12_601:
                    case CvtColorType::RGB2YUV_YV12_601:
                    {
                        dst[u_idx + 1].At<Tp>(h / 2, w / 2) = CVTCOLOR_601_RGB2U(Tp, R, G, B, SHIFT, 128);
                        dst[v_idx + 1].At<Tp>(h / 2, w / 2) = CVTCOLOR_601_RGB2V(Tp, R, G, B, SHIFT, 128);
                        break;
                    }
                    case CvtColorType::RGB2YUV_NV12_P010:
                    case CvtColorType::RGB2YUV_NV21_P010:
                    {
                        dst[1].At<Tp>(h / 2, w / 2, u_idx) = CVTCOLOR_601_RGB2U(Tp, R, G, B, SHIFT, 512);
                        dst[1].At<Tp>(h / 2, w / 2, v_idx) = CVTCOLOR_601_RGB2V(Tp, R, G, B, SHIFT, 512);
                        break;
                    }
                    default:
                    {
                        return Status::ERROR;
                    }
                }
            }

        }
    }

    return Status::OK;
}

template <typename Tp, MI_S32 SHIFT>
static Status Cvt601Rgb2Yuv444Benchmark(std::vector<Mat> &src, std::vector<Mat> &dst, CvtColorType type)
{
    AURA_UNUSED(type);
    MI_S32 height = src[0].GetSizes().m_height;
    MI_S32 width  = src[0].GetSizes().m_width;

    for (MI_S32 h = 0; h < height; h++)
    {
        for (MI_S32 w = 0; w < width; w++)
        {
            Tp R = src[0].At<Tp>(h, w, 0);
            Tp G = src[0].At<Tp>(h, w, 1);
            Tp B = src[0].At<Tp>(h, w, 2);

            dst[0].At<Tp>(h, w) = CVTCOLOR_601_RGB2Y(Tp, R, G, B, SHIFT);
            dst[1].At<Tp>(h, w) = CVTCOLOR_601_RGB2U(Tp, R, G, B, SHIFT, 128);
            dst[2].At<Tp>(h, w) = CVTCOLOR_601_RGB2V(Tp, R, G, B, SHIFT, 128);
        }
    }

    return Status::OK;
}

static Status Cvt601Rgb2YuvBenchmark(Context *ctx, std::vector<Mat> &src, std::vector<Mat> &dst, CvtColorType type)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = Cvt601Rgb2Yuv420Benchmark<MI_U8, 0>(src, dst, type);
            break;
        }
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = Cvt601Rgb2Yuv444Benchmark<MI_U8, 0>(src, dst, type);
            break;
        }
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = Cvt601Rgb2Yuv420Benchmark<MI_U16, 6>(src, dst, type);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    return ret;
}

static Status CheckTypeIs601Rgb2Yuv(CvtColorType type)
{
    switch (type)
    {
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            return Status::OK;
        }

        default:
        {
            return Status::ERROR;
        }
    }
}

static Status CvCvtColor(Context *ctx, std::vector<Mat> &src, std::vector<Mat> &dst, CvtColorType type)
{
    if (CheckTypeIs601Rgb2Yuv(type) == Status::OK)
    {
        return Cvt601Rgb2YuvBenchmark(ctx, src, dst, type);
    }

#if !defined(AURA_BUILD_XPLORER)
    MI_S32 cv_method = CvtColorTypeToOpencv(type);
    if (-1 == cv_method)
    {
        AURA_LOGD(ctx, AURA_TAG, "Opencv is not supprot cvtcolor type\n");
        return Status::ERROR;
    }

    cv::Mat cv_src = AuraMatToOpencvMat(src);
    cv::Mat cv_dst = AuraMatToOpencvMat(dst);
    cv::cvtColor(cv_src, cv_dst, cv_method);

    MI_S32 offset = 0;
    MI_U8 *data   = (MI_U8 *)cv_dst.data;
    for (MI_S32 i = 0; i < (MI_S32)dst.size(); i++)
    {
        MI_S32 row_bytes = dst[i].GetSizes().m_width * dst[i].GetSizes().m_channel * ElemTypeSize(dst[i].GetElemType());
        for (MI_S32 h = 0; h < dst[i].GetSizes().m_height; h++)
        {
            memcpy(dst[i].Ptr<MI_U8>(h), data + offset, row_bytes);
            offset += row_bytes;
        }
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class CvtColorTest : public TestBase<CvtColorParam::TupleTable, CvtColorParam::Tuple>
{
public:
    CvtColorTest(Context *ctx, CvtColorParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb", ElemType::U8, {512, 512, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb", ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgba", ElemType::U8, {512, 512, 4});
        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in CvtColorTest\n");
        }
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        CvtColorParam run_param(GetParam(index));

        std::vector<MatSize> imat_sizes = run_param.mat_sizes_pair.first;
        std::vector<MatSize> omat_sizes = run_param.mat_sizes_pair.second;

        AURA_LOGD(m_ctx, AURA_TAG, "cvtcolor param detail: elem_type(%s), mat_size(%dx%d), cvtcolor_type(%s)\n",
                  ElemTypesToString(run_param.elem_type).c_str(), imat_sizes[0].m_sizes.m_height, imat_sizes[0].m_sizes.m_width,
                  CvtColorTypeToString(run_param.cvt_type).c_str());

        MI_S32 src_len = imat_sizes.size();
        MI_S32 dst_len = omat_sizes.size();

        // creat iauras
        std::vector<Mat> src, dst, ref;
        src.reserve(src_len);
        dst.reserve(dst_len);
        ref.reserve(dst_len);

        for (MI_S32 i = 0; i < src_len; i++)
        {
            Mat src_tmp = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, imat_sizes[i].m_sizes, AURA_MEM_DEFAULT, imat_sizes[i].m_strides);
            src.push_back(src_tmp);
        }
        for (MI_S32 i = 0; i < dst_len; i++)
        {
            Mat dst_tmp = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes[i].m_sizes, AURA_MEM_DEFAULT, omat_sizes[i].m_strides);
            Mat ref_tmp = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes[i].m_sizes, AURA_MEM_DEFAULT, omat_sizes[i].m_strides);
            dst.push_back(dst_tmp);
            ref.push_back(ref_tmp);
        }

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = CvtColorTypeToString(run_param.cvt_type);
        result.input  = VecMatSizeToString(imat_sizes) + " " + ElemTypesToString(run_param.elem_type);
        result.output = VecMatSizeToString(omat_sizes) + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, ICvtColor, m_ctx, src, dst, run_param.cvt_type, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec = Executor(10, 2, time_val, CvCvtColor, m_ctx, src, ref, run_param.cvt_type);
            result.accu_benchmark = "OpenCV::CvtColor";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCvtColor execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = ICvtColor(m_ctx, src, ref, run_param.cvt_type, TargetType::NONE);
            result.accu_benchmark = "CvtColor(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        for (MI_S32 i = 0; i < src_len; i++)
        {
            m_factory.PutMats(src[i]);
        }
        for (MI_S32 i = 0; i < dst_len; i++)
        {
            m_factory.PutMats(dst[i], ref[i]);
        }

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_CVTCOLOR_UINT_TEST_HPP__