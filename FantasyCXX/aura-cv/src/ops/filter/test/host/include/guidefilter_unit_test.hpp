#ifndef AURA_OPS_GUIDEFILTER_UINT_TEST_HPP__
#define AURA_OPS_GUIDEFILTER_UINT_TEST_HPP__

#include "aura/ops/filter/guidefilter.hpp"
#include "aura/tools/unit_test.h"
#include "aura/ops/matrix.h"
#include "opencv_helper.hpp"

#if defined(AURA_ENABLE_OPENCL)
#include "aura/runtime/cl_mem.h"
#endif

using namespace aura;

AURA_TEST_PARAM(GuideFilterParam,
                ElemType,          elem_type,
                MatSize,           mat_size,
                MI_S32,            ksize,
                MI_F32,            eps,
                GuideFilterType,   type,
                BorderType,        border_type,
                OpTarget,          target);

template <typename Tp> struct GuideFilterOpenCVTraits;

template <> struct GuideFilterOpenCVTraits<MI_U8>
{
    static constexpr ElemType elem_type = ElemType::U16;
};
template <> struct GuideFilterOpenCVTraits<MI_S8>
{
    static constexpr ElemType elem_type = ElemType::S16;
};
template <> struct GuideFilterOpenCVTraits<MI_U16>
{
    static constexpr ElemType elem_type = ElemType::U32;
};
template <> struct GuideFilterOpenCVTraits<MI_S16>
{
    static constexpr ElemType elem_type = ElemType::S32;
};
template <> struct GuideFilterOpenCVTraits<MI_F16>
{
    static constexpr ElemType elem_type = ElemType::F32;
};
template <> struct GuideFilterOpenCVTraits<MI_F32>
{
    static constexpr ElemType elem_type = ElemType::F32;
};

template<typename Tp>
static AURA_VOID GuideFilterOpenCVHelper(cv::Mat &i, cv::Mat &p, cv::Mat &q, MI_S32 r, MI_F32 &eps, GuideFilterType &type, cv::BorderTypes border_type)
{
    if (GuideFilterType::FAST == type)
    {
        MI_S32 r_sub = MAX(r / 2, 1);
        MI_S32 wsize = 2 * r_sub + 1;
        auto elem_type = GuideFilterOpenCVTraits<Tp>::elem_type;

        //isub=resize(i)
        //psub=resize(p)
        cv::Mat i_sub(cv::Size(i.cols / 2, i.rows / 2), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat p_sub(cv::Size(i.cols / 2, i.rows / 2), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::resize(i, i_sub, i_sub.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(p, p_sub, p_sub.size(), 0, 0, cv::INTER_LINEAR);

        //meanI=fmean(I)
        cv::Mat mean_i(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_i_tmp;
        cv::boxFilter(i_sub, mean_i_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_i_tmp.convertTo(mean_i, mean_i.type());

        //meanP=fmean(P)
        cv::Mat mean_p(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_p_tmp;
        cv::boxFilter(p_sub, mean_p_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_p_tmp.convertTo(mean_p, mean_p.type());

        cv::Mat i_tmp(i_sub.size(), ElemTypeToOpencv(elem_type, i_sub.channels()));
        cv::Mat p_tmp(p_sub.size(), ElemTypeToOpencv(elem_type, p_sub.channels()));
        i_sub.convertTo(i_tmp, i_tmp.type());
        p_sub.convertTo(p_tmp, p_tmp.type());

        //corrI=fmean(I.*I)
        cv::Mat mean_ii(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_ii_tmp = i_tmp.mul(i_tmp);
        cv::boxFilter(mean_ii_tmp, mean_ii_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_ii_tmp.convertTo(mean_ii, mean_ii.type());

        //corrIp=fmean(I.*p)
        cv::Mat mean_ip(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_ip_tmp = i_tmp.mul(p_tmp);
        cv::boxFilter(mean_ip_tmp, mean_ip_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_ip_tmp.convertTo(mean_ip, mean_ip.type());

        //varI=corrI-meanI.*meanI
        cv::Mat var_i(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_mul_i = mean_i.mul(mean_i);
        cv::subtract(mean_ii, mean_mul_i, var_i);

        //covIp=corrIp-meanI.*meanp
        cv::Mat cov_ip(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat mean_mul_p = mean_i.mul(mean_p);
        cv::subtract(mean_ip, mean_mul_p, cov_ip);

        //a=covIp./(varI+eps)
        //b=meanp-a.*meanI
        cv::Mat a(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::Mat b(i_sub.size(), ElemTypeToOpencv(ElemType::F32, i_sub.channels()));
        cv::add(var_i, eps, var_i);
        cv::divide(cov_ip, var_i, a);
        cv::subtract(mean_p, a.mul(mean_i), b);

        //meana=fmean(a)
        //meanb=fmean(b)
        cv::Mat mean_a, mean_b;
        cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);

        //meanafull=resize(meana)
        //meanbfull=resize(meanb)
        cv::Mat mean_a_full(cv::Size(i.cols, i.rows), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_b_full(cv::Size(i.cols, i.rows), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::resize(mean_a, mean_a_full, mean_a_full.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(mean_b, mean_b_full, mean_b_full.size(), 0, 0, cv::INTER_LINEAR);

        //q=meana.*I+meanb
        cv::Mat i_tmp1(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        i.convertTo(i_tmp1, i_tmp1.type());
        cv::Mat tmp = mean_a_full.mul(i_tmp1) + mean_b_full;
        tmp.copyTo(q);
    }
    else // GuideFilterType::NORMAL
    {
        MI_S32 wsize = 2 * r + 1;
        auto elem_type = GuideFilterOpenCVTraits<Tp>::elem_type;

        //meanI=fmean(I)
        cv::Mat mean_i(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_i_tmp;
        cv::boxFilter(i, mean_i_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_i_tmp.convertTo(mean_i, mean_i.type());

        //meanP=fmean(P)
        cv::Mat mean_p(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_p_tmp;
        cv::boxFilter(p, mean_p_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_p_tmp.convertTo(mean_p, mean_p.type());

        cv::Mat i_tmp(i.size(), ElemTypeToOpencv(elem_type, i.channels()));
        cv::Mat p_tmp(p.size(), ElemTypeToOpencv(elem_type, p.channels()));
        i.convertTo(i_tmp, i_tmp.type());
        p.convertTo(p_tmp, p_tmp.type());

        //corrI=fmean(I.*I)
        cv::Mat mean_ii(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_ii_tmp = i_tmp.mul(i_tmp);
        cv::boxFilter(mean_ii_tmp, mean_ii_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_ii_tmp.convertTo(mean_ii, mean_ii.type());

        //corrIp=fmean(I.*p)
        cv::Mat mean_ip(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_ip_tmp = i_tmp.mul(p_tmp);
        cv::boxFilter(mean_ip_tmp, mean_ip_tmp, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        mean_ip_tmp.convertTo(mean_ip, mean_ip.type());

        //varI=corrI-meanI.*meanI
        cv::Mat var_i(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_mul_i = mean_i.mul(mean_i);
        cv::subtract(mean_ii, mean_mul_i, var_i);

        //covIp=corrIp-meanI.*meanp
        cv::Mat cov_ip(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat mean_mul_p = mean_i.mul(mean_p);
        cv::subtract(mean_ip, mean_mul_p, cov_ip);

        //a=covIp./(varI+eps)
        //b=meanp-a.*meanI
        cv::Mat a(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::Mat b(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        cv::add(var_i, eps, var_i);
        cv::divide(cov_ip, var_i, a);
        cv::subtract(mean_p, a.mul(mean_i), b);

        //meana=fmean(a)
        //meanb=fmean(b)
        cv::Mat mean_a, mean_b;
        cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);
        cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, border_type);

        //q=meana.*I+meanb
        cv::Mat i_tmp1(i.size(), ElemTypeToOpencv(ElemType::F32, i.channels()));
        i.convertTo(i_tmp1, i_tmp1.type());
        cv::Mat tmp = mean_a.mul(i_tmp1) + mean_b;
        tmp.copyTo(q);
    }

    return;
}

static Status CvGuideFilter(Context *ctx, Mat &src0, Mat &src1, Mat &dst, MI_S32 &ksize, MI_F32 &eps, GuideFilterType &type, BorderType &border_type)
{
    AURA_UNUSED(ctx);

    cv::Mat cv_src0 = MatToOpencv(src0);
    cv::Mat cv_src1 = MatToOpencv(src1);
    cv::Mat cv_ref  = MatToOpencv(dst);
    switch (src0.GetElemType())
    {
        case ElemType::U8:
        {
            GuideFilterOpenCVHelper<MI_U8>(cv_src0, cv_src1, cv_ref, ksize / 2, eps, type, (cv::BorderTypes)BorderTypeToOpencv(border_type));
            break;
        }

        case ElemType::U16:
        {
            GuideFilterOpenCVHelper<MI_U16>(cv_src0, cv_src1, cv_ref, ksize / 2, eps, type, (cv::BorderTypes)BorderTypeToOpencv(border_type));
            break;
        }

        case ElemType::S16:
        {
            GuideFilterOpenCVHelper<MI_S16>(cv_src0, cv_src1, cv_ref, ksize / 2, eps, type, (cv::BorderTypes)BorderTypeToOpencv(border_type));
            break;
        }

        case ElemType::F32:
        {
            GuideFilterOpenCVHelper<MI_F32>(cv_src0, cv_src1, cv_ref, ksize / 2, eps, type, (cv::BorderTypes)BorderTypeToOpencv(border_type));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

// using MatCmpResult = ArithmCmpResult<MatCmpPos>;
// add RelativeDiff for Fp16.

class GuideFilterTest : public TestBase<GuideFilterParam::TupleTable, GuideFilterParam::Tuple>
{
public:
    GuideFilterTest(Context *ctx, GuideFilterParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb", ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in GuideFilterTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        GuideFilterParam run_param(GetParam((index)));

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

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        GuideFilterParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize mat_size = run_param.mat_size;
        MI_F32 eps = run_param.eps;
        AURA_LOGD(m_ctx, AURA_TAG, "guidefilter param detail: %s\n", run_param.ToString().c_str());

        if ((run_param.ksize > 100) && ((run_param.mat_size.m_sizes.m_height >= 1024) ||
            (run_param.mat_size.m_sizes.m_width >= 2048)))
        {
            AURA_LOGD(m_ctx, AURA_TAG, "only test small size testcases when kernel size is greater than 100 \n");
            return 0;
        }

        // creat iauras
        std::vector<Mat> dsts, refs;
        Mat src0 = m_factory.GetRandomMat(-1000.0f, 1000.0f, elem_type, mat_size.m_sizes);
        Mat src1 = m_factory.GetRandomMat(-1000.0f, 1000.0f, elem_type, mat_size.m_sizes);
        Mat dst0 = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat dst1 = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref0 = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref1 = m_factory.GetEmptyMat(elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        dsts.push_back(dst0);
        dsts.push_back(dst1);
        refs.push_back(ref0);
        refs.push_back(ref1);

        MI_S32 loop_count = stress_count ? stress_count : 5;
        Scalar border_value = Scalar(0, 0, 0, 0);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        MI_F32 tolerance = (ElemType::F32 == elem_type) ? 0.5f : 3.0f;
        tolerance        = (TargetType::HVX == run_param.target.m_type) ? 7.f : tolerance;

        result.param  = GuideFilterTypeToString(run_param.type) + BorderTypeToString(run_param.border_type) + " | ksize: " + std::to_string(run_param.ksize);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        // run interface
        Status status_exec;
        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec  = Executor(loop_count, 2, time_val, IGuideFilter, m_ctx, src0, src1, dsts[0],
                                    run_param.ksize, eps, run_param.type, run_param.border_type,
                                    border_value, run_param.target);
            status_exec |= Executor(loop_count, 2, time_val, IGuideFilter, m_ctx, src0, src0, dsts[1],
                                    run_param.ksize, eps, run_param.type, run_param.border_type,
                                    border_value, run_param.target);
            if (Status::OK == status_exec)
            {
                result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
                result.perf_status = TestStatus::PASSED;
                AURA_LOGI(m_ctx, AURA_TAG, "time: %s\n", time_val.ToString().c_str());
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.perf_status = TestStatus::FAILED;
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }
        else
        {
            status_exec = Executor(loop_count, 2, time_val, IGuideFilter, m_ctx, src0, src1, dsts[0],
                                          run_param.ksize, eps, run_param.type, run_param.border_type,
                                          border_value, run_param.target);
            status_exec |= Executor(loop_count, 2, time_val, IGuideFilter, m_ctx, src0, src0, dsts[1],
                                    run_param.ksize, eps, run_param.type, run_param.border_type,
                                    border_value, run_param.target);
            if (Status::OK == status_exec)
            {
                result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
                result.perf_status = TestStatus::PASSED;
                AURA_LOGI(m_ctx, AURA_TAG, "time: %s\n", time_val.ToString().c_str());
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.perf_status = TestStatus::FAILED;
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            // only support F32/F16 accuracy test in None mode
            if (elem_type != ElemType::F32 && elem_type != ElemType::F16)
            {
                AURA_LOGD(m_ctx, AURA_TAG, "OpenCV only support F32/F16 type\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            Mat src_cv0, src_cv1, dst_cv;
            src_cv0 = m_factory.GetEmptyMat(ElemType::F32, mat_size.m_sizes, AURA_MEM_DEFAULT);
            src_cv1 = m_factory.GetEmptyMat(ElemType::F32, mat_size.m_sizes, AURA_MEM_DEFAULT);
            dst_cv  = m_factory.GetEmptyMat(ElemType::F32, mat_size.m_sizes, AURA_MEM_DEFAULT);

            status_exec |= IConvertTo(m_ctx, src0, src_cv0);
            status_exec |= IConvertTo(m_ctx, src1, src_cv1);
            status_exec |= Executor(loop_count, 2, time_val, CvGuideFilter, m_ctx, src_cv0, src_cv1, dst_cv, run_param.ksize, eps, run_param.type, run_param.border_type);
            status_exec |= IConvertTo(m_ctx, dst_cv, refs[0]);
            status_exec |= Executor(loop_count, 2, time_val, CvGuideFilter, m_ctx, src_cv0, src_cv0, dst_cv, run_param.ksize, eps, run_param.type, run_param.border_type);
            status_exec |= IConvertTo(m_ctx, dst_cv, refs[1]);

            result.perf_result["OpenCV"] = time_val;
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvGuideFilter execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.accu_benchmark = "OpenCV::GuideFilter";
        }
        else
        {
            status_exec  = IGuideFilter(m_ctx, src0, src1, refs[0], run_param.ksize, eps, run_param.type, run_param.border_type, border_value, OpTarget::None());
            status_exec |= IGuideFilter(m_ctx, src0, src0, refs[1], run_param.ksize, eps, run_param.type, run_param.border_type, border_value, OpTarget::None());

            result.accu_benchmark = "GuideFilter(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        if (MatCompare(m_ctx, dsts, refs, cmp_result, tolerance) == Status::OK)
        {
            if (run_param.type == GuideFilterType::NORMAL)
            {
                result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
                result.accu_result = cmp_result.ToString();
            }
            else
            {
                // since there is diff in resize, the error is amplified after square calculation
                MI_S32 index = (ElemType::F32 == elem_type) ? 1 : tolerance;
                result.accu_status =  (static_cast<MI_F32>(cmp_result.hist[index].second) / cmp_result.total) > 0.93 ? TestStatus::PASSED : TestStatus::FAILED;
                result.accu_result = cmp_result.ToString();
            }
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_GUIDEFILTER_UINT_TEST_HPP__
