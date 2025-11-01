/** @brief      : houghline uint test head for aura
 *  @file       : houghline_unit_test.hpp
 *  @author     : wuzhiwei3@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 27, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HOUGHLINE_UINT_TEST_HPP__
#define AURA_OPS_HOUGHLINE_UINT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/ops/feature2d.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct HoughLinesTestParam
{
    HoughLinesTestParam()
    {}

    HoughLinesTestParam(LinesType line_type, MI_F64 rho, MI_F64 theta, MI_S32 threshold, MI_F64 srn,
                        MI_F64 stn, MI_F64 min_theta, MI_F64 max_theta)
                        : line_type(line_type), rho(rho), theta(theta), threshold(threshold), srn(srn), stn(stn),
                        min_theta(min_theta), max_theta(max_theta)
    {}

    friend std::ostream& operator<<(std::ostream &os, HoughLinesTestParam houghlines_test_param)
    {
        os << "lines_type:" << houghlines_test_param.line_type << " | rho:" << houghlines_test_param.rho << " | theta:" << houghlines_test_param.theta
           << " | threshold:" << houghlines_test_param.threshold << " | srn:" << houghlines_test_param.srn << " | stn:" << houghlines_test_param.stn
           << " | min_theta:" << houghlines_test_param.min_theta << " | max_theta:" << houghlines_test_param.max_theta << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    LinesType line_type;
    MI_F64 rho;
    MI_F64 theta;
    MI_S32 threshold;
    MI_F64 srn;
    MI_F64 stn;
    MI_F64 min_theta;
    MI_F64 max_theta;
};

AURA_TEST_PARAM(HoughLinesParam,
                ElemType,      elem_type,
                MatSize,       mat_sizes,
                HoughLinesTestParam, param,
                OpTarget,      target);

static Status CvHoughLines(Context *ctx, Mat &mat, std::vector<Scalar> &lines, HoughLinesTestParam param)
{
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    if (mat.GetElemType() != ElemType::U8)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV CvHoughLines only support type u8\n");
        return Status::ERROR;
    }

    lines.clear();

    MI_S32 mat_cv_type = ElemTypeToOpencv(mat.GetElemType(), mat.GetSizes().m_channel);
    MI_S32 mat_cn = mat.GetSizes().m_channel;
    MI_S32 cv_type = 0;

    if (CV_8UC(mat_cn) != mat_cv_type)
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat mat_src = MatToOpencv(mat);

        if (LinesType::VEC2F == param.line_type)
        {
            std::vector<cv::Vec2f> lines_cv;
            lines.reserve(1024);

            cv::HoughLines(mat_src, lines_cv, param.rho, param.theta, param.threshold, param.srn, param.stn, param.min_theta, param.max_theta);
            MI_S32 size = lines_cv.size();

            if (size > 0)
            {
                if (((param.srn != 0) || (param.stn != 0)) && (0.f ==lines_cv[size - 1][0]) && (0.f ==lines_cv[size - 1][1]))
                {
                    size = size - 1;
                }
            }

            for (MI_S32 i = 0; i < size; i++)
            {
                lines.emplace_back(lines_cv[i][0], lines_cv[i][1], 0.f, 0.f);
            }
        }
        else if (LinesType::VEC3F == param.line_type)
        {
            std::vector<cv::Vec3f> lines_cv;
            lines_cv.reserve(1024);

            cv::HoughLines(mat_src, lines_cv, param.rho, param.theta, param.threshold, param.srn, param.stn, param.min_theta, param.max_theta);
            MI_S32 size = lines_cv.size();

            if (size > 0)
            {
                if (((param.srn != 0) || (param.stn != 0)) && (0.f ==lines_cv[size - 1][0]) && (0.f ==lines_cv[size - 1][1]) && (0.f ==lines_cv[size - 1][2]))
                {
                    size = size - 1;
                }
            }

            for (MI_S32 i = 0; i < size; i++)
            {
                lines.emplace_back(lines_cv[i][0], lines_cv[i][1], lines_cv[i][2], 0.f);
            }
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV HoughLines not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(mat);
    AURA_UNUSED(lines);
    AURA_UNUSED(param);
#endif

    return ret;
}

static AURA_VOID HoughLinesConvert(const std::vector<Scalar> &lines, std::vector<Scalar> &cvt_lines)
{
    for (auto it = lines.begin(); it != lines.end(); ++it)
    {
        cvt_lines.emplace_back(it->m_val[0] * Cos(it->m_val[1]), it->m_val[0] * Sin(it->m_val[1]), it->m_val[2], 0.f);
    }
}

class HoughLinesTest : public TestBase<HoughLinesParam::TupleTable, HoughLinesParam::Tuple>
{
public:
    HoughLinesTest(Context *ctx, HoughLinesParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";
        status |= m_factory.LoadBaseMat(data_file + "lines_487x487.gray", ElemType::U8, {487, 487});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        HoughLinesParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (run_param.mat_sizes.m_sizes.m_width  < 800 &&
                    run_param.mat_sizes.m_sizes.m_height < 600)
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
        HoughLinesParam run_param(GetParam((index)));

        // creat iauras
        Mat mat = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat mat_canny = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        std::vector<Scalar> lines_dst;
        std::vector<Scalar> lines_ref;
        std::vector<Scalar> cvt_lines_dst;
        std::vector<Scalar> cvt_lines_ref;

        lines_dst.reserve(1024);
        lines_ref.reserve(1024);
        cvt_lines_dst.reserve(1024);
        cvt_lines_ref.reserve(1024);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        UnorderedCmpResult<Scalar> cmp_result;
        TestResult result;

        // run interface
        //Canny(m_ctx, mat, mat_canny, 80, 150, 3, false, run_param.target);

        Status ret = Status::ERROR;
        ret = Executor(loop_count, 2, time_val, IHoughLines, m_ctx, mat_canny, lines_dst, run_param.param.line_type,
                             run_param.param.rho, run_param.param.theta, run_param.param.threshold, run_param.param.srn,
                             run_param.param.stn, run_param.param.min_theta, run_param.param.max_theta, run_param.target);

        result.param  = run_param.param.ToString();
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = "lines num:" + std::to_string(lines_dst.size());

        if (Status::OK == ret)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail\n");
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            ret = Executor(10, 2, time_val, CvHoughLines, m_ctx, mat_canny, lines_ref, run_param.param);
            result.accu_benchmark = "OpenCV::HoughLines";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvHoughLines execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IHoughLines(m_ctx, mat, lines_ref, run_param.param.line_type, run_param.param.rho, run_param.param.theta, run_param.param.threshold,
                                   run_param.param.srn, run_param.param.stn, run_param.param.min_theta, run_param.param.max_theta, TargetType::NONE);
            result.accu_benchmark = "HoughLines(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        HoughLinesConvert(lines_dst, cvt_lines_dst);
        HoughLinesConvert(lines_ref, cvt_lines_ref);

        UnorderedCompare<Scalar>(m_ctx, cvt_lines_dst, cvt_lines_ref, cmp_result,
                               Tolerate<Scalar>(0.95f, Scalar(0.5f, 0.5f, 2.f, 1e-5f)));
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(mat, mat_canny);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

struct HoughLinesPTestParam
{
    HoughLinesPTestParam()
    {}

    HoughLinesPTestParam(MI_F64 rho, MI_F64 theta, MI_S32 threshold, MI_F64 min_line_length, MI_F64 max_gap)
                         : rho(rho), theta(theta), threshold(threshold), min_line_length(min_line_length), max_gap(max_gap)
    {}

    friend std::ostream& operator<<(std::ostream &os, HoughLinesPTestParam houghlinesP_test_param)
    {
        os << "rho:" << houghlinesP_test_param.rho << " | theta:" << houghlinesP_test_param.theta
           << " | threshold:" << houghlinesP_test_param.threshold << " | min_line_length:" << houghlinesP_test_param.min_line_length
           << " | max_gap:" << houghlinesP_test_param.max_gap << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MI_F64 rho;
    MI_F64 theta;
    MI_S32 threshold;
    MI_F64 min_line_length;
    MI_F64 max_gap;
};

AURA_TEST_PARAM(HoughLinesPParam,
                ElemType,       elem_type,
                MatSize,        mat_sizes,
                HoughLinesPTestParam, param,
                OpTarget,       target);

static Status CvHoughLinesP(Context *ctx, Mat &mat, std::vector<Scalari> &lines, HoughLinesPTestParam param)
{
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    if (mat.GetElemType() != ElemType::U8)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV CvHoughLinesP only support type u8\n");
        return Status::ERROR;
    }

    lines.clear();

    MI_S32 mat_cv_type = ElemTypeToOpencv(mat.GetElemType(), mat.GetSizes().m_channel);
    MI_S32 mat_cn = mat.GetSizes().m_channel;
    MI_S32 cv_type = 0;

    if (CV_8UC(mat_cn) != mat_cv_type)
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat mat_src = MatToOpencv(mat);
        std::vector<cv::Vec4i> lines_cv;
        lines_cv.reserve(1024);

        cv::HoughLinesP(mat_src, lines_cv, param.rho, param.theta, param.threshold, param.min_line_length, param.max_gap);

        for (MI_U64 i = 0; i < lines_cv.size(); i++)
        {
            lines.emplace_back(lines_cv[i][0], lines_cv[i][1], lines_cv[i][2], lines_cv[i][3]);
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV HoughLinesP not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(mat);
    AURA_UNUSED(lines);
    AURA_UNUSED(param);
#endif

    return ret;
}

class HoughLinesPTest : public TestBase<HoughLinesPParam::TupleTable, HoughLinesPParam::Tuple>
{
public:
    HoughLinesPTest(Context *ctx, HoughLinesPParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";
        status |= m_factory.LoadBaseMat(data_file + "lines_487x487.gray", ElemType::U8, {487, 487});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        HoughLinesPParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (run_param.mat_sizes.m_sizes.m_width  < 800 &&
                    run_param.mat_sizes.m_sizes.m_height < 600)
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
        HoughLinesPParam run_param(GetParam((index)));

        // creat iauras
        Mat mat = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat mat_canny = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        std::vector<Scalari> lines_dst;
        std::vector<Scalari> lines_ref;
        lines_dst.reserve(1024);
        lines_ref.reserve(1024);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        UnorderedCmpResult<Scalari> cmp_result;
        TestResult result;

        // run interface
        ICanny(m_ctx, mat, mat_canny, 80, 150, 3, false, run_param.target);

        Status ret = Status::ERROR;
        ret = Executor(loop_count, 2, time_val, IHoughLinesP, m_ctx, mat_canny, lines_dst, run_param.param.rho, run_param.param.theta,
                             run_param.param.threshold, run_param.param.min_line_length, run_param.param.max_gap, run_param.target);

        result.param  = run_param.param.ToString();
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = "lines num:" + std::to_string(lines_dst.size());

        if (Status::OK == ret)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail\n");
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            ret = Executor(10, 2, time_val, CvHoughLinesP, m_ctx, mat_canny, lines_ref, run_param.param);
            result.accu_benchmark = "OpenCV::HoughLinesP";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvHoughLinesP execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IHoughLinesP(m_ctx, mat, lines_ref, run_param.param.rho, run_param.param.theta, run_param.param.threshold,
                                    run_param.param.min_line_length, run_param.param.max_gap, TargetType::NONE);
            result.accu_benchmark = "HoughLinesP(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        UnorderedCompare<Scalari>(m_ctx, lines_dst, lines_ref, cmp_result);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(mat, mat_canny);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

#endif // AURA_OPS_HOUGHLINE_UINT_TEST_HPP__