#ifndef AURA_OPS_MATRIX_GRID_DFT_IDFT_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_GRID_DFT_IDFT_UNIT_TEST_HPP__

#include "aura/ops/matrix/grid_dft.hpp"
#include "aura/ops/matrix/convert_to.hpp"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

#if defined(AURA_ENABLE_OPENCL)
#include "aura/runtime/cl_mem.h"
#endif

using namespace aura;

AURA_INLINE Status OpenCVGridDft(Mat &src, Mat &dst, DT_S32 grid_len)
{
    if (ElemType::F32 != src.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);

    DT_S32 height = cv_src.rows;
    DT_S32 width  = cv_dst.cols;
    DT_S32 grid_rows = height / grid_len;
    DT_S32 grid_cols = width / grid_len;
    for (DT_S32 i = 0; i < grid_rows; i++)
    {
        for (DT_S32 j = 0; j < grid_cols; j++)
        {
            DT_S32 x = j * grid_len;
            DT_S32 y = i * grid_len;
            cv::Rect roi(x, y, grid_len, grid_len);
            cv::Mat src_grid = cv_src(roi);
            cv::Mat dst_grid = cv_dst(roi);
            cv::dft(src_grid, dst_grid, cv::DFT_COMPLEX_OUTPUT);
        }
    }
#else
    AURA_UNUSED(dst);
    AURA_UNUSED(grid_len);
#endif

    return Status::OK;
}

#if !defined(MATSIZEPAIR)

#define MATSIZEPAIR
using MatSizePair = std::pair<MatSize, MatSize>;
static std::ostream& operator << (std::ostream &os, MatSizePair size_pair)
{
    os << "src mat size : " << size_pair.first << " dst mat size : " << size_pair.second << std::endl;
    return os;
}
#endif // MATSIZEPAIR

AURA_TEST_PARAM(GridDftParam,
                ElemType,     elem_type,
                MatSizePair,  mat_size,
                DT_S32,       grid_len,
                OpTarget,     target,
                ArrayType,    array_type);

class MatrixGridDftTest : public TestBase<GridDftParam::TupleTable, GridDftParam::Tuple>
{
public:
    MatrixGridDftTest(Context *ctx, GridDftParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        GridDftParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src    = m_factory.GetRandomMat(-1000, 1000, run_param.elem_type, run_param.mat_size.first.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.first.m_strides);
        Mat cv_src = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.first.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.first.m_strides); // opencv need float input
        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);

        TestTime time_val;
        MatCmpResult cmp_result;
        MatCmpResult cmp_result_temp0;
        MatCmpResult cmp_result_temp1;
        TestResult result;
        result.param  = "grid_len: " + std::to_string(run_param.grid_len);
        result.input  = run_param.mat_size.first.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.second.ToString() + " " + "F32";

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 5;
        Status status_exec = Executor(loop_count, 2, time_val, IGridDft, m_ctx, src, dst, run_param.grid_len, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
            std::cout << "execute time: " << time_val.ToString() << std::endl;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::GridDft";
            status_exec = IConvertTo(m_ctx, src, cv_src);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "ConvertTo for OpenCVDft execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }

            status_exec = Executor(5, 2, time_val, OpenCVGridDft, cv_src, ref, run_param.grid_len);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVGridDft execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            std::cout << "OpenCV time: " << time_val.ToString() << std::endl;
        }
        else
        {
            result.accu_benchmark = "GridDft(target::none)";
            status_exec = IGridDft(m_ctx, src, ref, run_param.grid_len, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1, 1);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context   *m_ctx;
    MatFactory m_factory;
};

AURA_INLINE Status OpenCVGridIDft(Mat &src, Mat &dst, DT_S32 grid_len)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    DT_S32 channel = dst.GetSizes().m_channel;

    DT_S32 height = cv_src.rows;
    DT_S32 width  = cv_dst.cols;
    DT_S32 grid_rows = height / grid_len;
    DT_S32 grid_cols = width / grid_len;

    for (DT_S32 i = 0; i < grid_rows; i++)
    {
        for (DT_S32 j = 0; j < grid_cols; j++)
        {
            DT_S32 x = j * grid_len;
            DT_S32 y = i * grid_len;
            cv::Rect roi(x, y, grid_len, grid_len);
            cv::Mat src_grid = cv_src(roi);
            cv::Mat dst_grid = cv_dst(roi);
            if (1 == channel)
            {
                cv::Mat idft_result;
                cv::idft(src_grid, idft_result, cv::DFT_SCALE);
                cv::Mat vec_idft_result[2];
                cv::split(idft_result, vec_idft_result);
                // only save real part
                vec_idft_result[0].convertTo(dst_grid, ElemTypeToOpencv(dst.GetElemType(), 1));
                if (ElemType::U32 == dst.GetElemType())
                {
                    cv::Mat mask = (vec_idft_result[0] > 0) / 255;
                    mask.convertTo(mask, dst_grid.type());
                    dst_grid = dst_grid.mul(mask);
                }
            }
            else
            {
                cv::idft(src_grid, dst_grid, cv::DFT_SCALE);
            }
        }
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(grid_len);
#endif

    return Status::OK;
}

AURA_TEST_PARAM(GridIDftParam,
                MatSize,   mat_size,
                DT_S32,    grid_len,
                OpTarget,  target,
                ArrayType, array_type);

class MatrixGridIDftTest : public TestBase<GridIDftParam::TupleTable, GridIDftParam::Tuple>
{
public:
    MatrixGridIDftTest(Context *ctx, GridIDftParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        GridIDftParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create src mat
        Mat src = m_factory.GetRandomMat(-1000, 1000, ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = "grid_len: " + std::to_string(run_param.grid_len);
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(ElemType::F32);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(ElemType::F32);

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 5;
        Status status_exec = Executor(loop_count, 2, time_val, IGridIDft, m_ctx, src, dst, run_param.grid_len, DT_TRUE, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
            std::cout << "execute time: " << time_val.ToString() << std::endl;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::GridIDft";
            status_exec = Executor(5, 2, time_val, OpenCVGridIDft, src, ref, run_param.grid_len);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVGridIDft execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            std::cout << "OpenCV time: " << time_val.ToString() << std::endl;
        }
        else
        {
            result.accu_benchmark = "GridIDft(target::none)";
            status_exec = IGridIDft(m_ctx, src, ref, run_param.grid_len, DT_TRUE, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1.0);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();
EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutAllMats();

        return 0;
    }
private:
    Context   *m_ctx;
    MatFactory m_factory;
};

AURA_TEST_PARAM(GridIDftRealParam,
                ElemType,    elem_type,
                MatSizePair, mat_size,
                DT_S32,      grid_len,
                OpTarget,    target,
                ArrayType,   array_type);

class MatrixGridIDftRealTest : public TestBase<GridIDftRealParam::TupleTable, GridIDftRealParam::Tuple>
{
public:
    MatrixGridIDftRealTest(Context *ctx, GridIDftRealParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        GridIDftRealParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        ElemType dst_type = run_param.elem_type;
        // Create src mat
        Mat src = m_factory.GetRandomMat(-1000, 1000, ElemType::F32, run_param.mat_size.first.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.first.m_strides);
        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(dst_type, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);
        Mat ref = m_factory.GetEmptyMat(dst_type, run_param.mat_size.second.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.second.m_strides);

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = "grid_len: " + std::to_string(run_param.grid_len);
        result.input  = run_param.mat_size.first.m_sizes.ToString() + " " + ElemTypesToString(ElemType::F32);
        result.output = run_param.mat_size.second.m_sizes.ToString() + " " + ElemTypesToString(dst_type);

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 5;
        Status status_exec = Executor(loop_count, 2, time_val, IGridIDft, m_ctx, src, dst, run_param.grid_len, DT_TRUE, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
            std::cout << "execute time: " << time_val.ToString() << std::endl;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::GridIDft";
            status_exec = Executor(5, 2, time_val, OpenCVGridIDft, src, ref, run_param.grid_len);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCVGridIDft execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            std::cout << "OpenCV time: " << time_val.ToString() << std::endl;
        }
        else
        {
            result.accu_benchmark = "GridIDft(target::none)";
            status_exec = IGridIDft(m_ctx, src, ref, run_param.grid_len, DT_TRUE, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1.0);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();
EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutAllMats();

        return 0;
    }
private:
    Context   *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_GRID_DFT_IDFT_UNIT_TEST_HPP__
