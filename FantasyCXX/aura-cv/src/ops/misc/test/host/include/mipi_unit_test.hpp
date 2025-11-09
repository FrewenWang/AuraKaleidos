/** @brief      : mipi uint test head for aura
 *  @file       : mipi_unit_test.hpp
 *  @author     : zhanghong16@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 13, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_MIPI_UINT_TEST_HPP__
#define AURA_OPS_MISC_MIPI_UINT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/tools/unit_test.h"

using namespace aura;

AURA_TEST_PARAM(MiscMipiParam,
                MatSize,  mat_sizes,
                ElemType, elem_type,
                OpTarget, target);

class MipiUnpackTest : public TestBase<MiscMipiParam::TupleTable, MiscMipiParam::Tuple>
{
public:
    MipiUnpackTest(Context *ctx, MiscMipiParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status ret = Status::ERROR;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        ret = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});

        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in MipiUnpackTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        MiscMipiParam run_param(GetParam((index)));
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        MiscMipiParam run_param(GetParam((index)));
        Scalar scale(0.8, 1.0);

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MatSize omat_sizes(run_param.mat_sizes, scale);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "unpack";
        result.input  = omat_sizes.ToString() + " " + ElemTypesToString(ElemType::U8);
        result.output = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status ret = Executor(loop_count, 2, time_val, IMipiUnpack, m_ctx, src, dst, run_param.target);

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
            result.accu_status = TestStatus::UNTESTED;
            result.accu_result = "no benchmark";
        }
        else
        {
            ret = IMipiUnpack(m_ctx, src, ref, TargetType::NONE);
            result.accu_benchmark = "MipiUnpack(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark MipiUnpack none execute failed\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }

            // compare
            MatCompare(m_ctx, dst, ref, cmp_result, 0);

            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

class MipiPackTest : public TestBase<MiscMipiParam::TupleTable, MiscMipiParam::Tuple>
{
public:
    MipiPackTest(Context *ctx, MiscMipiParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status ret = Status::ERROR;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        ret = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});

        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in MipiPackTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        MiscMipiParam run_param(GetParam((index)));
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        MiscMipiParam run_param(GetParam((index)));
        Scalar scale(1.25, 1.0);

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, ElemType::U16, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MatSize omat_sizes(run_param.mat_sizes, scale);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, omat_sizes.m_sizes, AURA_MEM_DEFAULT, omat_sizes.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "pack";
        result.input  = omat_sizes.ToString() + " " + ElemTypesToString(ElemType::U16);
        result.output = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        Status ret = Executor(loop_count, 2, time_val, IMipiPack, m_ctx, src, dst, run_param.target);

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
            result.accu_status = TestStatus::UNTESTED;
            result.accu_result = "no benchmark";;
        }
        else
        {
            ret = IMipiPack(m_ctx, src, ref, TargetType::NONE);
            result.accu_benchmark = "Mipipack(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark MipiPack none execute failed\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }

            // compare
            MatCompare(m_ctx, dst, ref, cmp_result, 0);

            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

#endif // AURA_OPS_MISC_MIPI_UINT_TEST_HPP__