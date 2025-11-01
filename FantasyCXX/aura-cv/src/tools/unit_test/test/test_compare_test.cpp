#include "aura/tools/unit_test.h"

using namespace aura;

NEW_TESTCASE(unit_test, iaura_compare, none)
{
    const auto width   = 768;
    const auto height  = 512;
    const auto channel = 3;

    Context *ctx = UnitTest::GetInstance()->GetContext();

    {
        Mat img_result(ctx, ElemType::F32, Sizes3(height, width, channel));
        Mat img_ref(ctx, ElemType::F32, Sizes3(height, width, channel));

        const MI_F32 tolerate = 1;
        const MI_F32 step     = 0.3;

        for (MI_S32 i = 0; i < 10; ++i)
        {
            for (MI_S32 j = 0; j < Ceil(tolerate / step) + 10; ++j)
            {
                auto t = step * j;
                img_result.At<MI_F32>(j, i, 0) = t;
            }
        }

        MatCmpResult cmp_result;
        Status status = MatCompare(ctx, img_result, img_ref, cmp_result, tolerate, step);

        if (Status::ERROR != status)
        {
            AURA_LOGD(ctx, AURA_TAG, cmp_result.ToString().c_str());
            AURA_LOGD(ctx, AURA_TAG, "MatCompare success \n");
        }
        else
        {
            AURA_LOGD(ctx, AURA_TAG, "MatCompare faled \n");
        }
    }

    {
        Mat img_result(ctx, ElemType::F32, Sizes3(height, width, channel));
        Mat img_ref(ctx, ElemType::F32, Sizes3(height, width, channel));

        const MI_F32 tolerate = 4;
        const MI_F32 step     = 1;

        for (MI_S32 i = 0; i < 20; ++i)
        {
            for (MI_S32 j = 0; j < Ceil(tolerate / step) + 10; ++j)
            {
                auto t                        = step * j;
                img_result.At<MI_U8>(j, i, 0) = t;
            }
        }

        MatCmpResult cmp_result;
        Status status = MatCompare_<MI_U8, MI_U8>(ctx, img_result, img_ref, cmp_result, tolerate, step);

        if (Status::ERROR != status)
        {
            AURA_LOGD(ctx, AURA_TAG, cmp_result.ToString().c_str());
            AURA_LOGD(ctx, AURA_TAG, "MatCompare success \n");
        }
        else
        {
            AURA_LOGD(ctx, AURA_TAG, "MatCompare faled \n");
        }
    }

    AddTestResult(TestStatus::PASSED);
    return;
}

NEW_TESTCASE(unit_test, array_compare, none)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // *********************************
    //        Normal Array Test
    // *********************************

    // int compare
    {
        ArrayCmpResult result;
        MI_S32 int_arr0[] = {1, 2, 3, 4, 5}; 
        MI_S32 int_arr1[] = {1, 2, 3, 4, 5}; 

        if (ArrayCompare(ctx, int_arr0, int_arr1, 5, result) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // int compare
    {
        ArrayCmpResult result;
        MI_S32 int_arr0[] = {1, 2, 3, 4, 5}; 
        MI_S32 int_arr1[] = {1, 2, 1, 4, 2}; 

        if (ArrayCompare(ctx, int_arr0, int_arr1, 5, result, 1, 1) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // float compare
    {
        ArrayCmpResult result;
        MI_F32 f_arr0[] = {1, 2, 3, 4, 5}; 
        MI_F32 f_arr1[] = {1, 2, 3, 4, 5}; 

        if (ArrayCompare(ctx, f_arr0, f_arr1, 5, result) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // float compare
    {
        ArrayCmpResult result;
        MI_F32 f_arr0[] = {1, 2,   3,   4, 5}; 
        MI_F32 f_arr1[] = {1, 2.5, 2.4, 4, 2}; 

        if (ArrayCompare(ctx, f_arr0, f_arr1, 5, result, 0.5, 0.5) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // *********************************
    //          Vector Test
    // *********************************

    // int compare
    {
        ArrayCmpResult result;
        std::vector<MI_S32> vi0{1, 2, 3, 4, 5}; 
        std::vector<MI_S32> vi1{1, 2, 3, 4, 5}; 

        if (ArrayCompare(ctx, vi0.begin(), vi1.begin(), 5, result) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // int compare
    {
        ArrayCmpResult result;
        std::vector<MI_S32> vi0{1, 2, 3, 4, 5}; 
        std::vector<MI_S32> vi1{1, 2, 1, 4, 2}; 

        if (ArrayCompare(ctx, vi0.begin(), vi1.begin(), 5, result, 1, 2) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // float compare
    {
        ArrayCmpResult result;
        std::vector<MI_F32> vf0{1, 2, 3, 4, 5}; 
        std::vector<MI_F32> vf1{1, 2, 3, 4, 5}; 

        if (ArrayCompare(ctx, vf0.begin(), vf1.begin(), 5, result) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // float compare
    {
        ArrayCmpResult result;
        std::vector<MI_F32> vf0{1, 2,   3,   4, 5}; 
        std::vector<MI_F32> vf1{1, 2.5, 2.4, 4, 2}; 
        if (ArrayCompare(ctx, vf0.begin(), vf1.begin(), 5, result, 0.5, 0.5) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    // *********************************
    //          single value
    // *********************************
    {
        ArrayCmpResult result;
        MI_F32 v0 = 1; 
        MI_F32 v1 = 1; 
        if (ArrayCompare(ctx, &v0, &v1, 1, result, 0.5, 0.5) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    {
        ArrayCmpResult result;
        MI_F32 v0 = 1; 
        MI_F32 v1 = 2; 
        if (ArrayCompare(ctx, &v0, &v1, 1, result, 0.5, 0.5) == Status::OK && result.status)
        {
            AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
    }

    AddTestResult(TestStatus::PASSED);
    // *********************************
    //          Illegal Test
    // *********************************
    // {
    //     struct TestInfo
    //     {
    //         MI_F32 v0;
    //         MI_S32 v1;
    //     };

    //     ArrayCmpResult result;
    //     TestInfo arr0[2] = {{0, 0}, {1, 1}};
    //     TestInfo arr1[2] = {{0, 0}, {1, 1}};

    //     if (ArrayCompare(ctx, arr0, arr1, 2, result) == Status::OK && result.status)
    //     {
    //         AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
    //     }
    //     else
    //     {
    //         AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
    //     }
    // }

    // {
    //     ArrayCmpResult result;
    //     std::list<AURA_VOID*> l0 = {MI_NULL, MI_NULL}; 
    //     std::list<AURA_VOID*> l1 = {MI_NULL, MI_NULL}; 

    //     if (ArrayCompare(ctx, l0.begin(), l1.begin(), 1, result) == Status::OK && result.status)
    //     {
    //         AURA_LOGD(ctx, AURA_TAG, "result is OK: %s\n\n", result.ToString().c_str());
    //     }
    //     else
    //     {
    //         AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
    //     }
    // }
}

NEW_TESTCASE(unit_test, scalar_compare, none)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    {
        Scalar s0(0, 2,   3,    4);
        Scalar s1(0, 2.1, 3.05, 6);

        ScalarCmpResult result;
        result.is_detail = MI_TRUE;

        // use default interface
        if (ScalarCompare(ctx, s0, s1, result) != Status::OK || 0 == result.status)
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
        ScalarCmpPos cmp_info = result.GetMaxDiffPos();
        AURA_LOGD(ctx, AURA_TAG, "compare info: %s \n\n", cmp_info.ToString().c_str());

        // use RelativeDiff
        if (ScalarCompare<RelativeDiff>(ctx, s0, s1, result, 0.2, 1.0) != Status::OK || 0 == result.status)
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
        cmp_info = result.GetMaxDiffPos();
        AURA_LOGD(ctx, AURA_TAG, "compare info: %s \n\n", cmp_info.ToString().c_str());
    }

    {
        std::vector<Scalar> s0(10, Scalar(0, 2,   3,    4));
        std::vector<Scalar> s1(10, Scalar(0, 2.1, 3.05, 6));

        ScalarCmpResult result;

        // use default interface
        if (ScalarCompare(ctx, s0, s1, result) != Status::OK || 0 == result.status)
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
        ScalarCmpPos cmp_info = result.GetMaxDiffPos();
        AURA_LOGD(ctx, AURA_TAG, "compare info: %s \n\n", cmp_info.ToString().c_str());

        // use RelativeDiff
        if (ScalarCompare<RelativeDiff>(ctx, s0, s1, result, 0.2, 1.0) != Status::OK || 0 == result.status)
        {
            AURA_LOGE(ctx, AURA_TAG, "compare failed: %s \n\n", result.ToString().c_str());
        }
        cmp_info = result.GetMaxDiffPos();
        AURA_LOGD(ctx, AURA_TAG, "compare info: %s \n\n", cmp_info.ToString().c_str());
    }

    AddTestResult(TestStatus::PASSED);
}
