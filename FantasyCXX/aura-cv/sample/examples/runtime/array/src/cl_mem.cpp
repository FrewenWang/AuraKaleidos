#include "sample_array.hpp"

aura::Status CLMemSampleTest(aura::Context *ctx)
{
#if defined(AURA_ENABLE_OPENCL)
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== CL Mat Constructor Begin ===================\n");

    // CLMem()
    {
        aura::CLMem cl_mat;
        AURA_LOGD(ctx, SAMPLE_TAG, "Construct a empty cl mat, cl mat type is invalid\n");
    }

    // CLMem(aura::Context *ctx, const CLMemParam &cl_param, aura::ElemType elem_type, const aura::Sizes3 &sizes, const Sizes &strides = Sizes());
    {
        aura::CLMem cl_mat(ctx, aura::CLMemParam(CL_MEM_READ_ONLY), aura::ElemType::U32, aura::Sizes3(9, 6, 3));
        if (!cl_mat.IsValid())
        {
            AURA_LOGE(ctx, SAMPLE_TAG, "Construct a cl mat failed\n");
            return aura::Status::ERROR;
        }
        
        cl_mat.Show();
    }

    // CLMem(aura::Context *ctx, const CLMemParam &cl_param, aura::ElemType elem_type, const aura::Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());
    {
        aura::Mat mat(ctx, aura::ElemType::U32, aura::Sizes3(9, 6, 3));
        aura::CLMem cl_mat(ctx, aura::CLMemParam(CL_MEM_READ_WRITE), mat.GetElemType(), mat.GetSizes(), mat.GetBuffer(), mat.GetStrides());
        if (!cl_mat.IsValid())
        {
            AURA_LOGE(ctx, SAMPLE_TAG, "Construct a cl mat failed\n");
            return aura::Status::ERROR;
        }
        
        cl_mat.Show();
    }

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== CL Mat Member Function Begin ===================\n");
    {
        aura::Mat mat(ctx, aura::ElemType::U32, aura::Sizes3(512, 512, 1));
        aura::CLMem cl_mat = aura::CLMem::FromArray(ctx, mat, aura::CLMemParam(CL_MEM_READ_ONLY, CL_R));
        AURA_LOGD(ctx, SAMPLE_TAG, "Use cl mat FromArray() function, create a cl mat from host mat\n");
        cl_mat.Show();
        if (!cl_mat.IsValid())
        {
            AURA_LOGE(ctx, SAMPLE_TAG, "Construct a cl mat failed\n");
            return aura::Status::ERROR;
        }
        DT_U32 *data = cl_mat.GetCLMemPtr<DT_U32>();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use cl mat GetCLMemPtr() function, cl mat data ptr is %p\n", data);

        DT_S32 channel = cl_mat.GetCLIauraChannelNum(CL_R);
        AURA_LOGD(ctx, SAMPLE_TAG, "Use cl mat GetCLIauraChannelNum() function, channel is %d\n", channel);
    }

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== CLMemSampleTest: Test Successed ===================\n");

    return aura::Status::OK;

#else
    AURA_LOGE(ctx, SAMPLE_TAG, "This platform doesn't support opencl\n");
    return aura::Status::ERROR;
#endif
}