#include "sample_array.hpp"

aura::Status MatSampleTest(aura::Context *ctx)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== Mat Constructor Begin ===================\n");

    // Mat();
    {
        aura::Mat mat;
        AURA_LOGD(ctx, SAMPLE_TAG, "Construct a empty mat, mat type is invalid\n");
    }

    // Mat(aura::Context *ctx, aura::ElemType elem_type, const aura::Sizes3 &sizes, DT_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());
    {
        aura::Mat mat(ctx, aura::ElemType::U32, aura::Sizes3(9, 6, 3));
        AURA_LOGD(ctx, SAMPLE_TAG, "Construct a mat, mat type is default\n");
        mat.Show();
    }

    // Mat(aura::Context *ctx, aura::ElemType elem_type, const aura::Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());
    {
        DT_VOID *ptr_heap = AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 1024, 128);
        aura::Buffer buffer = ctx->GetMemPool()->GetBuffer(ptr_heap);
        aura::Mat mat = aura::Mat(ctx, aura::ElemType::U8, aura::Sizes3(9, 6, 3), buffer);
        mat.Show();
        AURA_FREE(ctx, ptr_heap);
    }

    // Mat(const aura::Mat &mat);
    {
        aura::Mat mat0(ctx, aura::ElemType::U32, aura::Sizes3(9, 6, 3));
        aura::Mat mat1(mat0);
        AURA_LOGD(ctx, SAMPLE_TAG, "Ref count of mat0 and mat1 is 2, and use same buffer\n");
        mat1.Show();
    }

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== Mat Member Function Begin ===================\n");
    {
        aura::Mat mat(ctx, aura::ElemType::U32, aura::Sizes3(9, 6, 3), AURA_MEM_HEAP, aura::Sizes(9, 100));

        DT_VOID *data = mat.GetData();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetData() function, mat data ptr is %p\n", data);

        DT_S32 row = 5;
        DT_U32 *ptr = mat.Ptr<DT_U32>(row);
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat Ptr() function, mat row ptr is %p\n", ptr);

        DT_U32 row_value = mat.At<DT_U32>(row, 0, 0);
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat At() function, value is %d\n", row_value);

        aura::ArrayType array_type = mat.GetArrayType();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetArrayType() function, array type is %s\n", ArrayTypesToString(array_type).c_str());

        aura::ElemType elem_type = mat.GetElemType();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetElemType() function, element type is %s\n", ElemTypesToString(elem_type).c_str());

        DT_S32 ref_count = mat.GetRefCount();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetRefCount() function, ref count is %d\n", ref_count);

        DT_S32 row_pitch = mat.GetRowPitch();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetRowPitch() function, row pitch is %d\n", row_pitch);

        DT_S32 row_step = mat.GetRowStep();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetRowStep() function, row step is %d\n", row_step);

        aura::Sizes3 size = mat.GetSizes();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetSizes() function, mat size is (%d, %d, %d)\n", size.m_height, size.m_width, size.m_channel);

        aura::Sizes stride = mat.GetStrides();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetStrides() function, mat stride is (%d, %d)\n", stride.m_height, stride.m_width);

        DT_S64 total_byte = mat.GetTotalBytes();
        AURA_LOGD(ctx, SAMPLE_TAG, "Use mat GetTotalBytes() function, mat total byte is %ld\n", total_byte);
    }

    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MatSampleTest: Test Successed ===================\n");

    return aura::Status::OK;
}