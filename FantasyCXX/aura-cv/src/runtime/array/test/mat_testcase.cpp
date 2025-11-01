#include "aura/runtime/mat.h"
#include "aura/tools/unit_test.h"

#include <fstream>

using namespace aura;

struct MatMembers
{
    MatMembers(MI_BOOL valid, ElemType elem_type, MI_S32 mem_type, MI_S64 total_bytes, ArrayType array_type, MI_S32 row_step, MI_S32 ref_count)
               : valid(valid), elem_type(elem_type), mem_type(mem_type), total_bytes(total_bytes), array_type(array_type),
                 row_step(row_step), ref_count(ref_count)
    {}

    MI_BOOL   valid;
    ElemType  elem_type;
    MI_S32    mem_type;
    MI_S64    total_bytes;
    ArrayType array_type;
    MI_S32    row_step;
    MI_S32    ref_count;
};

static Status CheckMat(Context *ctx, const Mat &src, const MatMembers &mat_members,
                       const MI_CHAR *file, const MI_CHAR *func, MI_S32 line)
{
    Status ret = Status::OK;
    ret |= TestCheckEQ(ctx, src.IsValid(),       mat_members.valid,       "check Mat::IsValid() failed\n",       file, func, line);
    ret |= TestCheckEQ(ctx, src.GetElemType(),   mat_members.elem_type,   "check Mat::GetElemType() failed\n",   file, func, line);
    ret |= TestCheckEQ(ctx, src.GetMemType(),    mat_members.mem_type,    "check Mat::GetMemType() failed\n",    file, func, line);
    ret |= TestCheckEQ(ctx, src.GetTotalBytes(), mat_members.total_bytes, "check Mat::GetTotalBytes() failed\n", file, func, line);
    ret |= TestCheckEQ(ctx, src.GetArrayType(),  mat_members.array_type,  "check Mat::GetArrayType() failed\n",  file, func, line);
    ret |= TestCheckEQ(ctx, src.GetRowStep(),    mat_members.row_step,    "check Mat::GetRowStep() failed\n",    file, func, line);
    ret |= TestCheckEQ(ctx, src.GetRefCount(),   mat_members.ref_count,   "check Mat::GetRefCount() failed\n",   file, func, line);
    return ret;
}

#define CHECK_MAT(ctx, src, ref)    CheckMat(ctx, src, ref, __FILE__, __FUNCTION__, __LINE__)

template <typename Tp>
static AURA_VOID FillMat(Mat &src)
{
    MI_S32 cnt = 0;

    const auto &size = src.GetSizes();
    for (MI_S32 y = 0; y < size.m_height; ++y)
    {
        for (MI_S32 x = 0; x < size.m_width; ++x)
        {
            for (MI_S32 c = 0; c < size.m_channel; ++c)
            {
                src.At<Tp>(y, x, c) = SaturateCast<Tp>(cnt++);
            }
        }
    }
}

template <typename Tp>
static Status CheckCmpMatMem(Context *ctx, const Mat &src, const Mat &mat_roi, Rect &roi,
                             const MI_CHAR *file, const MI_CHAR *func, MI_S32 line)
{
    Status ret = Status::OK;

    auto size = mat_roi.GetSizes();

    auto offset_x = roi.m_x;
    auto offset_y = roi.m_y;
    for (MI_S32 y = 0; y < size.m_height; ++y)
    {
        for (MI_S32 x = 0; x < size.m_width; ++x)
        {
            for (MI_S32 c = 0; c < size.m_channel; ++c)
            {
                ret |= TestCheckEQ(ctx, mat_roi.At<Tp>(y, x, c), src.At<Tp>(y + offset_y, x + offset_x, c),
                                   "CheckCmpMatMem failed\n", file, func, line);
            }
        }
    }
    return ret;
}

#define CHECK_CMP_MAT_MEM(ctx, type, src, dst, rois)    CheckCmpMatMem<type>(ctx, src, dst, rois, __FILE__, __FUNCTION__, __LINE__)

NEW_TESTCASE(runtime_array_mat_constructor_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Mat();
    Mat mat;
    ret |= AURA_CHECK_EQ(ctx, mat.IsValid(), MI_FALSE, "check Mat::IsValid() failed\n");

    // Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, MI_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes(), MI_BOOL init = MI_TRUE);
    mat = Mat(ctx, ElemType::U16, Sizes3(9, 6, 3), AURA_MEM_INVALID);
    ret |= AURA_CHECK_EQ(ctx, mat.IsValid(), MI_FALSE, "check Mat::IsValid() failed\n");
    mat = Mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes(6, 100));
    ret |= CHECK_MAT(ctx, mat, MatMembers(MI_TRUE, ElemType::U32, AURA_MEM_HEAP, 100 * 9, ArrayType::MAT, 8, 1));

    // Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes());
    AURA_VOID *ptr_heap = AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 1024, 128);
    Buffer buffer = ctx->GetMemPool()->GetBuffer(ptr_heap);
    ret |= AURA_CHECK_EQ(ctx, mat.IsValid(), MI_TRUE, "check Mat::IsValid() failed\n");
    mat = Mat(ctx, ElemType::U8, Sizes3(9, 6, 3), buffer);
    ret |= CHECK_MAT(ctx, mat, MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_HEAP, 3 * 6 * 9, ArrayType::MAT, 6, 0));
    AURA_FREE(ctx, ptr_heap);

    ptr_heap = AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 16, 128);
    buffer = ctx->GetMemPool()->GetBuffer(ptr_heap);
    ret |= AURA_CHECK_EQ(ctx, buffer.IsValid(), MI_TRUE, "check Buffer::IsValid() failed\n");
    mat = Mat(ctx, ElemType::U8, Sizes3(9, 6, 3), buffer);
    ret |= AURA_CHECK_EQ(ctx, mat.IsValid(), MI_FALSE, "check Mat::IsValid() failed\n");
    AURA_FREE(ctx, ptr_heap);

    // Mat(const Mat &mat, const Rect &roi);
    mat = Mat(ctx, ElemType::F64, Sizes3(4, 3, 2), AURA_MEM_HEAP, Sizes(4, 64));
    FillMat<MI_F64>(mat);
    Mat mat_roi = Mat(mat, Rect(1, 1, 2, 2));
    ret |= CHECK_MAT(ctx, mat_roi, MatMembers(MI_TRUE, ElemType::F64, AURA_MEM_HEAP, 128, ArrayType::MAT, 4, 2));
    Rect roi(1, 1, 2, 2);
    ret |= CHECK_CMP_MAT_MEM(ctx, MI_F64, mat, mat_roi, roi);
    ret |= AURA_CHECK_EQ(ctx, mat_roi.IsContinuous(), MI_FALSE, "check Mat::IsContinuous() failed\n");

    mat = Mat(ctx, ElemType::F32, Sizes3(4, 3, 2));
    ret |= AURA_CHECK_EQ(ctx, mat.IsContinuous(), MI_TRUE, "check Mat::IsContinuous() failed\n");

    // static Mat* Create(Context *ctx, ElemType elem_type, const Sizes3 &sizes, MI_S32 mem_type = AURA_MEM_DEFAULT, const Sizes &strides = Sizes());
    Mat *mat_ptr = Create<Mat>(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes(6, 100));
    ret |= CHECK_MAT(ctx, *mat_ptr, MatMembers(MI_TRUE, ElemType::U32, AURA_MEM_HEAP, 100 * 9, ArrayType::MAT, 8, 1));
    Delete<Mat>(ctx, &mat_ptr);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_init_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes());
    Mat mat0 = mat;
    ret |= AURA_CHECK_EQ(ctx, mat0.IsValid(), MI_TRUE, "check Mat::IsValid() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mat0.GetRefCount(), (MI_S32)2, "check Mat::GetRefCount() failed\n");
    mat = mat0.Clone();
    ret |= AURA_CHECK_EQ(ctx, mat0.IsValid(), MI_TRUE, "check Mat::IsValid() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mat0.GetRefCount(), (MI_S32)1, "check Mat::GetRefCount() failed\n");

    mat.Release();
    mat.Release();
    ret |= AURA_CHECK_EQ(ctx, mat.IsValid(), MI_FALSE, "check Mat::IsValid() failed\n");
    ret |= CHECK_MAT(ctx, mat0, MatMembers(MI_TRUE, ElemType::U32, AURA_MEM_HEAP, 9 * 6 * 3 * 4, ArrayType::MAT, 6, 1));

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_clone_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes(6, 100));
    FillMat<MI_U32>(mat);
    Mat mat_clone = mat.Clone();
    ret |= AURA_CHECK_EQ(ctx, mat_clone.IsEqual(mat), MI_TRUE, "check Mat::IsEqual() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mat_clone.IsSizesEqual(mat), MI_TRUE, "check Mat::IsSizesEqual() failed\n");
    ret |= AURA_CHECK_EQ(ctx, mat_clone.IsChannelEqual(mat), MI_TRUE, "check Mat::IsChannelEqual() failed\n");
    ret |= AURA_CHECK_IEQ(ctx, mat.GetData(), mat_clone.GetData(), "check Mat::GetData() failed\n");

    Rect roi = Rect();
    ret |= CHECK_CMP_MAT_MEM(ctx, MI_U32, mat, mat_clone, roi);

    mat_clone = mat.Clone(Rect(1, 2, 3, 4), Sizes(6, 8));
    ret |= CHECK_MAT(ctx, mat_clone, MatMembers(MI_TRUE, ElemType::U32, AURA_MEM_HEAP, 216, ArrayType::MAT, 3, 1));
    roi = Rect(1, 2, 3, 4);
    ret |= CHECK_CMP_MAT_MEM(ctx, MI_U32, mat, mat_clone, roi);

    mat.Show();

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_dump_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U8, Sizes3(100, 200, 3), AURA_MEM_HEAP);
    FillMat<MI_U8>(mat);

    mat.Dump("dump");
    std::fstream fin("dump", std::ios::in | std::ios::binary);
    if (fin)
    {
        MI_U8 *src_ptr = (MI_U8 *)AURA_ALLOC(ctx, mat.GetTotalBytes());

        fin.read((MI_CHAR *)src_ptr, mat.GetTotalBytes());
        fin.close();

        MI_U8 *ref_ptr = mat.GetBuffer().GetData<MI_U8*>();
        for (MI_S64 i = 0; i < mat.GetTotalBytes(); ++i)
        {
            ret |= AURA_CHECK_EQ(ctx, ref_ptr[i], src_ptr[i], "check mat memory failed\n");
            if (ret != Status::OK)
            {
                break;
            }
        }
        AURA_FREE(ctx, src_ptr);
    }
    else
    {
        ret = Status::ERROR;
    }

    Mat mat_load(ctx, ElemType::U8, Sizes3(100, 200, 3), AURA_MEM_HEAP);
    mat_load.Load("dump");
    Rect roi = Rect(0, 0, mat_load.GetSizes().m_width, mat_load.GetSizes().m_height);
    ret |= CHECK_CMP_MAT_MEM(ctx, MI_U8, mat, mat_load, roi);

    if (ret != Status::OK)
    {
        printf("%s %d error: %s\n", __FILE__, __LINE__, ctx->GetLogger()->GetErrorString().c_str());
    }

    {
        Mat mat_row = mat_load.RowRange(10, 20);
        for (MI_S32 i = 10; i < 20; i++)
        {
            MI_U8 *src = mat_load.Ptr<MI_U8>(i);
            MI_U8 *dst = mat_row.Ptr<MI_U8>(i - 10);
            for (MI_S32 j = 0; j < mat_row.GetSizes().m_width * mat_row.GetSizes().m_channel; j++)
            {
                ret |= AURA_CHECK_EQ(ctx, src[j], dst[j], "check mat memory failed\n");
            }
        }
    }

    {
        Mat mat_col = mat_load.ColRange(10, 20);
        for (MI_S32 i = 0; i < mat_load.GetSizes().m_height; i++)
        {
            MI_U8 *src = mat_load.Ptr<MI_U8>(i);
            MI_U8 *dst = mat_col.Ptr<MI_U8>(i);
            for (MI_S32 j = 10 * mat_col.GetSizes().m_channel; j < 20 * mat_col.GetSizes().m_channel; j++)
            {
                ret |= AURA_CHECK_EQ(ctx, src[j], dst[j - 10 * mat_col.GetSizes().m_channel], "check mat memory failed\n");
            }
        }
    }

    const Mat mat_load_new = mat_load;

    {
        Mat mat_row = mat_load_new.RowRange(10, 20);
        for (MI_S32 i = 10; i < 20; i++)
        {
            const MI_U8 *src = mat_load_new.Ptr<MI_U8>(i);
            MI_U8 *dst = mat_row.Ptr<MI_U8>(i - 10);
            for (MI_S32 j = 0; j < mat_row.GetSizes().m_width * mat_row.GetSizes().m_channel; j++)
            {
                ret |= AURA_CHECK_EQ(ctx, src[j], dst[j], "check mat memory failed\n");
            }
        }
    }

    {
        Mat mat_col = mat_load_new.ColRange(10, 20);
        for (MI_S32 i = 0; i < mat_load.GetSizes().m_height; i++)
        {
            const MI_U8 *src = mat_load_new.Ptr<MI_U8>(i);
            MI_U8 *dst = mat_col.Ptr<MI_U8>(i);
            for (MI_S32 j = 10 * mat_col.GetSizes().m_channel; j < 20 * mat_col.GetSizes().m_channel; j++)
            {
                ret |= AURA_CHECK_EQ(ctx, src[j], dst[j - 10 * mat_col.GetSizes().m_channel], "check mat memory failed\n");
            }
        }
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_print_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S32 elem_types[] = {static_cast<MI_S32>(ElemType::U8),  static_cast<MI_S32>(ElemType::S8),
                           static_cast<MI_S32>(ElemType::U16), static_cast<MI_S32>(ElemType::S16),
                           static_cast<MI_S32>(ElemType::U32), static_cast<MI_S32>(ElemType::S32),
                           static_cast<MI_S32>(ElemType::F32), static_cast<MI_S32>(ElemType::F64)};


    for (MI_S32 i = 0; i < static_cast<MI_S32>(sizeof(elem_types) / sizeof(MI_S32)); i++)
    {
        Mat mat = Mat(ctx, static_cast<ElemType>(elem_types[i]), Sizes3(20, 24, 1));

        std::string prefix_name = std::string();

        switch (mat.GetElemType())
        {
            case ElemType::U8:
            {
                FillMat<MI_U8>(mat);
                mat.At<MI_U8>(0, 0, 0) = 255;
                prefix_name += "u8";
                break;
            }

            case ElemType::S8:
            {
                FillMat<MI_S8>(mat);
                mat.At<MI_S8>(0, 0, 0) = -126;
                prefix_name += "s8";
                break;
            }

            case ElemType::U16:
            {
                FillMat<MI_U16>(mat);
                mat.At<MI_U16>(0, 0, 0) = 65535;
                prefix_name += "u16";
                break;
            }

            case ElemType::S16:
            {
                FillMat<MI_S16>(mat);
                mat.At<MI_S16>(0, 0, 0) = -32767;
                prefix_name += "s16";
                break;
            }

            case ElemType::U32:
            {
                FillMat<MI_U32>(mat);
                mat.At<MI_U32>(0, 0, 0) = 12345678;
                prefix_name += "u32";
                break;
            }

            case ElemType::S32:
            {
                FillMat<MI_S32>(mat);
                mat.At<MI_S32>(0, 0, 0) = -1234567;
                prefix_name += "s32";
                break;
            }

            case ElemType::F32:
            {
                FillMat<MI_F32>(mat);
                mat.At<MI_F32>(0, 0, 0) = 1.2345f;
                prefix_name += "f32";
                break;
            }

            case ElemType::F64:
            {
                FillMat<MI_F64>(mat);
                mat.At<MI_F64>(0, 0, 0) = 1.2345;
                prefix_name += "f64";
                break;
            }
            default:
            {
                AURA_LOGE(ctx, AURA_TAG, "do not surpport elem type F16");
                ret = Status::ERROR;
            }
        }

        Rect roi = Rect(0, 0, 10, 10);

        MI_CHAR str[128];
        MI_S32 mode = 10;

        mat.Print();   //terminal decimal print
        mat.Print(16); //terminal hex print

#if defined(AURA_BUILD_HEXAGON)
        snprintf(str, sizeof(str), "./%s_mode_%ld.txt", prefix_name.c_str(), mode);
#else
        snprintf(str, sizeof(str), "./%s_mode_%d.txt", prefix_name.c_str(), mode);
#endif // AURA_BUILD_HEXAGON
        mat.Print(10, roi, str); //decimal file print

        mode = 16;
#if defined(AURA_BUILD_HEXAGON)
        snprintf(str, sizeof(str), "./%s_mode_%ld.txt", prefix_name.c_str(), mode);
#else
        snprintf(str, sizeof(str), "./%s_mode_%d.txt", prefix_name.c_str(), mode);
#endif // AURA_BUILD_HEXAGON
        mat.Print(16, roi, str); //hex file print
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_poly_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes(6, 100));
    Array &array = mat;
    const Array &array0 = mat.Clone();
    ret |= AURA_CHECK_EQ(ctx, array.GetArrayType(), ArrayType::MAT, "check Array::GetArrayType() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetElemType(), ElemType::U32, "check Array::GetElemType() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetRefCount(), (MI_S32)1, "check Mat::GetRefCount() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetRowPitch(), (MI_S32)100, "check Mat::GetRowPitch() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetRowStep(), (MI_S32)8, "check Mat::GetRowStep() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetSizes(), Sizes3(9, 6, 3), "check Mat::GetSizes() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetStrides(), Sizes(9, 100), "check Mat::GetStrides() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.GetTotalBytes(), static_cast<MI_S64>(900), "check Mat::GetTotalBytes() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.IsEqual(array0), MI_TRUE, "check Mat::IsEqual() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.IsSizesEqual(array0), MI_TRUE, "check Mat::IsSizesEqual() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.IsChannelEqual(array0), MI_TRUE, "check Mat::IsChannelEqual() failed\n");
    ret |= AURA_CHECK_EQ(ctx, array.IsValid(), MI_TRUE, "check Mat::IsValid() failed\n");
    array.Dump("dump.bin");
    array.Show();
    array.Release();

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_array_mat_roi_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U8, Sizes3(64, 64, 3), AURA_MEM_HEAP);

    Mat left_top = mat.Roi(Rect(0,   0, 32, 32));
    Mat righ_top = mat.Roi(Rect(32,  0, 32, 32));
    Mat left_bot = mat.Roi(Rect(0,  32, 32, 32));
    Mat righ_bot = mat.Roi(Rect(32, 32, 32, 32));
    Mat midd_mid = mat.Roi(Rect(16, 16, 32, 32));

    memset(left_top.GetData(), 0, left_top.GetTotalBytes());
    AURA_LOGD(ctx, AURA_TAG, "left_top memset OK, %d data bytes: %d\n", (MI_S32)left_top.At<MI_U8>(0, 0, 0), left_top.GetTotalBytes());
    memset(righ_top.GetData(), 1, righ_top.GetTotalBytes());
    AURA_LOGD(ctx, AURA_TAG, "righ_top memset OK, %d data bytes: %d\n", (MI_S32)righ_top.At<MI_U8>(0, 0, 0), righ_top.GetTotalBytes());
    memset(left_bot.GetData(), 2, left_bot.GetTotalBytes());
    AURA_LOGD(ctx, AURA_TAG, "left_bot memset OK, %d data bytes: %d\n", (MI_S32)left_bot.At<MI_U8>(0, 0, 0), left_bot.GetTotalBytes());
    memset(righ_bot.GetData(), 3, righ_bot.GetTotalBytes());
    AURA_LOGD(ctx, AURA_TAG, "righ_bot memset OK, %d data bytes: %d\n", (MI_S32)righ_bot.At<MI_U8>(0, 0, 0), righ_bot.GetTotalBytes());
    memset(midd_mid.GetData(), 4, midd_mid.GetTotalBytes());
    AURA_LOGD(ctx, AURA_TAG, "midd_mid memset OK, %d data bytes: %d\n", (MI_S32)midd_mid.At<MI_U8>(0, 0, 0), midd_mid.GetTotalBytes());

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_mat_write_read_bmp_test)
{
    Status   ret = Status::ERROR;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MI_S32 channels[3] = {1, 3, 4};
    for (MI_S32 i = 0; i < 3; i++)
    {
        Mat mat(ctx, ElemType::U8, Sizes3(100, 200, channels[i]), AURA_MEM_HEAP);
        FillMat<MI_U8>(mat);

        std::string fname = "example" + std::to_string(channels[i]) + ".bmp";
        ret = WriteBmp(ctx, mat, fname);

        Mat mat_read = ReadBmp(ctx, fname);

        Rect roi = Rect(0, 0, mat.GetSizes().m_width, mat.GetSizes().m_height);
        ret |= CHECK_CMP_MAT_MEM(ctx, MI_U8, mat, mat_read, roi);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_mat_write_read_yuv_test)
{
    Status   ret = Status::ERROR;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    using FormatSizesVec = std::vector<std::pair<IauraFormat, std::vector<Sizes3>>>;

    FormatSizesVec array =
    {
        {IauraFormat::NV12, {Sizes3{200, 400, 1}, Sizes3{100, 200, 2}}},
        {IauraFormat::NV21, {Sizes3{200, 400, 1}, Sizes3{100, 200, 2}}},
        {IauraFormat::YU12, {Sizes3{200, 400, 1}, Sizes3{100, 200, 1}, Sizes3{100, 200, 1}}},
        {IauraFormat::YV12, {Sizes3{200, 400, 1}, Sizes3{100, 200, 1}, Sizes3{100, 200, 1}}},
        {IauraFormat::P010, {Sizes3{200, 400, 1}, Sizes3{100, 200, 2}}},
        {IauraFormat::P016, {Sizes3{200, 400, 1}, Sizes3{100, 200, 2}}},
        {IauraFormat::I422, {Sizes3{200, 400, 1}, Sizes3{200, 200, 1}, Sizes3{200, 200, 1}}},
        {IauraFormat::I444, {Sizes3{200, 400, 1}, Sizes3{200, 400, 1}, Sizes3{200, 400, 1}}},
    };

    for (size_t i = 0; i < array.size(); i++)
    {
        IauraFormat format    = array[i].first;
        MI_S32      len       = array[i].second.size();
        ElemType    elem_type = (IauraFormat::P010 == format || IauraFormat::P016 == format) ? ElemType::U16 : ElemType::U8;

        std::vector<Mat> mats;
        for (MI_S32 j = 0; j < len; j++)
        {
            Mat mat(ctx, elem_type, array[i].second[j], AURA_MEM_HEAP);
            FillMat<MI_U8>(mat);
            mats.emplace_back(mat);
        }

        std::string fname = "example.raw";
        ret = WriteYuv(ctx, mats, format, fname);

        std::vector<Mat> mats_read = ReadYuv(ctx, fname, format, Sizes(200, 400));

        for (MI_S32 j = 0; j < len; j++)
        {
            Rect roi = Rect(0, 0, mats[j].GetSizes().m_width, mats[j].GetSizes().m_height);
            ret |= CHECK_CMP_MAT_MEM(ctx, MI_U8, mats[j], mats_read[j], roi);
        }
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_mat_reshape_test)
{
    Status   ret = Status::ERROR;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    Mat mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP);

    //mat reshape test
    ret = mat.Reshape(Sizes3(3, 6, 9));
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "mat Reshape failed, error=%s", ctx->GetLogger()->GetErrorString().c_str());
        goto EXIT;
    }

    ret = mat.Reshape(Sizes3(3, 9, 6));
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "mat Reshape failed, error=%s", ctx->GetLogger()->GetErrorString().c_str());
        goto EXIT;
    }

    mat = Mat(ctx, ElemType::U32, Sizes3(9, 6, 3), AURA_MEM_HEAP, Sizes(0, 128));
    ret = mat.Reshape(Sizes3(9, 2, 9));
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "mat Reshape failed, error=%s", ctx->GetLogger()->GetErrorString().c_str());
        goto EXIT;
    }

EXIT:
    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_mat_yuv_test)
{
    Status   ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    //yuv iaura test
    Mat mat_yuv(ctx, ElemType::U8, Sizes3(192, 200, 1));

    std::vector<Mat> nv12_mats = GetYuvIaura(ctx, mat_yuv, IauraFormat::NV12);
    ret |= CHECK_MAT(ctx, nv12_mats[0], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 128 * 200, ArrayType::MAT, 200, 3));
    ret |= CHECK_MAT(ctx, nv12_mats[1], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 200, ArrayType::MAT, 100, 3));
    AURA_LOGI(ctx, AURA_TAG, "GetYuvIaura NV12 test ok\n");

    std::vector<Mat> yu12_mats = GetYuvIaura(ctx, mat_yuv, IauraFormat::YU12);
    ret |= CHECK_MAT(ctx, yu12_mats[0], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 128 * 200, ArrayType::MAT, 200, 6));
    ret |= CHECK_MAT(ctx, yu12_mats[1], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 100, ArrayType::MAT, 100, 6));
    ret |= CHECK_MAT(ctx, yu12_mats[2], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 100, ArrayType::MAT, 100, 6));
    AURA_LOGI(ctx, AURA_TAG, "GetYuvIaura YU12 test ok\n");

    std::vector<Mat> i422_mats = GetYuvIaura(ctx, mat_yuv, IauraFormat::I422);
    ret |= CHECK_MAT(ctx, i422_mats[0], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 96 * 200, ArrayType::MAT, 200, 9));
    ret |= CHECK_MAT(ctx, i422_mats[1], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 96 * 100, ArrayType::MAT, 100, 9));
    ret |= CHECK_MAT(ctx, i422_mats[2], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 96 * 100, ArrayType::MAT, 100, 9));
    AURA_LOGI(ctx, AURA_TAG, "GetYuvIaura I422 test ok\n");

    std::vector<Mat> i444_mats = GetYuvIaura(ctx, mat_yuv, IauraFormat::I444, MI_TRUE);
    ret |= CHECK_MAT(ctx, i444_mats[0], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 200, ArrayType::MAT, 200, 1));
    ret |= CHECK_MAT(ctx, i444_mats[1], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 200, ArrayType::MAT, 200, 1));
    ret |= CHECK_MAT(ctx, i444_mats[2], MatMembers(MI_TRUE, ElemType::U8, AURA_MEM_DMA_BUF_HEAP, 64 * 200, ArrayType::MAT, 200, 1));
    AURA_LOGI(ctx, AURA_TAG, "GetYuvIaura I444 test ok\n");

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}