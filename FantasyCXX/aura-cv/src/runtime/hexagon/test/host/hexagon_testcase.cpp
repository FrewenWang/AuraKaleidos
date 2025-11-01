#include "aura/runtime/hexagon.h"
#include "aura/tools/unit_test.h"

using namespace aura;

struct CustomType
{
    MI_S32                         base;
    HexagonPowerLevel              level;
    Sequence<MI_U8>                seq;
    KeyPoint_<MI_F32>              keypoint;
    Point2_<MI_S16>                point2;
    Point3_<MI_U32>                point3;
    Rect_<MI_S32>                  rect;
    Scalar_<MI_F32>                scalar;
    Sizes2_<MI_F64>                sizes2;
    Sizes3_<MI_U64>                sizes3;
    std::string                    str;
    std::vector<KeyPoint_<MI_F32>> keypoint_array;
};

template <typename Tp, typename std::enable_if<std::is_same<Tp, CustomType>::value>::type* = MI_NULL>
static Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &param)
{
    Status ret = rpc_param->Set(param.base, param.level, param.seq, param.keypoint, param.point2,
                                param.point3, param.rect, param.scalar, param.sizes2, param.sizes3, param.str, param.keypoint_array);
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "Set failed\n");
    }
    return ret;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, CustomType>::value>::type* = MI_NULL>
static Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &param)
{
    Status ret = rpc_param->Get(param.base, param.level, param.seq, param.keypoint, param.point2,
                                param.point3, param.rect, param.scalar, param.sizes2, param.sizes3, param.str, param.keypoint_array);
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "Get failed\n");
    }
    return ret;
}

static MI_BOOL operator!=(const KeyPoint_<MI_F32> &a, const KeyPoint_<MI_F32> &b)
{
    return a.m_pt != b.m_pt || a.m_size != b.m_size || a.m_angle != b.m_angle ||
           a.m_response != b.m_response || a.m_octave != b.m_octave || a.m_class_id != b.m_class_id;
}

NEW_TESTCASE(runtime_hexagon_hexagon_rpc_param_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    AURA_LOGI(ctx, AURA_TAG, "aura hexagon version : %s\n", ctx->GetHexagonEngine()->GetVersion().c_str());

    using CustomTypeWrapper = HexagonRpcParamType<CustomType>;

    CustomType param_in, param_out;

    MI_U8 data_in[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, data_out[10];

    param_in.base           = 1;
    param_in.level          = HexagonPowerLevel::LOW;
    param_in.seq            = {data_in, 10};
    param_in.keypoint       = KeyPoint_<MI_F32>(1, 1, 0.5f);
    param_in.point2         = Point2_<MI_S16>(1, 2);
    param_in.point3         = Point3_<MI_U32>(1, 2, 3);
    param_in.rect           = Rect_<MI_S32>(1, 2, 3, 4);
    param_in.scalar         = Scalar_<MI_F32>(1, 2, 3, 4);
    param_in.sizes2         = Sizes2_<MI_F64>(1, 2);
    param_in.sizes3         = Sizes3_<MI_U64>(1, 2, 3);
    param_in.str            = "runtime_hexagon_hexagon_rpc_param_test";
    param_in.keypoint_array = {{1, 1, 0.5f}, {2, 2, 1.5f}};

    param_out.seq = {data_out, 10};

    HexagonRpcParam rpc_param(ctx);
    CustomTypeWrapper param(ctx, rpc_param);
    param.Set(param_in);
    param.Get(param_out, MI_TRUE);

    Status ret = Status::OK;

    ret |= AURA_CHECK_EQ(ctx, param_in.base,      param_out.base,      "check MI_S32 failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.level,     param_out.level,     "check HexagonPowerLevel failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.seq.len,   param_out.seq.len,   "check Sequence<MI_U8> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.point2,    param_out.point2,    "check Point2_<MI_S16> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.point3,    param_out.point3,    "check Point3_<MI_U32> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.rect,      param_out.rect,      "check Rect_<MI_S32> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.scalar,    param_out.scalar,    "check Scalar_<MI_F32> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.sizes2,    param_out.sizes2,    "check Sizes2_<MI_F64> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.sizes3,    param_out.sizes3,    "check Sizes3_<MI_U64> failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.str,       param_out.str,       "check std::string failed\n");
    ret |= AURA_CHECK_EQ(ctx, param_in.keypoint_array.size(), param_out.keypoint_array.size(), "check Sequence<KeyPoint_<MI_F32>> failed\n");
    for (MI_S32 i = 0; i < param_in.seq.len; i++)
    {
        ret |= AURA_CHECK_EQ(ctx, param_in.seq.data[i], param_out.seq.data[i], "check std::Sequence<MI_U8> failed\n");
    }
    if (param_in.keypoint != param_out.keypoint)
    {
        AURA_LOGE(ctx, AURA_TAG, "check KeyPoint_<MI_F32> failed\n");
        ret = Status::ERROR;
    }
    for (size_t i = 0; i < param_in.keypoint_array.size(); i++)
    {
        if (param_in.keypoint_array[i] != param_out.keypoint_array[i])
        {
            AURA_LOGE(ctx, AURA_TAG, "std::vector<KeyPoint_<MI_F32>*>\n");
            ret = Status::ERROR;
        }
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}

NEW_TESTCASE(runtime_hexagon_hexagon_query_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    HexagonEngine *engine = ctx->GetHexagonEngine();

    Status ret = Status::OK;

    HardwareInfo hw_info;

    ret = engine->QueryHWInfo(hw_info);

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "QueryHWInfo failed.\n");
        AddTestResult(AURA_GET_TEST_STATUS(ret));
    }
    else
    {
        AURA_LOGI(ctx, AURA_TAG, "HexagonEngine->QueryHWInfo arch_version : V%x\n", hw_info.arch_version);
        AURA_LOGI(ctx, AURA_TAG, "HexagonEngine->QueryHWInfo num_hvx_units: %d\n", hw_info.num_hvx_units);
        AURA_LOGI(ctx, AURA_TAG, "HexagonEngine->QueryHWInfo vtcm_layout  : %s\n", VtcmLayoutToString(hw_info.vtcm_layout).c_str());
    }

    RealTimeInfo rt_info;

    ret |= AURA_CHECK_EQ(ctx, Status::OK, engine->QueryRTInfo(HexagonRTQueryType::CURRENT_FREQ, rt_info), "QueryRTInfo CURRENT_FREQ failed.\n");
    AURA_LOGI(ctx, AURA_TAG, "HexagonEngine->QueryRTInfo CURRENT_FREQ : %f MHz\n", rt_info.cur_freq);

    ret |= AURA_CHECK_EQ(ctx, Status::OK, engine->QueryRTInfo(HexagonRTQueryType::VTCM_INFO, rt_info), "QueryRTInfo VTCM_INFO failed.\n");
    AURA_LOGI(ctx, AURA_TAG, "HexagonEngine->QueryRTInfo vtcm_layout  : %s\n", VtcmLayoutToString(rt_info.vtcm_layout).c_str());

    AddTestResult(AURA_GET_TEST_STATUS(ret));

}