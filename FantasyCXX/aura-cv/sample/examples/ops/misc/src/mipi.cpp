#include "sample_misc.hpp"

aura::Status MipiSampleTest(aura::Context *ctx, aura::TargetType type)
{
    AURA_LOGD(ctx, SAMPLE_TAG, "=================== MipiSampleTest Begin ===================\n");

    // ---------------------
    //     prepare data
    // ---------------------
    // create mat and resize
    aura::Sizes3 ori_size = {487, 487, 1};
    aura::Sizes3 src_size = {5000, 5000, 1};
    aura::Mat ori(ctx, aura::ElemType::U8, ori_size);
    aura::Mat src(ctx, aura::ElemType::U8, src_size);
    if (!(src.IsValid() && ori.IsValid()))
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "src or ori mat is invalid.");
        return aura::Status::ERROR;
    }

    // load data
    aura::Status status = ori.Load("data/comm/cameraman_487x487.gray");
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Mipi sample test load failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    status = IResize(ctx, ori, src, aura::InterpType::LINEAR);
    if (status != aura::Status::OK)
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "Mipi sample resize failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return status;
    }

    aura::Status status1, status2;

    {
        // ---------------------
        //     mipi pack prepare data
        // ---------------------
        // create mat
        aura::Mat src_u16(ctx, aura::ElemType::U16, src_size);
        IConvertTo(ctx, src, src_u16);

        aura::Sizes3 dst_size = {5000, 6250, 1};
        aura::Mat dst(ctx, aura::ElemType::U8, dst_size);

        // ---------------------
        //     run pack interface
        // ---------------------
        // print info
        AURA_LOGD(ctx, SAMPLE_TAG, "Test mipi pack param detail:\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
                  ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

        // run and time
        aura::Time start_time = aura::Time::Now();
        status1 = IMipiPack(ctx, src_u16, dst, aura::OpTarget(type));
        aura::Time end_time = aura::Time::Now();

        // ---------------------
        //     check pack result
        // ---------------------
        if (status1 != aura::Status::OK)
        {
            AURA_LOGE(ctx, SAMPLE_TAG, "Mipi Pack Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
            AURA_LOGE(ctx, SAMPLE_TAG, "=================== MipiSampleTest: Pack Test Failed ===================\n");
        }
        else
        {
            AURA_LOGD(ctx, SAMPLE_TAG, "Mipi Pack running time = %s\n", (end_time - start_time).ToString().c_str());
            dst.Dump("./mipi_pack_test.raw");
            AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./mipi_pack_test.raw\n");
            AURA_LOGD(ctx, SAMPLE_TAG, "=================== MipiSampleTest: Pack Test Succeeded ===================\n");
        }
    }

    {
        // ---------------------
        //     mipi unpack prepare data
        // ---------------------
        // create mat
        aura::Sizes3 dst_size = {5000, 4000, 1};
        aura::Mat dst(ctx, aura::ElemType::U8, dst_size);

        // ---------------------
        //     run pack interface
        // ---------------------
        // print info
        AURA_LOGD(ctx, SAMPLE_TAG, "Test mipi unpack param detail:\n");
        AURA_LOGD(ctx, SAMPLE_TAG, "elem_type(%s), mat_size(%s), target(%s)\n",
                  ElemTypesToString(src.GetElemType()).c_str(), src.GetSizes().ToString().c_str(), TargetTypeToString(type).c_str());

        // run and time
        aura::Time start_time = aura::Time::Now();
        status2 = IMipiUnpack(ctx, src, dst, aura::OpTarget(type));
        aura::Time end_time = aura::Time::Now();

        // ---------------------
        //     check unpack result
        // ---------------------
        if (status2 != aura::Status::OK)
        {
            AURA_LOGE(ctx, SAMPLE_TAG, "Mipi UnPack Interface Execute Fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
            AURA_LOGE(ctx, SAMPLE_TAG, "=================== MipiSampleTest: UnPack Test Failed ===================\n");
        }
        else
        {
            AURA_LOGD(ctx, SAMPLE_TAG, "Mipi UnPack running time = %s\n", (end_time - start_time).ToString().c_str());
            dst.Dump("./mipi_unpack_test.raw");
            AURA_LOGD(ctx, SAMPLE_TAG, "result stored at ./mipi_unpack_test.raw\n");
            AURA_LOGD(ctx, SAMPLE_TAG, "=================== MipiSampleTest: UnPack Test Succeeded ===================\n");
        }
    }

    return (status1 | status2);
}