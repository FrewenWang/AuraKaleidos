#include "aura/runtime/logger.h"
#include "aura/tools/unit_test.h"

using namespace aura;

static DT_VOID StdoutTest()
{
    // test with context
    std::shared_ptr<Context> ctx(new Context());
    Status ret = ctx->Initialize(LogOutput::STDOUT, LogLevel::DEBUG, "");
    if (ret != Status::OK)
    {
        return;
    }

    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_LOGE(ctx.get(), "std_out_ctx", "test for log error (with context) %d \n", i);
        AURA_LOGI(ctx.get(), "std_out_ctx", "test for log info  (with context) %d \n", i);
        AURA_LOGD(ctx.get(), "std_out_ctx", "test for log debug (with context) %d \n", i);
    }

    // test with PRINT
    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_PRINTE("std_out_PRINT", "test for log error (AURA_PRINT macro) %d \n", i);
        AURA_PRINTI("std_out_PRINT", "test for log info  (AURA_PRINT macro) %d \n", i);
        AURA_PRINTD("std_out_PRINT", "test for log debug (AURA_PRINT macro) %d \n", i);
    }

    // use stdout interface
    for (DT_S32 i = 0; i < 10; i++)
    {
        aura::StdoutPrint("std_out_print", LogLevel::ERROR, "test for log error (stdout print) %d \n", i);
        aura::StdoutPrint("std_out_print", LogLevel::INFO,  "test for log info  (stdout print) %d \n", i);
        aura::StdoutPrint("std_out_print", LogLevel::DEBUG, "test for log debug (stdout print) %d \n", i);
    }
}

static DT_VOID FarfTest()
{
    // test with context
    std::shared_ptr<Context> ctx(new Context());
    Status ret = ctx->Initialize(LogOutput::FARF, LogLevel::DEBUG, "");
    if (ret != Status::OK)
    {
        return;
    }

    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_LOGE(ctx.get(), "farf_ctx", "test for log error (with context) %d \n", i);
        AURA_LOGI(ctx.get(), "farf_ctx", "test for log info  (with context) %d \n", i);
        AURA_LOGD(ctx.get(), "farf_ctx", "test for log debug (with context) %d \n", i);
    }

    // test with PRINT
    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_PRINTE("farf_PRINT", "test for log error (AURA_PRINT macro) %d \n", i);
        AURA_PRINTI("farf_PRINT", "test for log info  (AURA_PRINT macro) %d \n", i);
        AURA_PRINTD("farf_PRINT", "test for log debug (AURA_PRINT macro) %d \n", i);
    }

    // use farf interface
    for (DT_S32 i = 0; i < 10; i++)
    {
        aura::FarfPrint(LogLevel::ERROR, "test for log error (logcat print) %d \n", i);
        aura::FarfPrint(LogLevel::INFO,  "test for log info  (logcat print) %d \n", i);
        aura::FarfPrint(LogLevel::DEBUG, "test for log debug (logcat print) %d \n", i);
    }
}

static DT_VOID FileTest()
{
    // test with context
    std::shared_ptr<Context> ctx(new Context());
    Status ret = ctx->Initialize(LogOutput::FILE, LogLevel::DEBUG, "/data/local/tmp/log_ctx.txt");
    if (ret != Status::OK)
    {
        return;
    }

    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_LOGE(ctx.get(), "file_ctx", "test for log error (with context) %d \n", i);
        AURA_LOGI(ctx.get(), "file_ctx", "test for log info  (with context) %d \n", i);
        AURA_LOGD(ctx.get(), "file_ctx", "test for log debug (with context) %d \n", i);
    }

    // use file interface
    // open file log_file.txt
    FILE *fp = fopen("/data/local/tmp/file_print.txt", "a+");
    for (DT_S32 i = 0; i < 10; i++)
    {
        aura::FilePrint(fp, "file_print", "test for log error (file print) %d \n", i);
        aura::FilePrint(fp, "file_print", "test for log info  (file print) %d \n", i);
        aura::FilePrint(fp, "file_print", "test for log debug (file print) %d \n", i);
    }

    if (fp)
    {
        fclose(fp);
    }
}

NEW_TESTCASE(runtime_utils_logger_test)
{
    StdoutTest();
    FarfTest();
    FileTest();
}