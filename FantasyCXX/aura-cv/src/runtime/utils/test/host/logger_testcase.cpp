#include "aura/runtime/logger.h"
#include "aura/tools/unit_test.h"

using namespace aura;

static DT_VOID StdoutTest()
{
    // test with context
    aura::Config cfg;
    cfg.SetLog(LogOutput::STDOUT, LogLevel::DEBUG);

    std::shared_ptr<Context> ctx(new Context(cfg));
    if (DT_NULL == ctx)
    {
        return;
    }

    Status ret = ctx->Initialize();
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

    // define a very long string
    std::string long_char(1000, 'a');
    long_char += '\n';
    long_char += std::string(1000, 'b') + '\n';
    long_char += std::string(1000, 'c') + '\n';
    long_char += std::string(1000, 'd') + '\n';
    long_char += std::string(1000, 'e') + '\n';
    long_char += "_end";

    AURA_LOGE(ctx.get(), "verylong_log", "%s \n", long_char.c_str());
}

static DT_VOID LogcatTest()
{
    // test with context
    aura::Config cfg;
    cfg.SetLog(LogOutput::LOGCAT, LogLevel::DEBUG);

    std::shared_ptr<Context> ctx(new Context(cfg));
    if (DT_NULL == ctx)
    {
        return;
    }

    Status ret = ctx->Initialize();
    if (ret != Status::OK)
    {
        return;
    }

    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_LOGE(ctx.get(), "logcat_ctx", "test for log error (with context) %d \n", i);
        AURA_LOGI(ctx.get(), "logcat_ctx", "test for log info  (with context) %d \n", i);
        AURA_LOGD(ctx.get(), "logcat_ctx", "test for log debug (with context) %d \n", i);
    }

    // test with PRINT
    for (DT_S32 i = 0; i < 10; i++)
    {
        AURA_PRINTE("logcat_PRINT", "test for log error (AURA_PRINT macro) %d \n", i);
        AURA_PRINTI("logcat_PRINT", "test for log info  (AURA_PRINT macro) %d \n", i);
        AURA_PRINTD("logcat_PRINT", "test for log debug (AURA_PRINT macro) %d \n", i);
    }

    // use logcat interface
    for (DT_S32 i = 0; i < 10; i++)
    {
        aura::LogcatPrint("logcat_print", LogLevel::ERROR, "test for log error (logcat print) %d \n", i);
        aura::LogcatPrint("logcat_print", LogLevel::INFO,  "test for log info  (logcat print) %d \n", i);
        aura::LogcatPrint("logcat_print", LogLevel::DEBUG, "test for log debug (logcat print) %d \n", i);
    }

    // define a very long string
    std::string long_char(1000, 'a');
    long_char += '\n';
    long_char += std::string(1000, 'b') + '\n';
    long_char += std::string(1000, 'c') + '\n';
    long_char += std::string(1000, 'd') + '\n';
    long_char += std::string(1000, 'e') + '\n';
    long_char += "_end";

    AURA_LOGE(ctx.get(), "verylong_log", "%s \n", long_char.c_str());
}

static DT_VOID FileTest()
{
    // test with context
    aura::Config cfg;
    cfg.SetLog(LogOutput::FILE, LogLevel::DEBUG, "log_ctx.txt");

    std::shared_ptr<Context> ctx(new Context(cfg));
    if (DT_NULL == ctx)
    {
        return;
    }

    Status ret = ctx->Initialize();
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
    FILE *fp = fopen("log_file.txt", "a+");
    for (DT_S32 i = 0; i < 10; i++)
    {
        aura::FilePrint(fp, "file_print", "test for log error (file print) %d \n", i);
        aura::FilePrint(fp, "file_print", "test for log info  (file print) %d \n", i);
        aura::FilePrint(fp, "file_print", "test for log debug (file print) %d \n", i);
    }

    // define a very long string
    std::string long_char(1000, 'a');
    long_char += '\n';
    long_char += std::string(1000, 'b') + '\n';
    long_char += std::string(1000, 'c') + '\n';
    long_char += std::string(1000, 'd') + '\n';
    long_char += std::string(1000, 'e') + '\n';
    long_char += "_end";

    AURA_LOGE(ctx.get(), "verylong_log", "%s \n", long_char.c_str());

    if (fp)
    {
        fclose(fp);
    }
}

NEW_TESTCASE(runtime_utils_logger_test)
{
    StdoutTest();
    LogcatTest();
    FileTest();
}