#include "aura/tools/json/serialize/runtime/array.hpp"
#include "aura/tools/json/serialize/runtime/types.hpp"
#include "aura/tools/json/json_helper.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Mat &mat)
{
    if (!mat.IsValid())
    {
        json["sizes"]    = Sizes3();
        json["strides"]  = Sizes();
        json["elemtype"] = ElemType::INVALID;
        json["path"]     = std::string();
    }
    else
    {
        json["sizes"]    = mat.GetSizes();
        json["strides"]  = mat.GetStrides();
        json["elemtype"] = mat.GetElemType();
        json["path"]     = JsonHelper::GetInstance().GetArrayPath(&mat);
    }
}

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Mat &mat)
{
    Sizes3      sizes;
    Sizes       strides;
    ElemType    elem_type;
    std::string path;

    json.at("sizes").get_to(sizes);
    json.at("strides").get_to(strides);
    json.at("elemtype").get_to(elem_type);
    json.at("path").get_to(path);

    Context *ctx = JsonHelper::GetInstance().GetContext();
    if (DT_NULL == ctx)
    {
        return;
    }

    mat = Mat(ctx, elem_type, sizes, AURA_MEM_DEFAULT, strides);
    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "mat is invalid");
        return;
    }

    Status ret = mat.Load(path);
    if (ret != Status::OK)
    {
        std::string info = "load mat " + path + "failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
    }
}

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Mat *mat)
{
    if (DT_NULL == mat)
    {
        to_json(json, Mat());
    }
    else
    {
        to_json(json, *mat);
    }
}

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Mat *&mat)
{
    Sizes3      sizes;
    Sizes       strides;
    ElemType    elem_type;
    std::string path;

    json.at("sizes").get_to(sizes);
    json.at("strides").get_to(strides);
    json.at("elemtype").get_to(elem_type);
    json.at("path").get_to(path);

    Context *ctx = JsonHelper::GetInstance().GetContext();
    if (DT_NULL == ctx)
    {
        return;
    }

    mat = Create<Mat>(ctx, elem_type, sizes, AURA_MEM_DEFAULT, strides);
    if ((DT_NULL == mat) || (!mat->IsValid()))
    {
        Delete<Mat>(ctx, &mat);
        return;
    }

    Status ret = mat->Load(path);
    if (ret != Status::OK)
    {
        std::string info = "load mat " + path + "failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
    }
}

#if defined(AURA_ENABLE_OPENCL)
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const CLMem &cl_mem)
{
    if (!cl_mem.IsValid())
    {
        json["sizes"]    = Sizes3();
        json["strides"]  = Sizes();
        json["elemtype"] = ElemType::INVALID;
        json["path"]     = std::string();
    }
    else
    {
        json["sizes"]    = cl_mem.GetSizes();
        json["strides"]  = cl_mem.GetStrides();
        json["elemtype"] = cl_mem.GetElemType();
        json["path"]     = JsonHelper::GetInstance().GetArrayPath(&cl_mem);
    }
}

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, CLMem &cl_mem)
{
    Sizes3      sizes;
    Sizes       strides;
    ElemType    elem_type;
    std::string path;

    json.at("sizes").get_to(sizes);
    json.at("strides").get_to(strides);
    json.at("elemtype").get_to(elem_type);
    json.at("path").get_to(path);

    Context *ctx = JsonHelper::GetInstance().GetContext();
    if (DT_NULL == ctx)
    {
        return;
    }

    cl_mem = CLMem(ctx, CLMemParam(CL_MEM_READ_WRITE), elem_type, sizes, strides);
    if (!cl_mem.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "cl_mem is invalid");
        return;
    }

    Status ret = cl_mem.Load(path);
    if (ret != Status::OK)
    {
        std::string info = "load cl_mem " + path + "failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
    }
}

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const CLMem *cl_mem)
{
    if (DT_NULL == cl_mem)
    {
        to_json(json, CLMem());
    }
    else
    {
        to_json(json, *cl_mem);
    }
}

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, CLMem *&cl_mem)
{
    Sizes3      sizes;
    Sizes       strides;
    ElemType    elem_type;
    std::string path;

    json.at("sizes").get_to(sizes);
    json.at("strides").get_to(strides);
    json.at("elemtype").get_to(elem_type);
    json.at("path").get_to(path);

    Context *ctx = JsonHelper::GetInstance().GetContext();
    if (DT_NULL == ctx)
    {
        return;
    }

    cl_mem = Create<CLMem>(ctx, CLMemParam(CL_MEM_READ_WRITE), elem_type, sizes, strides);
    if ((DT_NULL == cl_mem) || (!cl_mem->IsValid()))
    {
        Delete<CLMem>(ctx, &cl_mem);
        return;
    }

    Status ret = cl_mem->Load(path);
    if (ret != Status::OK)
    {
        std::string info = "load cl_mem " + path + "failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
    }
}
#endif

} // namespace aura