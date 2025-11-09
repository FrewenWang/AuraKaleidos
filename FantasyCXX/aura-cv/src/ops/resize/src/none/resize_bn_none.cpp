#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

Status ResizeBnNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnNoneImpl<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnNoneImpl<DT_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnNoneImpl<DT_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnNoneImpl<DT_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: DT_S16");
            }
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ResizeBnNoneImpl<MI_F16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: MI_F16");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = ResizeBnNoneImpl<DT_F32>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnNoneImpl failed, type: DT_F32");
            }
            break;
        }
#endif // AURA_BUILD_HOST

        default :
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace
