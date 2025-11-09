#include "aura/ops/filter/gaussian.hpp"
#include "gaussian_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

template <typename Tp>
struct GaussianTraits
{
    // DT_U32 DT_S32 DT_F32, DT_U8 DT_U16 DT_S16
    using KernelType = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;
    // DT_F32, DT_U8 DT_U32 DT_S32, DT_U16 DT_S16
    static constexpr DT_U32 Q = is_floating_point<Tp>::value ? 0 : (sizeof(Tp) > 1 ? 14 : 8);
};

GaussianHvx::GaussianHvx(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

Status GaussianHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                            BorderType border_type, const Scalar &border_value)
{
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7 && ksize != 9)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7/9");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16 &&
        elem_type != ElemType::U32 && elem_type != ElemType::S32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16/u32/s32");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianHvx::PrepareKmat()
{
    std::vector<DT_F32> kernel = GetGaussianKernel(m_ksize, m_sigma);

#define GET_GAUSSIAN_KMAT(type)                                     \
    using KernelType   = typename GaussianTraits<type>::KernelType; \
    constexpr DT_U32 Q = GaussianTraits<type>::Q;                   \
                                                                    \
    m_kmat = GetGaussianKmat<KernelType, Q>(m_ctx, kernel);         \

    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            GET_GAUSSIAN_KMAT(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GET_GAUSSIAN_KMAT(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GET_GAUSSIAN_KMAT(DT_S16)
            break;
        }

        case ElemType::U32:
        {
            GET_GAUSSIAN_KMAT(DT_U32)
            break;
        }

        case ElemType::S32:
        {
            GET_GAUSSIAN_KMAT(DT_S32)
            break;
        }

        case ElemType::F32:
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }

        default:
        {
            m_kmat = Mat();
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

#undef GET_GAUSSIAN_KMAT

    return Status::OK;
}

Status GaussianHvx::Initialize()
{
    if (GaussianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::Initialize() failed");
        return Status::ERROR;
    }

    // Prepare kmat
    if (PrepareKmat() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PrepareKmat failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianHvx::Run()
{
    /// 我们
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            /// TODO 我们重点看一下3*3的卷积的实现
            ret = Gaussian3x3Hvx(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value);
            break;
        }

        case 5:
        {
            ret = Gaussian5x5Hvx(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value);
            break;
        }

        case 7:
        {
            ret = Gaussian7x7Hvx(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value);
            break;
        }

        case 9:
        {
            ret = Gaussian9x9Hvx(m_ctx, *src, *dst, m_kmat, m_border_type, m_border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status GaussianHvx::DeInitialize()
{
    m_kmat.Release();

    if (GaussianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::string GaussianHvx::ToString() const
{
    return GaussianImpl::ToString();
}

Status GaussianRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    DT_S32 ksize;
    DT_F32 sigma;
    BorderType border_type;
    Scalar border_value;

    GaussianInParamHvx in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, ksize, sigma, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }
    ///// 实例化对象的高斯函数对象
    Gaussian gaussian(ctx, OpTarget::Hvx());
    /// 对应的Op调用，我们将反序列化之后的数据来进行
    return OpCall(ctx, gaussian, &src, &dst, ksize, sigma, border_type, border_value);
}

////// TODO 这个地方就是将我们的OP算法的身份标签注册到的远程的
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_GAUSSIAN_OP_NAME, GaussianRpc);

} // namespace aura