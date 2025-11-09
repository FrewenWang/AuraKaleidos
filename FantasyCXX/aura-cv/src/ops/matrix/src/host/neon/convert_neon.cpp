#include "convert_to_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#define AURA_LANES_F32 (4)

namespace aura
{

template <typename Tp> struct CvtVecTraits;

template <> struct CvtVecTraits<DT_U8>  { using Type = uint8x8_t;     };
template <> struct CvtVecTraits<DT_S8>  { using Type = int8x8_t;      };
template <> struct CvtVecTraits<DT_U16> { using Type = uint16x8_t;    };
template <> struct CvtVecTraits<DT_S16> { using Type = int16x8_t;     };
template <> struct CvtVecTraits<DT_U32> { using Type = uint32x4x2_t;  };
template <> struct CvtVecTraits<DT_S32> { using Type = int32x4x2_t;   };
template <> struct CvtVecTraits<MI_F16> { using Type = float16x8_t;   };
template <> struct CvtVecTraits<DT_F32> { using Type = float32x4x2_t; };

template <typename Dt, typename St = Dt, typename std::enable_if<(std::is_same<Dt, St>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE typename CvtVecTraits<Dt>::Type Cvt8(const typename CvtVecTraits<St>::Type &v)
{
    return v;
}

#define DECL_CVT8_FUNC(type_src, type_dst) \
    template <typename Dt, typename std::enable_if<(std::is_same<Dt, type_dst>::value)>::type* = DT_NULL> \
    AURA_ALWAYS_INLINE typename CvtVecTraits<Dt>::Type Cvt8(const typename CvtVecTraits<type_src>::Type &v)

DECL_CVT8_FUNC(DT_S32, DT_U32)
{
    int32x4_t vqs32_zero;
    neon::vdup(vqs32_zero, 0);
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vreinterpret(neon::vmax(v.val[0], vqs32_zero));
    v2q.val[1] = neon::vreinterpret(neon::vmax(v.val[1], vqs32_zero));
    return v2q;
}

DECL_CVT8_FUNC(DT_U32, DT_S32)
{
    uint32x4_t vqu32_max;
    neon::vdup(vqu32_max, (DT_U32)2147483647);
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vreinterpret(neon::vmin(v.val[0], vqu32_max));
    v2q.val[1] = neon::vreinterpret(neon::vmin(v.val[1], vqu32_max));
    return v2q;
}

DECL_CVT8_FUNC(DT_S16, DT_U16)
{
    int16x8_t vqs16_zero;
    neon::vdup(vqs16_zero, 0);
    return neon::vreinterpret(neon::vmax(v, vqs16_zero));
}

DECL_CVT8_FUNC(DT_U16, DT_S16)
{
    uint16x8_t vqu16_max;
    neon::vdup(vqu16_max, (DT_U16)32767);
    return neon::vreinterpret(neon::vmin(v, vqu16_max));
}

DECL_CVT8_FUNC(DT_S8, DT_U8)
{
    int8x8_t vds8_zero;
    neon::vdup(vds8_zero, 0);
    return neon::vreinterpret(neon::vmax(v, vds8_zero));
}

DECL_CVT8_FUNC(DT_U8, DT_S8)
{
    uint8x8_t vdu8_max;
    neon::vdup(vdu8_max, (DT_U8)127);
    return neon::vreinterpret(neon::vmin(v, vdu8_max));
}

DECL_CVT8_FUNC(DT_S32, DT_F32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vcvt<Dt>(v.val[0]);
    v2q.val[1] = neon::vcvt<Dt>(v.val[1]);
    return v2q;
}

DECL_CVT8_FUNC(DT_U32, DT_F32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vcvt<Dt>(v.val[0]);
    v2q.val[1] = neon::vcvt<Dt>(v.val[1]);
    return v2q;
}

DECL_CVT8_FUNC(DT_F32, DT_S32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vcvtn<Dt>(v.val[0]);
    v2q.val[1] = neon::vcvtn<Dt>(v.val[1]);
    return v2q;
}

DECL_CVT8_FUNC(DT_F32, DT_U32)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(v));
}

DECL_CVT8_FUNC(DT_S32, DT_U16)
{
    return neon::vcombine(neon::vqmovun(v.val[0]), neon::vqmovun(v.val[1]));
}

DECL_CVT8_FUNC(DT_S32, DT_S16)
{
    return neon::vcombine(neon::vqmovn(v.val[0]), neon::vqmovn(v.val[1]));
}
DECL_CVT8_FUNC(DT_U32, DT_U16)
{
    return neon::vcombine(neon::vqmovn(v.val[0]), neon::vqmovn(v.val[1]));
}

DECL_CVT8_FUNC(DT_U32, DT_S16)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(v));
}

DECL_CVT8_FUNC(DT_S16, DT_S32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vmovl(neon::vgetlow(v));
    v2q.val[1] = neon::vmovl(neon::vgethigh(v));
    return v2q;
}

DECL_CVT8_FUNC(DT_U16, DT_S32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vmovl(neon::vgetlow(v));
    v2q.val[1] = neon::vmovl(neon::vgethigh(v));
    return v2q;
}

DECL_CVT8_FUNC(DT_U16, DT_U32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vmovl(neon::vgetlow(v));
    v2q.val[1] = neon::vmovl(neon::vgethigh(v));
    return v2q;
}

DECL_CVT8_FUNC(DT_S16, DT_U32)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_S16, DT_U8)
{
    return neon::vqmovun(v);
}

DECL_CVT8_FUNC(DT_S16, DT_S8)
{
    return neon::vqmovn(v);
}

DECL_CVT8_FUNC(DT_U16, DT_U8)
{
    return neon::vqmovn(v);
}

DECL_CVT8_FUNC(DT_U16, DT_S8)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(v));
}

DECL_CVT8_FUNC(DT_U8, DT_S16)
{
    return neon::vmovl(v);
}

DECL_CVT8_FUNC(DT_U8, DT_U16)
{
    return neon::vmovl(v);
}

DECL_CVT8_FUNC(DT_S8, DT_S16)
{
    return neon::vmovl(v);
}

DECL_CVT8_FUNC(DT_S8, DT_U16)
{
    return Cvt8<Dt>(Cvt8<DT_U8>(v));
}

DECL_CVT8_FUNC(DT_U16, DT_F32)
{
    return Cvt8<Dt>(Cvt8<DT_U32>(v));
}

DECL_CVT8_FUNC(DT_S16, DT_F32)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(v));
}

DECL_CVT8_FUNC(DT_F32, DT_U16)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(v));
}

DECL_CVT8_FUNC(DT_F32, DT_S16)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(v));
}

DECL_CVT8_FUNC(DT_U32, DT_U8)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_S32, DT_U8)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_U32, DT_S8)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(v));
}

DECL_CVT8_FUNC(DT_S32, DT_S8)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(v));
}

DECL_CVT8_FUNC(DT_U8, DT_U32)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_S8, DT_U32)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_U8, DT_S32)
{
    return Cvt8<Dt>(Cvt8<DT_U16>(v));
}

DECL_CVT8_FUNC(DT_S8, DT_S32)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(v));
}

DECL_CVT8_FUNC(DT_U8, DT_F32)
{
    return Cvt8<Dt>(Cvt8<DT_U32>(Cvt8<DT_U16>(v)));
}

DECL_CVT8_FUNC(DT_S8, DT_F32)
{
    return Cvt8<Dt>(Cvt8<DT_S32>(Cvt8<DT_S16>(v)));
}

DECL_CVT8_FUNC(DT_F32, DT_U8)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(Cvt8<DT_S32>(v)));
}

DECL_CVT8_FUNC(DT_F32, DT_S8)
{
    return Cvt8<Dt>(Cvt8<DT_S16>(Cvt8<DT_S32>(v)));
}

#if defined(AURA_ENABLE_NEON_FP16)
DECL_CVT8_FUNC(DT_F32, MI_F16)
{
    return neon::vcombine(neon::vcvt<Dt>(v.val[0]), neon::vcvt<Dt>(v.val[1]));
}

DECL_CVT8_FUNC(MI_F16, DT_F32)
{
    typename CvtVecTraits<Dt>::Type v2q;
    v2q.val[0] = neon::vcvt<Dt>(neon::vgetlow(v));
    v2q.val[1] = neon::vcvt<Dt>(neon::vgethigh(v));
    return v2q;
}

DECL_CVT8_FUNC(MI_F16, DT_U8)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(MI_F16, DT_S8)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(MI_F16, DT_U16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(MI_F16, DT_S16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(MI_F16, DT_U32)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(MI_F16, DT_S32)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_U8, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_S8, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_U16, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_S16, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_U32, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}

DECL_CVT8_FUNC(DT_S32, MI_F16)
{
    return Cvt8<Dt>(Cvt8<DT_F32>(v));
}
#endif

template <typename Dt, typename St, typename std::enable_if<((sizeof(St) == 1))>::type* = DT_NULL>
AURA_ALWAYS_INLINE typename CvtVecTraits<Dt>::Type LoadPairAs(const St *p)
{
    return Cvt8<Dt>(neon::vload1(p));
}

template <typename Dt, typename St, typename std::enable_if<((sizeof(St) == 2))>::type* = DT_NULL>
AURA_ALWAYS_INLINE typename CvtVecTraits<Dt>::Type LoadPairAs(const St *p)
{
    return Cvt8<Dt>(neon::vload1q(p));
}

template <typename Dt, typename St, typename std::enable_if<((sizeof(St) == 4))>::type* = DT_NULL>
AURA_ALWAYS_INLINE typename CvtVecTraits<Dt>::Type LoadPairAs(const St *p)
{
    typename CvtVecTraits<St>::Type v2q;
    v2q.val[0] = neon::vload1q(p);
    v2q.val[1] = neon::vload1q(p + AURA_LANES_F32);
    return Cvt8<Dt>(v2q);
}

template <typename Dt, typename St, typename std::enable_if<((sizeof(Dt) < 4))>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID StorePairAs(Dt *p, const typename CvtVecTraits<St>::Type &v)
{
    typename CvtVecTraits<Dt>::Type vq = Cvt8<Dt>(v);
    neon::vstore(p, vq);
}

template <typename Dt, typename St, typename std::enable_if<((sizeof(Dt) == 4))>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID StorePairAs(Dt *p, const typename CvtVecTraits<St>::Type &v)
{
    typename CvtVecTraits<Dt>::Type v2q = Cvt8<Dt>(v);
    neon::vstore(p, v2q.val[0]);
    neon::vstore(p + AURA_LANES_F32, v2q.val[1]);
}

enum class CvtMethod
{
    NO_SCALE    = 0,
    MUL_ADD,
    MUL_ONLY,
    ADD_ONLY,
};

template <CvtMethod METHOD, typename St, typename Dt,
          typename std::enable_if<(CvtMethod::NO_SCALE == METHOD)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtCore(const St *src, Dt *dst, float32x4_t &va, float32x4_t &vb)
{
    AURA_UNUSED(va);
    AURA_UNUSED(vb);
    StorePairAs<Dt, Dt>(dst, LoadPairAs<Dt, St>(src));
}

template <CvtMethod METHOD, typename St, typename Dt,
          typename std::enable_if<(CvtMethod::MUL_ADD == METHOD)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtCore(const St *src, Dt *dst, float32x4_t &va, float32x4_t &vb)
{
    auto v2q = LoadPairAs<DT_F32, St>(src);
    v2q.val[0] = neon::vmla(vb, va, v2q.val[0]);
    v2q.val[1] = neon::vmla(vb, va, v2q.val[1]);
    StorePairAs<Dt, DT_F32>(dst, v2q);
}

template <CvtMethod METHOD, typename St, typename Dt,
          typename std::enable_if<(CvtMethod::MUL_ONLY == METHOD)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtCore(const St *src, Dt *dst, float32x4_t &va, float32x4_t &vb)
{
    AURA_UNUSED(vb);
    auto v2q = LoadPairAs<DT_F32, St>(src);
    v2q.val[0] = neon::vmul(v2q.val[0], va);
    v2q.val[1] = neon::vmul(v2q.val[1], va);
    StorePairAs<Dt, DT_F32>(dst, v2q);
}

template <CvtMethod METHOD, typename St, typename Dt,
          typename std::enable_if<(CvtMethod::ADD_ONLY == METHOD)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtCore(const St *src, Dt *dst, float32x4_t &va, float32x4_t &vb)
{
    AURA_UNUSED(va);
    auto v2q = LoadPairAs<DT_F32, St>(src);
    v2q.val[0] = neon::vadd(v2q.val[0], vb);
    v2q.val[1] = neon::vadd(v2q.val[1], vb);
    StorePairAs<Dt, DT_F32>(dst, v2q);
}

template <CvtMethod METHOD, typename St, typename Dt>
static Status ConvertToNeonImpl(const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, DT_S32 start_row, DT_S32 end_row)
{
    float32x4_t vqf32_a, vqf32_b;
    neon::vdup(vqf32_a, alpha);
    neon::vdup(vqf32_b, beta);

    Sizes3 sz = src.GetSizes();
    DT_S32 align = AURA_LANES_F32 * 2;
    DT_S32 width = (sz.m_width * sz.m_channel);
    DT_S32 width_align = (width & (-align));

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const St *src_row = src.Ptr<St>(y);
        Dt       *dst_row = dst.Ptr<Dt>(y);

        DT_S32 x = 0;
        for (x = 0; x < width_align; x += align)
        {
            CvtCore<METHOD>(src_row + x, dst_row + x, vqf32_a, vqf32_b);
        }
        if (x < width)
        {
            x = width - align;
            CvtCore<METHOD>(src_row + x, dst_row + x, vqf32_a, vqf32_b);
        }
    }
    return Status::OK;
}

template <typename St, typename Dt>
static Status ConvertToNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, const OpTarget &target)
{
    AURA_UNUSED(target);
    DT_BOOL no_alpha = (Abs(alpha - 1.0) < DBL_EPSILON);
    DT_BOOL no_beta  = (Abs(beta) < DBL_EPSILON);
    CvtMethod method = (no_alpha && no_beta) ? CvtMethod::NO_SCALE
                                             : ((no_alpha) ? CvtMethod::ADD_ONLY
                                                           : ((no_beta) ? CvtMethod::MUL_ONLY
                                                                        : CvtMethod::MUL_ADD));

    if (CvtMethod::NO_SCALE == method && (src.GetElemType() == dst.GetElemType()))
    {
        return src.CopyTo(dst);
    }

    Status ret = Status::ERROR;
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    switch (method)
    {
        case CvtMethod::NO_SCALE:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height,
                                  ConvertToNeonImpl<CvtMethod::NO_SCALE, St, Dt>, std::cref(src), std::ref(dst), alpha, beta);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor ConvertToNeonImpl CvtMethod::NO_SCALE failed.");
            }
            break;
        }
        case CvtMethod::MUL_ADD:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height,
                                  ConvertToNeonImpl<CvtMethod::MUL_ADD, St, Dt>, std::cref(src), std::ref(dst), alpha, beta);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor ConvertToNeonImpl CvtMethod::MUL_ADD failed.");
            }
            break;
        }
        case CvtMethod::MUL_ONLY:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height,
                                  ConvertToNeonImpl<CvtMethod::MUL_ONLY, St, Dt>, std::cref(src), std::ref(dst), alpha, beta);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor ConvertToNeonImpl CvtMethod::MUL_ONLY failed.");
            }
            break;
        }
        case CvtMethod::ADD_ONLY:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height,
                                  ConvertToNeonImpl<CvtMethod::ADD_ONLY, St, Dt>, std::cref(src), std::ref(dst), alpha, beta);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor ConvertToNeonImpl CvtMethod::ADD_ONLY failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "ConvertToNeon method is unsupported.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ConvertToNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ConvertToNeonHelper<Tp, DT_U8>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ConvertToNeonHelper<Tp, DT_S8>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ConvertToNeonHelper<Tp, DT_U16>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ConvertToNeonHelper<Tp, DT_S16>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ConvertToNeonHelper<Tp, DT_U32>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ConvertToNeonHelper<Tp, DT_S32>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ConvertToNeonHelper<Tp, MI_F16>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = ConvertToNeonHelper<Tp, DT_F32>(ctx, src, dst, alpha, beta, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNeonHelper<Tp, DT_F32> failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "ConvertToNeon dst type is unsupported.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

ConvertToNeon::ConvertToNeon(Context *ctx, const OpTarget &target) : ConvertToImpl(ctx, target)
{}

Status ConvertToNeon::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta)
{
    if (ConvertToImpl::SetArgs(src, dst, alpha, beta) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConvertToImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConvertToNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = ConvertToNeonHelper<DT_U8>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ConvertToNeonHelper<DT_S8>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ConvertToNeonHelper<DT_U16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ConvertToNeonHelper<DT_S16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ConvertToNeonHelper<DT_U32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ConvertToNeonHelper<DT_S32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ConvertToNeonHelper<MI_F16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = ConvertToNeonHelper<DT_F32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeonHelper<DT_F32> failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNeon src type is unsupported.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura