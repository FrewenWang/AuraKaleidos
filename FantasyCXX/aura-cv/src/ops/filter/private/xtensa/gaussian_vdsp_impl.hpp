#ifndef AURA_OPS_FILTER_XTENSA_GAUSSIAN_VDSP_IMPL_HPP__
#define AURA_OPS_FILTER_XTENSA_GAUSSIAN_VDSP_IMPL_HPP__

#include "aura/ops/filter/xtensa/gaussian_vdsp.hpp"

#include "tileManager.h"
#include "tileManager_FIK_api.h"
#include "tileManager_api.h"

#include <new>

namespace aura
{
namespace xtensa
{

// Tp = MI_U8, MI_S8, MI_S16 MI_U16
template <typename Tp>
struct GaussianTraits
{
    using SumType = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;
    using KernelType = typename std::conditional<sizeof(Tp) == 1, xb_int32pr, xb_int64pr>::type;
    static constexpr MI_U32 Q = is_floating_point<Tp>::value ? 0 : (sizeof(Tp) == 2 ? 14 : 8);
};

class GaussianTile : public VdspOpTile
{
public:
    GaussianTile(TileManager tm);

    Status SetArgs(const TileWrapper *src, TileWrapper *dst, MI_S32 ksize, MI_F32 sigma);

    Status Initialize();

    Status DeInitialize();

    Status Run();

private:
    Status PrepareKmat();

private:
    MI_S32            m_ksize;
    MI_F32            m_sigma;
    ElemType          m_elem_type;
    MI_S32            m_channel;
    const TileWrapper *m_xv_src_tile;
    TileWrapper       *m_xv_dst_tile;
    AURA_VOID           *m_kernel;
};

class GaussianFrame : public VdspOpFrame
{
public:
    GaussianFrame(TileManager tm);

    Status SetArgs(const Mat *src, Mat *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    Status DeInitialize();

    Status Run();

private:
    static AURA_VOID Prepare(xvTileManager *xv_tm, RefTile *xv_ref_tile, AURA_VOID *obj, AURA_VOID *tiles, MI_S32 flag);

    static MI_S32 Execute(AURA_VOID *obj, AURA_VOID *tiles);

private:
    MI_S32       m_ksize;
    MI_F32       m_sigma;
    MI_S32       m_src_sizes;
    MI_S32       m_dst_sizes;
    GaussianTile *m_gaussian_tile;
};

MI_S32 Gaussian3x3Vdsp(const xvTile *src, xvTile *dst, AURA_VOID *kernel, ElemType elem_type, MI_S32 channel);
Status GaussianRpc(TileManager xv_tm, XtensaRpcParam &rpc_param);

using GaussianInParamVdsp = XtensaRpcParamType<Mat, Mat, MI_S32, MI_F32, BorderType, Scalar>;

} // namespace xtensa
} // namespace aura

#endif //AURA_OPS_FILTER_XTENSA_GAUSSIAN_VDSP_IMPL_HPP__