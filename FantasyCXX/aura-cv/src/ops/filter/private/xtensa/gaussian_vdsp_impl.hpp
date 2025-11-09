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

// Tp = DT_U8, DT_S8, DT_S16 DT_U16
template <typename Tp>
struct GaussianTraits
{
    using SumType = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;
    using KernelType = typename std::conditional<sizeof(Tp) == 1, xb_int32pr, xb_int64pr>::type;
    static constexpr DT_U32 Q = is_floating_point<Tp>::value ? 0 : (sizeof(Tp) == 2 ? 14 : 8);
};

class GaussianTile : public VdspOpTile
{
public:
    GaussianTile(TileManager tm);

    Status SetArgs(const TileWrapper *src, TileWrapper *dst, DT_S32 ksize, DT_F32 sigma);

    Status Initialize();

    Status DeInitialize();

    Status Run();

private:
    Status PrepareKmat();

private:
    DT_S32            m_ksize;
    DT_F32            m_sigma;
    ElemType          m_elem_type;
    DT_S32            m_channel;
    const TileWrapper *m_xv_src_tile;
    TileWrapper       *m_xv_dst_tile;
    DT_VOID           *m_kernel;
};

class GaussianFrame : public VdspOpFrame
{
public:
    GaussianFrame(TileManager tm);

    Status SetArgs(const Mat *src, Mat *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar());

    Status DeInitialize();

    Status Run();

private:
    static DT_VOID Prepare(xvTileManager *xv_tm, RefTile *xv_ref_tile, DT_VOID *obj, DT_VOID *tiles, DT_S32 flag);

    static DT_S32 Execute(DT_VOID *obj, DT_VOID *tiles);

private:
    DT_S32       m_ksize;
    DT_F32       m_sigma;
    DT_S32       m_src_sizes;
    DT_S32       m_dst_sizes;
    GaussianTile *m_gaussian_tile;
};

DT_S32 Gaussian3x3Vdsp(const xvTile *src, xvTile *dst, DT_VOID *kernel, ElemType elem_type, DT_S32 channel);
Status GaussianRpc(TileManager xv_tm, XtensaRpcParam &rpc_param);

using GaussianInParamVdsp = XtensaRpcParamType<Mat, Mat, DT_S32, DT_F32, BorderType, Scalar>;

} // namespace xtensa
} // namespace aura

#endif //AURA_OPS_FILTER_XTENSA_GAUSSIAN_VDSP_IMPL_HPP__