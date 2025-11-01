#ifndef AURA_OPS_MATRIX_ROTATE_IMPL_HPP__
#define AURA_OPS_MATRIX_ROTATE_IMPL_HPP__

#include "aura/ops/matrix/rotate.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{
    template <typename Tp, RotateType Rt, MI_S32 C> struct RotateNoneFunctor;

template <typename Tp, MI_S32 C>
struct RotateNoneFunctor<Tp, RotateType::ROTATE_90, C>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;

    AURA_ALWAYS_INLINE Status operator()(const Mat &src, Mat &dst, MI_S32 start_blk, MI_S32 end_blk)
    {
        using BLOCK = struct { Tp val[C]; };

        const MI_U8 *src_data = (MI_U8 *)src.GetData();
        MI_U8 *dst_data       = (MI_U8 *)dst.GetData();

        MI_U32 src_pitch = src.GetRowPitch();
        MI_U32 dst_pitch = dst.GetRowPitch();

        MI_S32 width  = dst.GetSizes().m_width;
        MI_S32 height = dst.GetSizes().m_height;

        MI_S32 start_row = start_blk * BLOCK_SIZE;
        MI_S32 end_row   = Min(end_blk * BLOCK_SIZE, height);

        MI_S32 x = 0;
        MI_S32 y = Max(static_cast<MI_S32>(0), start_row);
        MI_S32 w = width;
        MI_S32 h = Min(height, end_row);

        MI_S32 w_align4 = (w & (-4));
        MI_S32 h_align4 = (h & (-4));

        for (; y < h_align4; y += 4)
        {
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * y);
            BLOCK *d1 = (BLOCK *)(dst_data + dst_pitch * (y + 1));
            BLOCK *d2 = (BLOCK *)(dst_data + dst_pitch * (y + 2));
            BLOCK *d3 = (BLOCK *)(dst_data + dst_pitch * (y + 3));

            x = 0;
            for (; x < w_align4; x += 4)
            {
                MI_S32 sx = width - x - 1;
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * sx);
                const BLOCK *s1 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 3));

                d0[x]     = s0[0];
                d0[x + 1] = s1[0];
                d0[x + 2] = s2[0];
                d0[x + 3] = s3[0];

                d1[x]     = s0[1];
                d1[x + 1] = s1[1];
                d1[x + 2] = s2[1];
                d1[x + 3] = s3[1];

                d2[x]     = s0[2];
                d2[x + 1] = s1[2];
                d2[x + 2] = s2[2];
                d2[x + 3] = s3[2];

                d3[x]     = s0[3];
                d3[x + 1] = s1[3];
                d3[x + 2] = s2[3];
                d3[x + 3] = s3[3];
            }

            for (; x < w; x++)
            {
                MI_S32 sx = width - x - 1;
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * sx);

                d0[x] = s0[0];
                d1[x] = s0[1];
                d2[x] = s0[2];
                d3[x] = s0[3];
            }
        }

        for (; y < h; y++)
        {
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * y);

            x = 0;
            for (; x < w_align4; x += 4)
            {
                MI_S32 sx = width - x - 1;
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * sx);
                const BLOCK *s1 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (sx - 3));

                d0[x]     = s0[0];
                d0[x + 1] = s1[0];
                d0[x + 2] = s2[0];
                d0[x + 3] = s3[0];
            }

            for (; x < w; x++)
            {
                MI_S32 sx = width - x - 1;
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * sx);

                d0[x] = s0[0];
            }
        }
        return Status::OK;
    }
};

template <typename Tp, MI_S32 C>
struct RotateNoneFunctor<Tp, RotateType::ROTATE_180, C>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;

    AURA_ALWAYS_INLINE Status operator()(const Mat &src, Mat &dst, MI_S32 start_blk, MI_S32 end_blk)
    {
        using BLOCK = struct { Tp val[C]; };

        const MI_U8 *src_data = (MI_U8 *)src.GetData();
        MI_U8 *dst_data       = (MI_U8 *)dst.GetData();

        MI_U32 src_pitch = src.GetRowPitch();
        MI_U32 dst_pitch = dst.GetRowPitch();

        MI_S32 width  = dst.GetSizes().m_width;
        MI_S32 height = dst.GetSizes().m_height;

        MI_S32 start_row = start_blk * BLOCK_SIZE;
        MI_S32 end_row   = Min(end_blk * BLOCK_SIZE, height);

        MI_S32 x = 0;
        MI_S32 y = Max(static_cast<MI_S32>(0), start_row);
        MI_S32 w = width;
        MI_S32 h = Min(height, end_row);

        MI_S32 w_align4 = (w & (-4));
        MI_S32 h_align4 = (h & (-4));

        for (; y < h_align4; y += 4)
        {
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * (height - y - 1));
            BLOCK *d1 = (BLOCK *)(dst_data + dst_pitch * (height - y - 2));
            BLOCK *d2 = (BLOCK *)(dst_data + dst_pitch * (height - y - 3));
            BLOCK *d3 = (BLOCK *)(dst_data + dst_pitch * (height - y - 4));

            const BLOCK *s0 = (const BLOCK *)(src_data + src_pitch * y);
            const BLOCK *s1 = (const BLOCK *)(src_data + src_pitch * (y + 1));
            const BLOCK *s2 = (const BLOCK *)(src_data + src_pitch * (y + 2));
            const BLOCK *s3 = (const BLOCK *)(src_data + src_pitch * (y + 3));

            x = 0;
            for (; x < w_align4; x += 4)
            {
                MI_S32 sx = width - x - 1;
                d0[x]     = s0[sx];
                d0[x + 1] = s0[sx - 1];
                d0[x + 2] = s0[sx - 2];
                d0[x + 3] = s0[sx - 3];
                d1[x]     = s1[sx];
                d1[x + 1] = s1[sx - 1];
                d1[x + 2] = s1[sx - 2];
                d1[x + 3] = s1[sx - 3];
                d2[x]     = s2[sx];
                d2[x + 1] = s2[sx - 1];
                d2[x + 2] = s2[sx - 2];
                d2[x + 3] = s2[sx - 3];
                d3[x]     = s3[sx];
                d3[x + 1] = s3[sx - 1];
                d3[x + 2] = s3[sx - 2];
                d3[x + 3] = s3[sx - 3];
            }

            for (; x < w; x++)
            {
                MI_S32 sx = width - x - 1;
                d0[x] = s0[sx];
                d1[x] = s1[sx];
                d2[x] = s2[sx];
                d3[x] = s3[sx];
            }
        }

        for (; y < h; y++)
        {
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * (height - y - 1));
            const BLOCK *s0 = (const BLOCK *)(src_data + src_pitch * y);

            x = 0;
            for (; x < w_align4; x += 4)
            {
                MI_S32 sx = width - x - 1;
                d0[x]     = s0[sx];
                d0[x + 1] = s0[sx - 1];
                d0[x + 2] = s0[sx - 2];
                d0[x + 3] = s0[sx - 3];
            }

            for (; x < w; x++)
            {
                d0[x] = s0[width - x - 1];
            }
        }
        return Status::OK;
    }
};

template <typename Tp, MI_S32 C>
struct RotateNoneFunctor<Tp, RotateType::ROTATE_270, C>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;

    AURA_ALWAYS_INLINE Status operator()(const Mat &src, Mat &dst, MI_S32 start_blk, MI_S32 end_blk)
    {
        using BLOCK = struct { Tp val[C]; };

        const MI_U8 *src_data = (MI_U8 *)src.GetData();
        MI_U8 *dst_data       = (MI_U8 *)dst.GetData();

        MI_U32 src_pitch = src.GetRowPitch();
        MI_U32 dst_pitch = dst.GetRowPitch();

        MI_S32 width  = dst.GetSizes().m_width;
        MI_S32 height = dst.GetSizes().m_height;

        MI_S32 start_row = start_blk * BLOCK_SIZE;
        MI_S32 end_row   = Min(end_blk * BLOCK_SIZE, height);

        MI_S32 x = 0;
        MI_S32 y = Max(static_cast<MI_S32>(0), start_row);
        MI_S32 w = width;
        MI_S32 h = Min(height, end_row);

        MI_S32 w_align4 = (w & (-4));
        MI_S32 h_align4 = (h & (-4));

        for (; y < h_align4; y += 4)
        {
            MI_S32 sy = height - y - 1;
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * y);
            BLOCK *d1 = (BLOCK *)(dst_data + dst_pitch * (y + 1));
            BLOCK *d2 = (BLOCK *)(dst_data + dst_pitch * (y + 2));
            BLOCK *d3 = (BLOCK *)(dst_data + dst_pitch * (y + 3));

            x = 0;
            for (; x < w_align4; x += 4)
            {
                const BLOCK *s0 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * x);
                const BLOCK *s1 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 3));

                d0[x]     = s0[0];
                d0[x + 1] = s1[0];
                d0[x + 2] = s2[0];
                d0[x + 3] = s3[0];

                d1[x]     = s0[-1];
                d1[x + 1] = s1[-1];
                d1[x + 2] = s2[-1];
                d1[x + 3] = s3[-1];

                d2[x]     = s0[-2];
                d2[x + 1] = s1[-2];
                d2[x + 2] = s2[-2];
                d2[x + 3] = s3[-2];

                d3[x]     = s0[-3];
                d3[x + 1] = s1[-3];
                d3[x + 2] = s2[-3];
                d3[x + 3] = s3[-3];
            }

            for (; x < w; x++)
            {
                const BLOCK *s0 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * x);

                d0[x] = s0[0];
                d1[x] = s0[-1];
                d2[x] = s0[-2];
                d3[x] = s0[-3];
            }
        }

        for (; y < h; y++)
        {
            MI_S32 sy = height - y - 1;
            BLOCK *d0 = (BLOCK *)(dst_data + dst_pitch * y);

            x = 0;
            for (; x < w_align4; x += 4)
            {
                const BLOCK *s0 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * x);
                const BLOCK *s1 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * (x + 3));

                d0[x]     = s0[0];
                d0[x + 1] = s1[0];
                d0[x + 2] = s2[0];
                d0[x + 3] = s3[0];
            }

            for (; x < w; x++)
            {
                const BLOCK *s0 = (const BLOCK *)(src_data + sy * sizeof(BLOCK) + src_pitch * x);

                d0[x] = s0[0];
            }
        }
        return Status::OK;
    }
};

class RotateImpl : public OpImpl
{
public:
    RotateImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, RotateType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
    RotateType   m_type;
};

class RotateNone : public RotateImpl
{
public:
    RotateNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, RotateType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)

// RotateTraits
template <typename Tp, RotateType, MI_S32> struct RotateNeonFunctor;

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_90, 1>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_90, 2>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U16;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_90, 3>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_90, 4>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;
    using SType = MI_U32;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_180, 1>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_180, 2>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U16;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_180, 3>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_180, 4>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;
    using SType = MI_U32;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_270, 1>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_270, 2>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U16;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_270, 3>
{
    constexpr static MI_S32 BLOCK_SIZE = 8;
    using SType = MI_U8;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

template <> struct RotateNeonFunctor<MI_U8, RotateType::ROTATE_270, 4>
{
    constexpr static MI_S32 BLOCK_SIZE = 4;
    using SType = MI_U32;
    AURA_VOID operator()(SType*, SType*, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32, MI_U32);
};

class RotateNeon : public RotateImpl
{
public:
    RotateNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, RotateType type) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class RotateCL : public RotateImpl
{
public:
    RotateCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, RotateType type) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 ochannel, RotateType type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem    m_cl_src;
    CLMem    m_cl_dst;
    MI_S32   m_elem_counts;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_ROTATE_IMPL_HPP__
