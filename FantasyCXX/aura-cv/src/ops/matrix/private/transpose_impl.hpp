#ifndef AURA_OPS_MATRIX_TRANSPOSE_IMPL_HPP__
#define AURA_OPS_MATRIX_TRANSPOSE_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{
template <typename Tp, MI_S32 C>
struct TransposeNoneFunctor
{
    constexpr static MI_S32 BLOCK_SIZE = 4;

    AURA_VOID operator()(const MI_U8 *src_data, MI_U8 *dst_data, MI_S32 x, MI_S32 y, MI_S32 w, MI_S32 h,
                       MI_S32 src_pitch, MI_S32 dst_pitch)
    {
        using BLOCK = struct { Tp val[C]; };

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
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * x);
                const BLOCK *s1 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 3));

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
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + x * src_pitch);

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
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * x);
                const BLOCK *s1 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 1));
                const BLOCK *s2 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 2));
                const BLOCK *s3 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + src_pitch * (x + 3));

                d0[x]     = s0[0];
                d0[x + 1] = s1[0];
                d0[x + 2] = s2[0];
                d0[x + 3] = s3[0];
            }
            for (; x < w; x++)
            {
                const BLOCK *s0 = (const BLOCK *)(src_data + y * sizeof(BLOCK) + x * src_pitch);

                d0[x] = s0[0];
            }
        }
    }

    Status operator()(const Mat &src, Mat &dst, MI_S32 start_blk, MI_S32 end_blk)
    {
        MI_S32 start_row = start_blk * 16;
        MI_S32 end_row   = Min(end_blk * 16, dst.GetSizes().m_height);
        const MI_U8 *src_data = (MI_U8 *)src.GetData();
        MI_U8 *dst_data       = (MI_U8 *)dst.GetData();

        MI_U32 src_pitch = src.GetRowPitch();
        MI_U32 dst_pitch = dst.GetRowPitch();

        MI_S32 width  = dst.GetSizes().m_width;
        MI_S32 height = dst.GetSizes().m_height;

        MI_S32 x = 0;
        MI_S32 y = Max(static_cast<MI_S32>(0), start_row);
        MI_S32 w = width;
        MI_S32 h = Min(height, end_row);

        operator()(src_data, dst_data, x, y, w, h, src_pitch, dst_pitch);

        return Status::OK;
    }
};

class TransposeImpl : public OpImpl
{
public:
    TransposeImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
};

class TransposeNone : public TransposeImpl
{
public:
    TransposeNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
Status TransposeU8Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status TransposeU16Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status TransposeU32Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

class TransposeNeon : public TransposeImpl
{
public:
    TransposeNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class TransposeCL : public TransposeImpl
{
public:
    TransposeCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 ochannel);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    MI_S32 m_elem_counts;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_TRANSPOSE_IMPL_HPP__
