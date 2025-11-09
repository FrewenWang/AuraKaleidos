#ifndef AURA_OPS_MATRIX_FLIP_IMPL_HPP__
#define AURA_OPS_MATRIX_FLIP_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class MakeBorderImpl : public OpImpl
{
public:
    MakeBorderImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom,
                           DT_S32 left, DT_S32 right, BorderType type,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32 m_top;
    DT_S32 m_bottom;
    DT_S32 m_left;
    DT_S32 m_right;
    BorderType m_type;
    Scalar m_border_value;

    const Array *m_src;
    Array       *m_dst;
};

class MakeBorderNone : public MakeBorderImpl
{
public:
    MakeBorderNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom,
                   DT_S32 left, DT_S32 right, BorderType type,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C, typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE DT_VOID MakeBorderOneRow(const Mat &src, const DT_S32 idx_row, const DT_S32 width, const DT_S32 ksize,
                                     Tp *src_row_border, const Scalar &border_value)
{
    AURA_UNUSED(border_value);

    const Sizes3 sz     = src.GetSizes();
    const DT_S32 height = sz.m_height;
    const DT_S32 ksh    = ksize / 2;
    DT_S32 idx_row_eff  = Min(Max(idx_row, (DT_S32)0), height - 1);

    const Tp *src_row = src.Ptr<Tp>(idx_row_eff);

    for (DT_S32 x = 0; x < ksh; x++)
    {
        for (DT_S32 c = 0; c < C; c++)
        {
            *src_row_border++ = src_row[c];
        }
    }

    memcpy(src_row_border, src_row, width * C * sizeof(Tp));
    src_row_border += width * C;

    for (DT_S32 x = 0; x < ksh; x++)
    {
        for (DT_S32 c = 0; c < C; c++)
        {
            *src_row_border++ = src_row[(width - 1) * C + c];
        }
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C, typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE DT_VOID MakeBorderOneRow(const Mat &src, const DT_S32 idx_row, const DT_S32 width, const DT_S32 ksize,
                                     Tp *src_row_border, const Scalar &border_value)
{
    const Sizes3 sz     = src.GetSizes();
    const DT_S32 height = sz.m_height;
    const DT_S32 ksh    = ksize / 2;

    Tp border[C];
    for (DT_S32 c = 0; c < C; c++)
    {
        border[c] = static_cast<Tp>(border_value.m_val[c]);
    }

    if (idx_row < 0 || idx_row > height - 1)
    {
        for (DT_S32 x = 0; x < width + ksize - 1; x++)
        {
            for (DT_S32 c = 0; c < C; c++)
            {
                *src_row_border++ = border[c];
            }
        }
    }
    else
    {
        const Tp *src_row = src.Ptr<Tp>(idx_row);

        for (DT_S32 x = 0; x < ksh; x++)
        {
            for (DT_S32 c = 0; c < C; c++)
            {
                *src_row_border++ = border[c];
            }
        }

        memcpy(src_row_border, src_row, width * C * sizeof(Tp));
        src_row_border += width * C;

        for (DT_S32 x = 0; x < ksh; x++)
        {
            for (DT_S32 c = 0; c < C; c++)
            {
                *src_row_border++ = border[c];
            }
        }
    }
}

template <typename Tp, BorderType BORDER_TYPE, DT_S32 C, typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE DT_VOID MakeBorderOneRow(const Mat &src, const DT_S32 idx_row, const DT_S32 width, const DT_S32 ksize,
                                     Tp *src_row_border, const Scalar &border_value)
{
    AURA_UNUSED(border_value);

    const Sizes3 sz     = src.GetSizes();
    const DT_S32 height = sz.m_height;
    const DT_S32 ksh    = ksize / 2;
    DT_S32 idx_row_eff  = idx_row < 0 ? -idx_row : idx_row;
    idx_row_eff         = (idx_row_eff > height - 1) ? (2 * (height - 1) - idx_row_eff) : idx_row_eff;

    const Tp *src_row = src.Ptr<Tp>(idx_row_eff);

    for (DT_S32 x = ksh; x > 0; x--)
    {
        for (DT_S32 c = 0; c < C; c++)
        {
            *src_row_border++ = src_row[x * C + c];
        }
    }

    memcpy(src_row_border, src_row, width * C * sizeof(Tp));
    src_row_border += width * C;

    for (DT_S32 x = width; x > width - ksh; x--)
    {
        for (DT_S32 c = 0; c < C; c++)
        {
            *src_row_border++ = src_row[(x - 2) * C + c];
        }
    }
}

} // namespace aura

#endif // AURA_OPS_MATRIX_FLIP_IMPL_HPP__
