/** @brief      : dct impl header for aura
 *  @file       : dct_impl.hpp
 *  @author     : zhaojianguo1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : sep. 14, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MATRIX_DCT_IMPL_HPP__
#define AURA_OPS_MATRIX_DCT_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

#include <complex>

namespace aura
{

template <DT_U8 IS_INVERSE>
AURA_ALWAYS_INLINE DT_VOID GetDctExpTable(std::complex<DT_F32> *exp_table, DT_S32 n)
{
    DT_F32 pi_pe = AURA_PI / (n * 2);

    if (!IS_INVERSE)
    {
        for (DT_S32 i = 0; i < n; ++i)
        {
            DT_F32 theta = pi_pe * i;
            exp_table[i].real(Cos(theta));
            exp_table[i].imag(Sin(theta));
        }
    }
    else
    {
        for (DT_S32 i = 0; i < n; ++i)
        {
            DT_F32 theta = pi_pe * i;
            exp_table[i].real(Cos(theta));
            exp_table[i].imag(-Sin(theta));
        }
    }
}

class DctImpl: public OpImpl
{
public:
    DctImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array *> GetInputArrays() const override;

    std::vector<const Array *> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
};

class DctNone : public DctImpl
{
public:
    DctNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class DctNeon : public DctImpl
{
public:
    DctNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif

class IDctImpl: public OpImpl
{
public:
    IDctImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    Status Initialize() override;

    Status DeInitialize() override;

    std::vector<const Array *> GetInputArrays() const override;

    std::vector<const Array *> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
    Mat         m_mid;
};

class IDctNone : public IDctImpl
{
public:
    IDctNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class IDctNeon : public IDctImpl
{
public:
    IDctNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_DCT_IMPL_HPP__
