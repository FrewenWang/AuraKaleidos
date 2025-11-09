/** @brief      : calc hist impl for aura
 *  @file       : calc_hist_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Mar. 20, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HIST_CALCHIST_IMPL_HPP__
#define AURA_OPS_HIST_CALCHIST_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class CalcHistImpl : public OpImpl
{
public:
    CalcHistImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, DT_S32 channel, std::vector<DT_U32> &hist, DT_S32 hist_size, 
                           const Scalar &ranges, const Array *mask = NULL, DT_BOOL accumulate = DT_FALSE);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32   m_channel;
    DT_S32   m_hist_size;
    Scalar   m_ranges;
    DT_BOOL  m_accumulate;

    const Array *m_src;
    const Array *m_mask;
    std::vector<DT_U32> *m_hist;
};

class CalcHistNone : public CalcHistImpl
{
public:
    CalcHistNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, DT_S32 channel, std::vector<DT_U32> &hist, DT_S32 hist_size, 
                   const Scalar &ranges, const Array *mask = NULL, DT_BOOL accumulate = DT_FALSE) override;

    Status Run() override;
};

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class CalcHistHvx : public CalcHistImpl
{
public:
    CalcHistHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, DT_S32 channel, std::vector<DT_U32> &hist, DT_S32 hist_size, 
                   const Scalar &ranges, const Array *mask = NULL, DT_BOOL accumulate = DT_FALSE) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using CalcHistInParam  = HexagonRpcParamType<Mat, DT_S32, std::vector<DT_U32>, DT_S32, Scalar, Mat, DT_BOOL>;
using CalcHistOutParam = HexagonRpcParamType<std::vector<DT_U32>>;
#  define AURA_OPS_HIST_CALCHIST_OP_NAME          "CalcHist"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_HIST_CALCHIST_IMPL_HPP__