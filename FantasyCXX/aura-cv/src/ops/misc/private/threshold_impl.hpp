/** @brief      : threshold impl header for aura
 *  @file       : threshold_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_THRESHOLD_IMPL_HPP__
#define AURA_OPS_MISC_THRESHOLD_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class ThresholdImpl : public OpImpl
{
public:
    ThresholdImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    Status ReCalcThresh(DT_S32 &thresh);

protected:
    DT_F32       m_thresh;
    DT_F32       m_max_val;
    DT_S32       m_type;

    const Array *m_src;
    Array       *m_dst;
};

class ThresholdNone : public ThresholdImpl
{
public:
    ThresholdNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class ThresholdNeon : public ThresholdImpl
{
public:
    ThresholdNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class ThresholdHvx : public ThresholdImpl
{
public:
    ThresholdHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using ThresholdInParam = HexagonRpcParamType<Mat, Mat, DT_F32, DT_F32, DT_S32>;
#  define AURA_OPS_MISC_THRESHOLD_OP_NAME          "Threshold"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

}

#endif // AURA_OPS_MISC_THRESHOLD_IMPL_HPP__