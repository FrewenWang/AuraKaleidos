/** @brief      : mipi impl for aura
 *  @file       : mipi_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_MIPI_IMPL_HPP__
#define AURA_OPS_MISC_MIPI_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class MipiPackImpl : public OpImpl
{
public:
    MipiPackImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
};

class MipiPackNone : public MipiPackImpl
{
public:
    MipiPackNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class MipiPackNeon : public MipiPackImpl
{
public:
    MipiPackNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class MipiPackHvx : public MipiPackImpl
{
public:
    MipiPackHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using MipiPackInParam = HexagonRpcParamType<Mat, Mat>;
#  define AURA_OPS_MISC_MIPIPACK_OP_NAME          "MipiPack"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

class MipiUnPackImpl : public OpImpl
{
public:
    MipiUnPackImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
};

class MipiUnPackNone : public MipiUnPackImpl
{
public:
    MipiUnPackNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class MipiUnPackNeon : public MipiUnPackImpl
{
public:
    MipiUnPackNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class MipiUnPackHvx : public MipiUnPackImpl
{
public:
    MipiUnPackHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using MipiUnPackInParam = HexagonRpcParamType<Mat, Mat>;

#  define AURA_OPS_MISC_MIPIUNPACK_OP_NAME        "MipiUnPack"
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_MISC_MIPI_IMPL_HPP__