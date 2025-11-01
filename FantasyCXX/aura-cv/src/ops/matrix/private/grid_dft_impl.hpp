#ifndef AURA_OPS_MATRIX_GRID_DFT_IMPL_HPP__
#define AURA_OPS_MATRIX_GRID_DFT_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class GridDftImpl : public OpImpl
{
public:
    GridDftImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array *m_dst;
    MI_S32 m_grid_len;
};

class GridDftNone : public GridDftImpl
{
public:
    GridDftNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class GridDftNeon : public GridDftImpl
{
public:
    GridDftNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class GridDftCL : public GridDftImpl
{
public:
    GridDftCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 grid_len);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_param;
    Mat m_param;
    MI_S32 m_local_buffer_size;
    std::string m_profiling_string;
};
#endif

class GridIDftImpl : public OpImpl
{
public:
    GridIDftImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array *m_dst;
    MI_S32 m_grid_len;
    MI_BOOL m_with_scale;
};

class GridIDftNone : public GridIDftImpl
{
public:
    GridIDftNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class GridIDftNeon : public GridIDftImpl
{
public:
    GridIDftNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class GridIDftCL : public GridIDftImpl
{
public:
    GridIDftCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 grid_len, MI_S32 with_scale, MI_BOOL save_real_only);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_param;
    Mat m_param;
    MI_S32 m_local_buffer_size;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_GRID_DFT_IMPL_HPP__
