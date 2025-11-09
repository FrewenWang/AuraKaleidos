#ifndef AURA_RUNTIME_OPENCL_CL_RUNTIME_IMPL_HPP__
#define AURA_RUNTIME_OPENCL_CL_RUNTIME_IMPL_HPP__

#include "aura/runtime/opencl/cl_runtime.hpp"
#include "cl_program_container.hpp"

namespace aura
{

Status FindFirstGPU(std::shared_ptr<cl::Platform> &cl_platform, std::shared_ptr<cl::Device> &cl_device);

#if defined(CL_VERSION_2_0)
class AllocatorSVM : public Allocator
{
public:
    AllocatorSVM(CLRuntime *rt);
    ~AllocatorSVM(DT_VOID);

    Buffer Allocate(DT_S64 size, DT_S32 align = 0);
    DT_VOID Free(Buffer &);

    Status Map(const Buffer &);
    Status Unmap(const Buffer &);

    DT_BOOL IsValid() const;

private:
    DT_BOOL       m_valid;
    DT_BOOL       m_fine_grain;
    std::shared_ptr<cl::Context>      m_cl_context;
    std::shared_ptr<cl::CommandQueue> m_cl_command_queue;
};
#endif

class MobileCLRuntime : public CLRuntime
{
public:
    MobileCLRuntime(Context *ctx,
                    std::shared_ptr<cl::Platform> &cl_platform,
                    std::shared_ptr<cl::Device> &cl_device,
                    const CLEngineConfig &cl_conf);

    ~MobileCLRuntime();

    std::shared_ptr<cl::Program> GetCLProgram(const std::string &program_name,
                                              const std::string &kernel_src = std::string(),
                                              const std::string &build_options = std::string()) const override;

    std::shared_ptr<cl::Platform>     GetPlatform() override;
    std::shared_ptr<cl::Device>       GetDevice() override;
    std::shared_ptr<cl::Context>      GetContext() override;
    std::shared_ptr<cl::CommandQueue> GetCommandQueue() override;

    DT_BOOL IsValid() const override;
    DT_BOOL IsNonUniformWorkgroupsSupported() const override;

    cl::Buffer*  CreateCLBuffer(cl_mem_flags cl_flags, size_t size) override;
    cl::Iaura2D* CreateCLIaura2D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height) override;
    cl::Iaura3D* CreateCLIaura3D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height, size_t depth) override;

#if defined(CL_VERSION_2_0)
    cl::Buffer* InitCLBufferWithSvm(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method);
    cl::Iaura2D* InitCLIaura2DWithSvm(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &fmt, size_t width,
                                      size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method);
#endif

    DT_S32 GetCLAddrAlignSize() const override;
    DT_BOOL IsMemShareSupported() const override;
    DT_S32 GetCLLengthAlignSize() const override;
    DT_S32 GetCLSliceAlignSize(const cl_iaura_format &cl_fmt, size_t width, size_t height) const override;

    std::string GetCLMaxConstantSizeString(DT_S32 n) override;
    cl::NDRange GetCLDefaultLocalSize(DT_U32 max_group_size, cl::NDRange &global_size) override;

    GpuInfo GetGpuInfo() const override;

    Status CheckKernelWorkSize();

protected:
    virtual Status CreateCLCommandQueue();
    virtual Status CreateCLProgram(const std::string &extenal_version);
    Status Initialize() override;

    Status CreatePrecompiledCLProgram(const std::string &file_path, const std::string &prefix) override;
    DT_VOID DeleteCLMem(DT_VOID **ptr) override;
    DT_VOID RegisterSvmAllocator();

protected:
    DT_BOOL                             m_valid;
    Context                             *m_ctx;
    std::shared_ptr<CLEngineConfig>     m_cl_conf;
    std::shared_ptr<cl::Platform>       m_cl_platform;
    std::shared_ptr<cl::Device>         m_cl_device;
    std::shared_ptr<cl::Context>        m_cl_context;
    std::shared_ptr<cl::CommandQueue>   m_cl_command_queue;
    std::shared_ptr<CLProgramContainer> m_cl_program_container;

    std::unordered_map<DT_UPTR_T, DT_S32> m_cl_membk;
    DT_BOOL m_is_fine_grain;
    DT_S32 m_iaura_pitch_align;
    DT_S32 m_cache_line_size;
    std::string m_cl_extensions_str;
    DT_F32 m_cl_version;
    cl_bool m_is_support_non_uniform_workgroups;
    DT_BOOL m_3d_iaura_write_support;
    std::mutex m_cl_membk_mutex;
};

enum class AdrenoCLIonType
{
    CL_ION_INVALID        = 0,
    CL_ION_NOT_SUPPOSE,
    CL_ION_CACHED,
    CL_ION_UNCACHED,
};
class AdrenoCLRuntime : public MobileCLRuntime
{
public:
    AdrenoCLRuntime(Context *ctx,
                    std::shared_ptr<cl::Platform> &cl_platform,
                    std::shared_ptr<cl::Device> &cl_device,
                    const CLEngineConfig &cl_conf);

    ~AdrenoCLRuntime()
    {}

    DT_S32 GetIauraRowPitch(DT_S32 width, DT_S32 height, cl_iaura_format cl_fmt) const;
    DT_S32 GetIauraSlicePitch(DT_S32 width, DT_S32 height, cl_iaura_format cl_fmt) const;
    DT_S32 GetCLAddrAlignSize() const override;
    DT_BOOL IsMemShareSupported() const override;
    DT_S32 GetCLLengthAlignSize() const override;
    DT_S32 GetCLSliceAlignSize(const cl_iaura_format &cl_fmt, size_t width, size_t height) const override;

    cl::Buffer*  InitCLBuffer(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method) override;
    cl::Iaura2D* InitCLIaura2D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                               size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method) override;
    cl::Iaura3D* InitCLIaura3D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                               size_t height, size_t depth, size_t row_pitch, size_t slice_pitch, CLMemSyncMethod &cl_sync_method) override;

    std::string GetCLMaxConstantSizeString(DT_S32 n) override;

    GpuInfo GetGpuInfo() const override;

protected:
    Status Initialize() override;
    Status CreateCLProgram(const std::string &extenal_version) override;

private:
    DT_U32  m_qcom_ext_mem_padding;
    DT_U32  m_qcom_page_size;
    DT_U32  m_qcom_host_cache_policy;
    AdrenoCLIonType m_cl_ion_type;

private:
    std::vector<cl_context_properties> ParseContextProps(CLPerfLevel cl_perf_level, CLPriorityLevel cl_priority_level);
};

class MaliCLRuntime : public MobileCLRuntime
{
public:
    MaliCLRuntime(Context *ctx,
                  std::shared_ptr<cl::Platform> &cl_platform,
                  std::shared_ptr<cl::Device> &cl_device,
                  const CLEngineConfig &cl_conf);

    ~MaliCLRuntime();

    cl::Buffer* InitCLBuffer(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method)  override;
    cl::Iaura2D* InitCLIaura2D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt,
                               size_t width, size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method) override;
    cl::Iaura3D* InitCLIaura3D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt,
                               size_t width, size_t height, size_t depth, size_t row_pitch,
                               size_t slice_pitch, CLMemSyncMethod &cl_sync_method) override;

    std::string GetCLMaxConstantSizeString(DT_S32 n) override;

    DT_S32 GetCLAddrAlignSize() const override;
    DT_BOOL IsMemShareSupported() const override;
    DT_S32 GetCLLengthAlignSize() const override;

    GpuInfo GetGpuInfo() const override;

protected:
    Status Initialize() override;

    using ClImportMemoryARMFunc = cl_mem (*)(cl_context cl_context,
                                             cl_mem_flags cl_flags,
                                             const cl_import_properties_arm *cl_properties,
                                             void *memory,
                                             size_t size,
                                             cl_int *errorcode_ret);
    ClImportMemoryARMFunc m_cl_import_func = DT_NULL;

private:
    friend void CLArmPrintCB(const char *buffer, size_t len, size_t complete, void *user_data);
    Status ParseContextProps(CLPerfLevel cl_perf_level, CLPriorityLevel cl_priority_level, cl_queue_properties *queue_properties, DT_S32 ind);

    std::string m_dma_buf_heap_name;
};

std::string GetClProgramString(const std::string &name);

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_RUNTIME_IMPL_HPP__
