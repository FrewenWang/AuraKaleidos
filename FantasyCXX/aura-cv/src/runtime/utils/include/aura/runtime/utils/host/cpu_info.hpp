#ifndef AURA_RUNTIME_CORE_CPU_INFO_HPP__
#define AURA_RUNTIME_CORE_CPU_INFO_HPP__

#include "aura/runtime/core.h"
#include "aura/runtime/logger.h"

#include <vector>
#include <thread>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup cpuinfo Runtime Core CpuInfo
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup cpuinfo
 * @{
*/

class AURA_EXPORTS CpuInfo
{
public:
    static CpuInfo& Get()
    {
        static CpuInfo cpu_info;
        return cpu_info;
    }

    std::vector<MI_S32> GetCpuIdxs(CpuAffinity affinity)
    {
        if (CpuAffinity::BIG == affinity)
        {
            return m_big_cpu_idxs;
        }
        else if (CpuAffinity::LITTLE == affinity)
        {
            return m_little_cpu_idxs;
        }
        else
        {
            return m_all_cpu_idxs;
        }
    }

    MI_BOOL IsAtomicsSupported()
    {
        return m_atomics_valid;
    }

private:
    CpuInfo() : m_atomics_valid(MI_FALSE)
    {
        InitCpuIdxs();

        InitAtomics();
    }

    AURA_VOID InitCpuIdxs()
    {
        MI_S32 total_cores_count = Max(static_cast<MI_S32>(std::thread::hardware_concurrency()), 1);

#if defined(AURA_BUILD_ANDROID)
        Status status = Status::OK;

        MI_S64 avg_freq_khz = 0;
        std::vector<MI_S64> cpu_freq_khz;

        MI_S32 cpu_cores_count = 0;
        MI_S64 total_freq_khz = 0;
        for (MI_S32 index = 0; index < total_cores_count; index++)
        {
            MI_S32 freq_values = 0;

            MI_CHAR nama_buffer[256] = {0};
            sprintf(nama_buffer, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", index);

            FILE *fp = fopen(nama_buffer, "r");
            if (MI_NULL == fp)
            {
                status = Status::ERROR;
                AURA_PRINTE(AURA_TAG, "Get CpuInfo from /sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq failed use default result\n", index);
                break;
            }

            cpu_cores_count++;
            MI_S32 ret = fscanf(fp, "%d", &freq_values);
            if (ret != 1 && freq_values < 0)
            {
                status = Status::ERROR;
                AURA_PRINTE(AURA_TAG, "Get CpuInfo from /sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq failed use default result\n", index);
                fclose(fp);
                break;
            }
            fclose(fp);
            cpu_freq_khz.push_back(freq_values);
            total_freq_khz += freq_values;
        }

        if (status != Status::OK)
        {
            for (MI_S32 index = 0; index < total_cores_count; index++)
            {
                m_big_cpu_idxs.push_back(index);
                m_little_cpu_idxs.push_back(index);
                m_all_cpu_idxs.push_back(index);
            }
        }
        else
        {
            avg_freq_khz = total_freq_khz / cpu_cores_count;

            for (MI_S32 index = 0; index < total_cores_count; index++)
            {
                /// For cpu cores with same freq, cores are all big cores
                if (cpu_freq_khz[index] >= avg_freq_khz)
                {
                    m_big_cpu_idxs.push_back(index);
                }
                else
                {
                    m_little_cpu_idxs.push_back(index);
                }
                m_all_cpu_idxs.push_back(index);
            }
        }
#else
        for (MI_S32 index = 0; index < total_cores_count; index++)
        {
            m_big_cpu_idxs.push_back(index);
            m_little_cpu_idxs.push_back(index);
            m_all_cpu_idxs.push_back(index);
        }
#endif
    }

    AURA_VOID InitAtomics()
    {
#if defined(AURA_BUILD_ANDROID)
        MI_CHAR str_buffer[256] = {0};
        FILE *fp = fopen("/proc/cpuinfo", "rt");
        if (fp)
        {
            while (fgets(str_buffer, sizeof(str_buffer), fp))
            {
                if (strstr(str_buffer, "atomics"))
                {
                    m_atomics_valid = MI_TRUE;
                    break;
                }
            }
            fclose(fp);
        }
        else
        {
            m_atomics_valid = MI_TRUE;
        }
#else
        m_atomics_valid = MI_TRUE;
#endif
    }

private:
    std::vector<MI_S32> m_all_cpu_idxs;
    std::vector<MI_S32> m_big_cpu_idxs;
    std::vector<MI_S32> m_little_cpu_idxs;
    MI_BOOL m_atomics_valid;
};

/**
 * @}
*/

} // namespace aura

#endif // AURA_RUNTIME_CORE_CPU_INFO_HPP__