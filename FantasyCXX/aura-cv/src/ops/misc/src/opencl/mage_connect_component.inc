#include "cl_helper.inc"

#define V2Tp                    VTYPE(Tp, 2)
#define V4Tp                    VTYPE(Tp, 4)

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_subgroups                : enable

#if ADRENO
#  pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle        : enable
#  pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size     : enable

#  define STRIP_SIZE             (4)
#  define SUBGROUP_SIZE          (64)
#  define BITMASK_TYPE           ulong

#  if BALLOT
#    pragma OPENCL EXTENSION cl_khr_subgroup_ballot          : enable
#    define GET_BALLOT(predict)                                                                                         \
     ({                                                                                                                 \
         const uint2 mask_v2u32_curr = sub_group_ballot(predict).s01;                                                   \
         BITMASK_TYPE mask_u64 = ((BITMASK_TYPE)mask_v2u32_curr.s1 << 32) | (BITMASK_TYPE)mask_v2u32_curr.s0;           \
         mask_u64;                                                                                                      \
     })
#  else // before 8650, sub_group_ballot is unsupported, here is backup
#    define GET_BALLOT(predict)                                                                                         \
     ({                                                                                                                 \
         subg_ballots_pool[lx] = predict;                                                                               \
         sub_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_sub_group);                                                \
         BITMASK_TYPE result = 0;                                                                                       \
         __attribute__((opencl_unroll_hint(SUBGROUP_SIZE)))                                                             \
         for (int i = 0; i < SUBGROUP_SIZE; i++)                                                                        \
         {                                                                                                              \
             result |= (subg_ballots_pool[i] > 0) ? (1lu << i) : 0;                                                     \
         }                                                                                                              \
         result;                                                                                                        \
     })
#  endif // BALLOT

#else // Mali

#  define STRIP_SIZE             (4)
#  define SUBGROUP_SIZE          (16)
#  define BITMASK_TYPE           ushort

#  if BALLOT
#    define GET_BALLOT(predict)    (convert_ushort(sub_group_ballot(predict).s0 & 0xffff))
#  else // some mali gpu driver may not support sub_group_ballot, here is backup
#    define GET_BALLOT(predict)                                                                                         \
     ({                                                                                                                 \
         subg_ballots_pool[lx] = predict;                                                                               \
         sub_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_sub_group);                                                \
         BITMASK_TYPE result = 0;                                                                                       \
         __attribute__((opencl_unroll_hint(SUBGROUP_SIZE)))                                                             \
         for (int i = 0; i < SUBGROUP_SIZE; i++)                                                                        \
         {                                                                                                              \
             result |= (subg_ballots_pool[i] > 0) ? (1u << i) : 0;                                                      \
         }                                                                                                              \
         result;                                                                                                        \
     })
#  endif // BALLOT
#endif // ADRENO

#if IS_UINT(Tp)
#  define CONVERT_CCL           convert_uint
#else
#  define CONVERT_CCL           convert_int
#endif // IS_UINT(Tp)

#define START_DIS_16BITS(bit_mask, offset)           clz(convert_ushort(~convert_ushort((bit_mask << (16 - offset)) & 0xffff)))
#define END_DIS_16BITS(bit_mask, offset)             ctz(convert_ushort(~convert_ushort((bit_mask >> (offset + 1) & 0xffff))))
#define START_DIS_32BITS(bit_mask, offset)           clz(~(bit_mask << (32 - offset)))
#define END_DIS_32BITS(bit_mask, offset)             ctz(~(bit_mask >> (offset + 1)))
#define START_DIS_64BITS(bit_mask, offset)           CONVERT_CCL(clz(~(bit_mask << (64l - offset))))
#define END_DIS_64BITS(bit_mask, offset)             CONVERT_CCL(ctz(~(bit_mask >> (offset + 1l))))

#if ADRENO
#  define START_DISTANCE(bit_mask, offset)             START_DIS_64BITS(bit_mask, offset)
#else
#  define START_DISTANCE(bit_mask, offset)             START_DIS_16BITS(bit_mask, offset)
#endif // ADRENO

void Merge(volatile global Tp *table, int label_1, int label_2)
{
    while (label_1 >= 0 && label_1 != label_2 && (label_1 != (table[label_1] - 1)))
    {
        label_1 = table[label_1] - 1;
    }
    while (label_2 >= 0 && label_1 != label_2 && (label_2 != (table[label_2] - 1)))
    {
        label_2 = table[label_2] - 1;
    }

    while (label_1 >= 0 && label_2 >= 0 && label_1 != label_2)
    {
        if (label_1 < label_2)
        {
            label_1 ^= label_2;
            label_2 ^= label_1;
            label_1 ^= label_2;
        }
        int label_3 = atom_min(table + label_1, label_2 + 1) - 1;
        label_1 = select(label_3, label_2, label_1 == label_3);
    }
}