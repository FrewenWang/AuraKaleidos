#ifndef AURA_TOOLS_UNIT_TEST_TEST_UNORDERED_COMPARE_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_UNORDERED_COMPARE_HPP__

#include "aura/runtime/mat.h"

#include <vector>
#include <algorithm>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup unit_test Unit Test
 * @}
 */

namespace aura
{
/**
 * @addtogroup unit_test
 * @{
 */

/**
 * @brief Represents a position for unordered comparison.
 * 
 * @tparam Tp The type of elements in the position.
 */
template <typename Tp>
struct UnorderedCmpPos
{
    Tp val[3];      /*!< Array representing values (src, ref, diff). */
    MI_S32 pos[2];  /*!< Array representing positions (src_pos, dst_pos). */

    /**
     * @brief Default constructor for UnorderedCmpPos.
     */
    UnorderedCmpPos() : val{}, pos{}
    {}

    /**
     * @brief Constructor for UnorderedCmpPos.
     *
     * @param src Source value.
     * @param ref Reference value.
     * @param diff Difference value.
     * @param src_pos Source position.
     * @param dst_pos Destination position.
     */
    UnorderedCmpPos(Tp src, Tp ref, Tp diff, MI_S32 src_pos, MI_S32 dst_pos) : val{src, ref, diff}, pos{src_pos, dst_pos}
    {}

    /**
     * @brief Overloaded output stream operator to print UnorderedCmpPos.
     */
    friend std::ostream& operator<<(std::ostream &os, const UnorderedCmpPos &cmp_pos)
    {
        os << "src index: [" << cmp_pos.pos[0] << "] ref_index: [" << cmp_pos.pos[1] << "]";
        os << " src(" << cmp_pos.val[0] << ") ref(" << cmp_pos.val[1] << ") diff(" << cmp_pos.val[2] << ")";
        return os;
    }

    /**
     * @brief Converts UnorderedCmpPos to a string.
     */
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

/**
 * @brief Represents the result of unordered comparison.
 * 
 * @tparam Tp The type of elements in the result.
 */
template <typename Tp>
struct UnorderedCmpResult
{
    MI_BOOL status;                                     /*!< Status of the comparison. */
    MI_BOOL detail;                                     /*!< Detail of the comparison. */
    MI_F32 precision;                                   /*!< Precision of the comparison. */
    MI_F32 recall;                                      /*!< Recall of the comparison. */
    MI_F32 f1_score;                                    /*!< F1 score of the comparison. */
    std::vector<UnorderedCmpPos<Tp>> matched_detail;    /*!< Vector of matched details. */
    std::vector<Tp> unmatched_detail[2];                /*!< Array of unmatched details for source and reference. */

    /**
     * @brief Default constructor for UnorderedCmpResult.
     */
    UnorderedCmpResult() : status(MI_FALSE), detail(MI_FALSE), precision(0.f), recall(0.f), f1_score(0.f)
    {}

    /**
     * @brief Overloaded output stream operator to print UnorderedCmpResult.
     */
    friend std::ostream& operator<<(std::ostream &os, const UnorderedCmpResult &result)
    {
        MI_CHAR buffer[1024] = {0};

        snprintf(buffer, sizeof(buffer), "precision(%.4f%%) recall(%.4f%%) f1_score(%.4f%%)", 100.f * result.precision, 100.f * result.recall, 100.f * result.f1_score);

        os << buffer;
        return os;
    }

    /**
     * @brief Converts UnorderedCmpResult to a string.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
};

template<typename Tp> struct Tolerate;

/**
 * @brief Template specialization for Keypoint types.
 * 
 * @tparam Tp The type of elements in the Tolerate structure.
 *
 * @param initial_value The initial value for tolerance comparison.
 */
#define KEYPOINT_TOLERATE(Tp, initial_value)                                                        \
template<> struct Tolerate<Tp>                                                                      \
{                                                                                                   \
    MI_F64   t_ratio;                                                                               \
    Tp t_thres;                                                                                     \
                                                                                                    \
    Tolerate(MI_F32 t_ratio = 1.0f, const Tp &kp = initial_value)                                   \
             : t_ratio{t_ratio}, t_thres{kp}                                                        \
    {}                                                                                              \
                                                                                                    \
    Tolerate(const Tp &src, const Tp &ref)                                                          \
    {                                                                                               \
        t_thres.m_pt.m_x   = Abs(src.m_pt.m_x   - ref.m_pt.m_x);                                    \
        t_thres.m_pt.m_y   = Abs(src.m_pt.m_y   - ref.m_pt.m_y);                                    \
        t_thres.m_size     = Abs(src.m_size     - ref.m_size);                                      \
        t_thres.m_angle    = Abs(src.m_angle    - ref.m_angle);                                     \
        t_thres.m_response = Abs(src.m_response - ref.m_response);                                  \
        t_thres.m_octave   = Abs(src.m_octave   - ref.m_octave);                                    \
        t_thres.m_class_id = Abs(src.m_class_id - ref.m_class_id);                                  \
    }                                                                                               \
                                                                                                    \
    operator Tp()                                                                                   \
    {                                                                                               \
        return t_thres;                                                                             \
    }                                                                                               \
                                                                                                    \
    MI_BOOL operator<=(const Tolerate &t)                                                         \
    {                                                                                               \
        if (t_thres.m_pt.m_x <= t.t_thres.m_pt.m_x &&                                               \
            t_thres.m_pt.m_y <= t.t_thres.m_pt.m_y &&                                               \
            t_thres.m_size <= t.t_thres.m_size &&                                                   \
            t_thres.m_angle <= t.t_thres.m_angle &&                                                 \
            t_thres.m_response <= t.t_thres.m_response &&                                           \
            t_thres.m_octave <= t.t_thres.m_octave &&                                               \
            t_thres.m_class_id <= t.t_thres.m_class_id)                                             \
        {                                                                                           \
            return MI_TRUE;                                                                         \
        }                                                                                           \
        return MI_FALSE;                                                                            \
    }                                                                                               \
};

/**
 * @brief Template specialization for Scalar types.
 * 
 * @tparam Tp The type of elements in the Tolerate structure.
 *
 * @param initial_value The initial value for tolerance comparison.
 */
#define SCALAR_TOLERATE(Tp, initial_value)                                                          \
template<> struct Tolerate<Tp>                                                                      \
{                                                                                                   \
    MI_F64 t_ratio;                                                                                 \
    Tp t_thres;                                                                                     \
                                                                                                    \
    Tolerate(MI_F32 t_ratio = 1.0f, const Tp &t_thres = initial_value)                              \
             : t_ratio{t_ratio}, t_thres{t_thres}                                                   \
    {}                                                                                              \
                                                                                                    \
    Tolerate(const Tp &src, const Tp &ref)                                                          \
    {                                                                                               \
        t_thres.m_val[0] = Abs(src.m_val[0] - ref.m_val[0]);                                        \
        t_thres.m_val[1] = Abs(src.m_val[1] - ref.m_val[1]);                                        \
        t_thres.m_val[2] = Abs(src.m_val[2] - ref.m_val[2]);                                        \
        t_thres.m_val[3] = Abs(src.m_val[3] - ref.m_val[3]);                                        \
    }                                                                                               \
                                                                                                    \
    operator Tp()                                                                                   \
    {                                                                                               \
        return t_thres;                                                                             \
    }                                                                                               \
                                                                                                    \
    MI_BOOL operator<=(const Tolerate &t)                                                         \
    {                                                                                               \
        if (t_thres.m_val[0] <= t.t_thres.m_val[0] && t_thres.m_val[1] <= t.t_thres.m_val[1]        \
            && t_thres.m_val[2] <= t.t_thres.m_val[2] && t_thres.m_val[3] <= t.t_thres.m_val[3])    \
        {                                                                                           \
            return MI_TRUE;                                                                         \
        }                                                                                           \
        return MI_FALSE;                                                                            \
    }                                                                                               \
};

/**
 * @brief Tolerance specialization for KeyPoint types.
 */
KEYPOINT_TOLERATE(KeyPoint, KeyPoint(1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f, 0, 0))

/**
 * @brief Tolerance specialization for KeyPointi types.
 */
KEYPOINT_TOLERATE(KeyPointi, KeyPointi(0, 0, 1e-5f, 1e-5f, 1e-5f, 0, 0))

/**
 * @brief Tolerance specialization for Scalar types.
 */
SCALAR_TOLERATE(Scalar, Scalar(1e-5f, 1e-5f, 1e-5f, 1e-5f))

/**
 * @brief Tolerance specialization for Scalari types.
 */
SCALAR_TOLERATE(Scalari, Scalari(0, 0, 0, 0))

/**
 * @brief L1 distance calculation for KeyPoint or KeyPointi types.
 * 
 * @tparam Tp The type of elements for L1 distance calculation.
 *
 * @param src The source value.
 * @param ref The reference value.
 * 
 * @return L1 distance between source and reference values.
 */
template <typename Tp, typename std::enable_if<std::is_same<KeyPoint, Tp>::value || std::is_same<KeyPointi, Tp>::value>::type * = MI_NULL>
struct KeyPointL1Dis
{
    /**
     * @brief Calculates L1 distance between source and reference KeyPoint values.
     * 
     * @param src The source KeyPoint.
     * @param ref The reference KeyPoint.
     * 
     * @return L1 distance between source and reference KeyPoint values.
     */
    MI_F64 operator()(const Tp &src, const Tp &ref) const
    {
        return Abs(static_cast<MI_F64>(src.m_pt.m_x - ref.m_pt.m_x)) + Abs(static_cast<MI_F64>(src.m_pt.m_y - ref.m_pt.m_y));
    }

    /**
     * @brief Get a string representation of the KeyPointL1Dis type.
     */
    static std::string ToString()
    {
        return "KeyPointL1Dis";
    }
};

/**
 * @brief Callable structure for calculating L2 distance between KeyPoint or KeyPointi values.
 * 
 * @tparam Tp The type of elements for L2 distance calculation.
 */
template <typename Tp, typename std::enable_if<std::is_same<KeyPoint, Tp>::value || std::is_same<KeyPointi, Tp>::value>::type * = MI_NULL>
struct KeyPointL2Dis
{
    /**
     * @brief Calculates L2 distance between source and reference KeyPoint values.
     * 
     * @param src The source KeyPoint.
     * @param ref The reference KeyPoint.
     * 
     * @return L2 distance between source and reference KeyPoint values.
     */
    MI_F64 operator()(const Tp &src, const Tp &ref) const
    {
        return Pow(static_cast<MI_F64>(src.m_pt.m_x - ref.m_pt.m_x), 2) + Pow(static_cast<MI_F64>(src.m_pt.m_y - ref.m_pt.m_y), 2);
    }

    /**
     * @brief Get a string representation of the KeyPointL2Dis type.
     */
    static std::string ToString()
    {
        return "KeyPointL2Dis";
    }
};

/**
 * @brief structure for calculating L1 distance between Scalar or Scalari values.
 * 
 * @tparam Tp The type of elements for L1 distance calculation.
 */
template <typename Tp, typename std::enable_if<std::is_same<Scalar, Tp>::value || std::is_same<Scalari, Tp>::value>::type * = MI_NULL>
struct ScalarL1Dis
{
    /**
     * @brief Calculates L1 distance between source and reference Scalar values.
     * 
     * @param src The source Scalar.
     * @param ref The reference Scalar.
     * 
     * @return L1 distance between source and reference Scalar values.
     */
    MI_F64 operator()(const Tp &src, const Tp &ref) const
    {
        return Abs(static_cast<MI_F64>(src.m_val[0] - ref.m_val[0])) + Abs(static_cast<MI_F64>(src.m_val[1] - ref.m_val[1]))
               + Abs(static_cast<MI_F64>(src.m_val[2] - ref.m_val[2])) + Abs(static_cast<MI_F64>(src.m_val[3] - ref.m_val[3]));
    }

    /**
     * @brief Get a string representation of the ScalarL1Dis type.
     */
    static std::string ToString()
    {
        return "ScalarL1Dis";
    }
};

/**
 * @brief structure for calculating L2 distance between Scalar or Scalari values.
 * 
 * @tparam Tp The type of elements for L2 distance calculation.
 */
template <typename Tp, typename std::enable_if<std::is_same<Scalar, Tp>::value || std::is_same<Scalari, Tp>::value>::type * = MI_NULL>
struct ScalarL2Dis
{
    /**
     * @brief Calculates L2 distance between source and reference Scalar values.
     * 
     * @param src The source Scalar.
     * @param ref The reference Scalar.
     * 
     * @return L2 distance between source and reference Scalar values.
     */
    MI_F64 operator()(const Tp &src, const Tp &ref) const
    {
        return Pow(static_cast<MI_F64>(src.m_val[0] - ref.m_val[0]), 2) + Pow(static_cast<MI_F64>(src.m_val[1] - ref.m_val[1]), 2)
               + Pow(static_cast<MI_F64>(src.m_val[2] - ref.m_val[2]), 2) + Pow(static_cast<MI_F64>(src.m_val[3] - ref.m_val[3]), 2);
    }

    /**
     * @brief Get a string representation of the ScalarL2Dis type.
     */
    static std::string ToString()
    {
        return "ScalarL2Dis";
    }
};

/**
 * @brief Tag type for indicating L1 norm.
 */
struct L1NormTag {};


/**
 * @brief Tag type for indicating L2 norm.
 */
struct L2NormTag {};

/**
 * @brief Traits class for distance functors with different norm types.
 * 
 * @tparam Tp The type of elements.
 *
 * @tparam Tag The norm tag type (L1NormTag or L2NormTag).
 */
template<typename Tp, typename Tag> struct DisFunctorTraits;

/**
 * @brief Specialization for KeyPoint and L1 norm.
 */
template<> struct DisFunctorTraits<KeyPoint, L1NormTag>  { using type = KeyPointL1Dis<KeyPoint>; };

/**
 * @brief Specialization for KeyPoint and L2 norm.
 */
template<> struct DisFunctorTraits<KeyPoint, L2NormTag>  { using type = KeyPointL2Dis<KeyPoint>; };

/**
 * @brief Specialization for KeyPointi and L1 norm.
 */
template<> struct DisFunctorTraits<KeyPointi, L1NormTag> { using type = KeyPointL1Dis<KeyPointi>; };

/**
 * @brief Specialization for KeyPointi and L2 norm.
 */
template<> struct DisFunctorTraits<KeyPointi, L2NormTag> { using type = KeyPointL2Dis<KeyPointi>; };

/**
 * @brief Specialization for Scalar and L1 norm.
 */
template<> struct DisFunctorTraits<Scalar, L1NormTag>    { using type = ScalarL1Dis<Scalar>; };

/**
 * @brief Specialization for Scalar and L2 norm.
 */
template<> struct DisFunctorTraits<Scalar, L2NormTag>    { using type = ScalarL2Dis<Scalar>; };

/**
 * @brief Specialization for Scalari and L1 norm.
 */
template<> struct DisFunctorTraits<Scalari, L1NormTag>   { using type = ScalarL1Dis<Scalari>; };

/**
 * @brief Specialization for Scalari and L2 norm.
 */
template<> struct DisFunctorTraits<Scalari, L2NormTag>   { using type = ScalarL2Dis<Scalari>; };

/**
 * @brief Performs unordered comparison between two vectors of elements.
 *
 * @tparam Tp The type of elements in the vectors.
 * @tparam Tag The norm tag type (L1NormTag or L2NormTag).
 *
 * @param ctx The pointer to the Context object.
 * @param src The source vector.
 * @param ref The reference vector.
 * @param result Reference to an UnorderedCmpResult to store the comparison result.
 * @param tolerate Tolerance values for comparison (default is Tolerate<Tp>()).
 *
 * @return The status of the comparison (OK or ERROR).
 */
template <typename Tp, typename Tag = L2NormTag>
Status UnorderedCompare(Context *ctx, const std::vector<Tp> &src, const std::vector<Tp> &ref,
                        UnorderedCmpResult<Tp> &result, const Tolerate<Tp> &tolerate = Tolerate<Tp>())
{
    result.status = MI_FALSE;

    using DisFunc = typename DisFunctorTraits<Tp, Tag>::type;
    DisFunc dis_func;

    auto copy_func = [](const std::vector<Tp> &input, std::vector<std::pair<Tp, MI_BOOL>> &output)
    {
        output = std::vector<std::pair<Tp, MI_BOOL>>(input.size());
        for (size_t i = 0; i < output.size(); ++i)
        {
            output[i].first  = input[i];
            output[i].second = MI_TRUE;
        }
    };

    auto copy_func_inv = [&](const std::vector<std::pair<Tp, MI_BOOL>> &input, MI_S32 cp_id)
    {
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (input[i].second)
            {
                result.unmatched_detail[cp_id].push_back(input[i].first);
            }
        }
    };

    std::vector<std::pair<Tp, MI_BOOL>> srcs, refs;
    copy_func(src, srcs);
    copy_func(ref, refs);

    MI_S32 matched_num = 0;
    MI_S32 src_pos = 0, ref_pos = 0;
    while (MI_TRUE)
    {
        MI_S32 i = 0, j = 0;
        std::pair<Tp, MI_BOOL> *pt_src = MI_NULL;
        std::pair<Tp, MI_BOOL> *pt_ref = MI_NULL;
        MI_F64 min_dis = std::numeric_limits<MI_F64>::max();

        for (auto it_src = srcs.begin(); it_src != srcs.end(); ++it_src, i++)
        {
            if (!it_src->second)
            {
                continue;
            }

            j = 0;

            for (auto it_ref = refs.begin(); it_ref != refs.end(); ++it_ref, j++)
            {
                if (!it_ref->second)
                {
                    continue;
                }

                MI_F64 tmp_dis = dis_func(it_src->first, it_ref->first);

                if (tmp_dis < min_dis)
                {
                    min_dis = tmp_dis;
                    pt_src = &*it_src;
                    pt_ref = &*it_ref;
                    src_pos = i;
                    ref_pos = j;
                }
            }
        }

        if (std::numeric_limits<MI_F64>::max() == min_dis)
        {
            break;
        }

        Tolerate<Tp> kpt_tmp(pt_src->first, pt_ref->first);
        if (kpt_tmp <= tolerate)
        {
            if (result.detail)
            {
                result.matched_detail.emplace_back(pt_src->first, pt_ref->first, kpt_tmp, src_pos, ref_pos);
            }
            ++matched_num;
            pt_src->second = MI_FALSE;
            pt_ref->second = MI_FALSE;
        }
        else
        {
            break;
        }
    }

    if (result.detail)
    {
        copy_func_inv(srcs, 0);
        copy_func_inv(refs, 1);
    }

    result.precision = static_cast<MI_F32>(matched_num + 1e-6f) / (src.size() + 1e-6f);
    result.recall    = static_cast<MI_F32>(matched_num + 1e-6f) / (ref.size() + 1e-6f);
    result.f1_score  = 2 * (result.precision * result.recall) / (result.precision + result.recall);

    if (result.f1_score < tolerate.t_ratio)
    {
        result.status = MI_FALSE;
    }
    else
    {
        result.status = MI_TRUE;
    }

    MI_S32 num_tp = matched_num;
    MI_S32 num_fp = src.size() - matched_num;
    MI_S32 num_fn = ref.size() - matched_num;
    AURA_LOGI(ctx, AURA_TAG, "**********Compare error distribution:**********\n");
    AURA_LOGI(ctx, AURA_TAG, "total num: src: %lu ref: %lu\n", src.size(), ref.size());
    AURA_LOGI(ctx, AURA_TAG, "true positive num: %d, false positive num: %d, false negtive num: %d\n", num_tp, num_fp, num_fn);
    AURA_LOGI(ctx, AURA_TAG, "precision: %.4f%%, recall: %.4f%%, f1_score: %.4f%%\n", 100.f * result.precision, 100.f * result.recall, 100.f * result.f1_score);

    return Status::OK;
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_UNORDERED_COMPARE_HPP__
