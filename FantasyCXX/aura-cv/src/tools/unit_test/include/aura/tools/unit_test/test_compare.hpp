#ifndef AURA_TOOLS_UNIT_TEST_TEST_COMPARE_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_COMPARE_HPP__

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
 * @brief structure for absolute difference comparison.
 */
struct AURA_EXPORTS AbsDiff
{
    /**
     * @brief Calculates the absolute difference between two values.
     *
     * @param val0 First value.
     * @param val1 Second value.
     *
     * @return Absolute difference between val0 and val1.
     */
    DT_F64 operator()(const DT_F64 val0, const DT_F64 val1) const
    {
        return Abs(val0 - val1);
    }

    /**
     * @brief Gets a string representation of the comparison method.
     */
    static std::string ToString()
    {
        return "AbsDiff";
    }
};

/**
 * @brief structure for relative difference comparison.
 */
struct AURA_EXPORTS RelativeDiff
{
    /**
     * @brief Calculates the relative difference between two values.
     *
     * @param val0 First value.
     * @param val1 Second value.
     *
     * @return Relative difference between val0 and val1.
     */
    DT_F64 operator()(const DT_F64 val0, const DT_F64 val1) const
    {
        if (val0 == val1)
        {
            return 0;
        }
        else
        {
            DT_F64 abs_diff = Abs(val0 - val1);
            DT_F64 max_abs_val = Max(Abs(val0), Abs(val1));
            return abs_diff / max_abs_val;
        }
    }

    /**
     * @brief Gets a string representation of the comparison method.
     */
    static std::string ToString()
    {
        return "RelativeDiff";
    }
};

/**
 * @brief structure containing position and values of difference between two Mat.
 */
struct AURA_EXPORTS MatCmpPos
{
    DT_S32 pos[3];  /*!> h, w, c */
    DT_F64 val[3];  /*!> src, ref, diff */

    /**
     * @brief Default constructor initializing position and values.
     */
    MatCmpPos() : pos{-1, -1, -1}, val{0, 0, 0}
    {}

    /**
     * @brief Constructor initializing position and values with given parameters.
     *
     * @param h Height position.
     * @param w Width position.
     * @param c Channel position.
     * @param src Source matrix value.
     * @param ref Reference matrix value.
     * @param diff Absolute difference between source and reference values.
     */
    MatCmpPos(DT_S32 h, DT_S32 w, DT_S32 c, DT_F64 src, DT_F64 ref, DT_F64 diff)
            : pos{h, w, c}, val{src, ref, diff}
    {}

    /**
     * @brief Overloaded stream insertion operator to print MatCmpPos to ostream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const MatCmpPos &diff)
    {
        os << "diff pos: [" << diff.pos[0] << ", " << diff.pos[1] << ", " << diff.pos[2] << "]";
        os << " src(" << diff.val[0] << ") ref(" << diff.val[1] << ") diff(" << diff.val[2] << ")";
        return os;
    }

    /**
     * @brief Gets a string representation of MatCmpPos.
     */
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

/**
 * @brief Overloaded greater-than operator for MatCmpPos.
 *
 * @param p0 First MatCmpPos to compare.
 * @param p1 Second MatCmpPos to compare.
 *
 * @return True if diff value of p0 is greater than p1, false otherwise.
 */
AURA_INLINE DT_BOOL operator>(const MatCmpPos &p0, const MatCmpPos &p1)
{
    return p0.val[2] > p1.val[2];
}

/**
 * @brief structure containing position and values of differences between two Array.
 */
struct ArrayCmpPos
{
    DT_S32 pos;     /*!> idx */
    DT_F64 val[3];  /*!> src, ref, diff  */

    /**
     * @brief Constructor initializing position and values with given parameters.
     *
     * @param pos Index position.
     * @param src Source array value.
     * @param ref Reference array value.
     * @param diff Absolute difference between source and reference values.
     */
    ArrayCmpPos(DT_S32 pos = -1, DT_F64 src = 0, DT_F64 ref = 0, DT_F64 diff = 0) : pos{pos}, val{src, ref, diff}
    {}

    /**
     * @brief Overloaded stream insertion operator to print ArrayCmpPos to ostream.
     */
    friend std::ostream& operator<<(std::ostream &os, const ArrayCmpPos &diff)
    {
        os << "diff pos: [" << diff.pos << "]";
        os << " src(" << diff.val[0] << ") ref(" << diff.val[1] << ") diff(" << diff.val[2] << ")";
        return os;
    }

    /**
     * @brief Gets a string representation of ArrayCmpPos.
     */
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

/**
 * @brief Overloaded greater-than operator for ArrayCmpPos.
 *
 * @param p0 First ArrayCmpPos to compare.
 * @param p1 Second ArrayCmpPos to compare.
 *
 * @return True if diff value of p0 is greater than p1, false otherwise.
 */
AURA_INLINE DT_BOOL operator>(const ArrayCmpPos &p0, const ArrayCmpPos &p1)
{
    return p0.val[2] > p1.val[2];
}

/**
 * @brief structure containing position and values of differences between two scalars.
 */
struct ScalarCmpPos
{
    DT_S32 pos[2]; /*!< Array representing the position (vector_idx, scalar_idx). */
    DT_F64 val[3]; /*!< Array representing values (src, ref, diff). */

    /**
     * @brief Default constructor initializing position and values.
     */
    ScalarCmpPos() : pos{-1, -1}, val{0, 0, 0}
    {}

    /**
     * @brief Constructor initializing position and values.
     *
     * @param pos0 Position in the vector.
     * @param pos1 Scalar index (0 ~ 3).
     * @param src Source value.
     * @param ref Reference value.
     * @param diff Difference value.
     */
    ScalarCmpPos(DT_S32 pos0, DT_S32 pos1, DT_F64 src = 0, DT_F64 ref = 0, DT_F64 diff = 0) : pos{pos0, pos1}, val{src, ref, diff}
    {}

    /**
     * @brief Overloaded output stream operator to print ScalarCmpPos.
     */
    friend std::ostream& operator<<(std::ostream &os, const ScalarCmpPos &diff)
    {
        os << "diff pos: [" << diff.pos[0] << ", " << diff.pos[1] << "]";
        os << " src(" << diff.val[0] << ") ref(" << diff.val[1] << ") diff(" << diff.val[2] << ")";
        return os;
    }

    /**
     * @brief Converts ScalarCmpPos to a string.
     */
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

/**
 * @brief Overloaded greater-than operator for ScalarCmpPos.
 *
 * @param p0 ScalarCmpPos operand.
 * @param p1 ScalarCmpPos operand.
 *
 * @return True if the value at index 2 of p0 is greater than the value at index 2 of p1, otherwise false.
 */
AURA_INLINE DT_BOOL operator>(const ScalarCmpPos &p0, const ScalarCmpPos &p1)
{
    return p0.val[2] > p1.val[2];
}

/**
 * @brief Represents the result of arithmetic comparison.
 *
 * @tparam Tp The type of elements in the result.
 */
template<typename Tp>
struct ArithmCmpResult
{
    DT_S32 status;                                  /*!< Status of the comparison. */
    DT_S32 total;                                   /*!< Total number of elements. */
    std::string cmp_method;                         /*!< Comparison method. */
    std::vector<std::pair<DT_F64, DT_S32>> hist;    /*!< Vector of tolerances and corresponding pixel counts. */

    DT_BOOL is_detail;                              /*!< Indicates whether detailed information is available. */
    std::vector<Tp> detail;                         /*!< Vector of detailed differences. */

    /**
     * @brief Default constructor for ArithmCmpResult.
     */
    ArithmCmpResult() : status(0), total(0), is_detail(DT_FALSE)
    {
        hist.clear();
        detail.clear();
    }

    /**
     * @brief Clears the result.
     */
    DT_VOID Clear()
    {
        status = 0;
        total = 0;
        cmp_method.clear();
        hist.clear();
        detail.clear();
    }

    /**
     * @brief Overloaded output stream operator to print ArithmCmpResult.
     */
    friend std::ostream& operator<<(std::ostream &os, const ArithmCmpResult &result)
    {
        os << result.cmp_method << ": ";

        DT_CHAR buffer[1024] = {0};
        for (size_t i = 0; i < result.hist.size(); i++)
        {
            auto hi = result.hist[i];
            snprintf(buffer, sizeof(buffer), "%.4f%%(<=%f) ", 100. * hi.second / result.total, hi.first);
            os << buffer;
        }

        snprintf(buffer, sizeof(buffer), "%.4f%%(>%f)", 100. * (result.total - result.hist[result.hist.size() - 1].second) / result.total, result.hist[result.hist.size() - 1].first);
        os << buffer;
        return os;
    }

    /**
     * @brief Converts ArithmCmpResult to a string.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    /**
     * @brief Gets the vector of detailed differences.
     *
     * @return Vector of detailed differences.
     */
    std::vector<Tp> GetVecDiffPos()
    {
        return detail;
    }

    /**
     * @brief Get maximum difference.
     *
     * @return Maximum difference.
     */
    Tp GetMaxDiffPos()
    {
        if (detail.empty())
        {
            return Tp();
        }

        Tp max_diff = detail[0];
        size_t n_vec = detail.size();
        for (size_t i = 1; i < n_vec; i++)
        {
            if (detail[i] > max_diff)
            {
                max_diff = detail[i];
            }
        }

        return max_diff;
    }
};

using MatCmpResult    = ArithmCmpResult<MatCmpPos>;    /*!< Type alias for arithmetic comparison result with MatCmpPos. */
using ArrayCmpResult  = ArithmCmpResult<ArrayCmpPos>;  /*!< Type alias for arithmetic comparison result with ArrayCmpPos. */
using ScalarCmpResult = ArithmCmpResult<ScalarCmpPos>; /*!< Type alias for arithmetic comparison result with ScalarCmpPos. */


/**
 * @brief Compares two matrices and calculates the arithmetic comparison result with detailed information.
 *
 * @tparam Tp0 The type of elements in the source matrix.
 * @tparam Tp1 The type of elements in the reference matrix.
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param ref The reference matrix.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename Tp0, typename Tp1, typename CmpFunc = AbsDiff>
Status MatCompare_(Context *ctx, const Mat &src, const Mat &ref, MatCmpResult &result,
                   DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    result.Clear();

    tolerate = Abs(tolerate);
    step     = Min(Abs(step), tolerate);
    cmp_eps  = Max(Min(cmp_eps, tolerate), DBL_EPSILON);

    CmpFunc compare_func;
    result.cmp_method = CmpFunc::ToString();

    // check size
    DT_S32 tot_pixel = 0;
    auto size_src = src.GetSizes();
    auto size_ref = ref.GetSizes();
    if (size_ref != size_src)
    {
        AURA_LOGE(ctx, AURA_TAG, "mat size_src not match");
        return Status::ERROR;
    }
    tot_pixel += size_src.Total();
    result.total = tot_pixel;

    // compare
    DT_F64 max_diff = 0;
    DT_F64 max_diff_src_val = 0;
    DT_F64 max_diff_ref_val = 0;
    DT_S64 interval = (0 == step) ? 0 : Ceil(tolerate / step);

    std::vector<DT_S32> diff_count(interval + 2, 0);

    Sizes3 max_pos = {0, 0, 0};

    auto size = src.GetSizes();

    for (auto y = 0; y < size.m_height; ++y)
    {
        const Tp0 *src_c = src.Ptr<Tp0>(y);
        const Tp1 *ref_c = ref.Ptr<Tp1>(y);
        for (auto x = 0; x < size.m_width; ++x)
        {
            for (auto c = 0; c < size.m_channel; ++c)
            {
                DT_S32 idx = x * size.m_channel + c;
                DT_F64 v_src = SaturateCast<DT_F64>(src_c[idx]);
                DT_F64 v_ref = SaturateCast<DT_F64>(ref_c[idx]);

                // calculate compare error using compare function which can be defined by user
                DT_F64 diff = compare_func(v_src, v_ref);

                if (!std::isnormal(diff) && diff != 0.0)
                {
                    max_diff          = diff;
                    max_diff_src_val  = v_src;
                    max_diff_ref_val  = v_ref;
                    max_pos.m_height  = y;
                    max_pos.m_width   = x;
                    max_pos.m_channel = c;
                    diff_count[interval + 1]++;
                    goto EXIT;
                }

                if (diff > cmp_eps)
                {
                    if (result.is_detail)
                    {
                        if (diff > tolerate)
                        {
                            result.detail.emplace_back(MatCmpPos(y, x, c, v_src, v_ref, diff));
                        }
                    }

                    // update max_diff
                    if (diff > max_diff)
                    {
                        max_diff          = diff;
                        max_diff_src_val  = v_src;
                        max_diff_ref_val  = v_ref;
                        max_pos.m_height  = y;
                        max_pos.m_width   = x;
                        max_pos.m_channel = c;
                    }

                    DT_S32 diff_idx = (0 == step) ? (diff > tolerate) : Min(Ceil(diff / step), interval + 1);
                    diff_count[diff_idx]++;
                }
                else
                {
                    diff_count[0]++;
                }
            }
        }
    }

EXIT:
    AURA_LOGI(ctx, AURA_TAG, "MatCompare error distribution: \n");

    if (max_diff > tolerate || (!std::isnormal(max_diff) && max_diff != 0.0))
    {
        AURA_LOGI(ctx, AURA_TAG, "Max diff(%.4f) src/ref(%.4f %.4f) Pos(%d %d %d) \n", max_diff, max_diff_src_val, max_diff_ref_val, max_pos.m_height, max_pos.m_width, max_pos.m_channel);
    }

    DT_F64 cur_tol = 0;
    DT_S32 cur_sum = 0;
    DT_F64 cur_pct = 0;

    for (DT_S32 i = 0; i <= interval; i++)
    {
        cur_tol = Min(i * step, tolerate);
        cur_sum += diff_count[i];
        cur_pct = 1. * cur_sum / tot_pixel;
        result.hist.emplace_back(std::make_pair(cur_tol, cur_sum));
        AURA_LOGI(ctx, AURA_TAG, "diff <= %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);
    }

    cur_tol = tolerate;
    cur_sum = tot_pixel - cur_sum;
    cur_pct = 1. * cur_sum / tot_pixel;
    AURA_LOGI(ctx, AURA_TAG, "diff > %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);

    if (max_diff > tolerate || (!std::isnormal(max_diff) && max_diff != 0.0))
    {
        result.status = DT_FALSE;
    }
    else
    {
        result.status = DT_TRUE;
    }

    if (!std::isnormal(max_diff) && max_diff != 0.0)
    {
        AURA_LOGI(ctx, AURA_TAG, "max_diff is not a normal value %f\n", max_diff);
    }

    return Status::OK;
}

/**
 * @brief Helper function to compare two matrices with element-wise comparison using a specified comparison function.
 *
 * @tparam Tp The type of elements in the matrices.
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param ref The reference matrix.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename Tp, typename CmpFunc = AbsDiff>
Status MatCompareHelper(Context *ctx, const Mat &src, const Mat &ref, MatCmpResult &result,
                        DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    switch (ref.GetElemType())
    {
        case ElemType::U8:
        {
            MatCompare_<Tp, DT_U8, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S8:
        {
            MatCompare_<Tp, DT_S8, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::U16:
        {
            MatCompare_<Tp, DT_U16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S16:
        {
            MatCompare_<Tp, DT_S16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            MatCompare_<Tp, MI_F16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::U32:
        {
            MatCompare_<Tp, DT_U32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S32:
        {
            MatCompare_<Tp, DT_S32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::F32:
        {
            MatCompare_<Tp, DT_F32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::F64:
        {
            MatCompare_<Tp, DT_F64, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        default:
        {
            AURA_LOGE(ctx, AURA_TAG, "iaura elem type not support");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

/**
 * @brief Compare two matrices with element-wise comparison using a specified comparison function.
 *
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param ref The reference matrix.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename CmpFunc = AbsDiff>
Status MatCompare(Context *ctx, const Mat &src, const Mat &ref, MatCmpResult &result,
                  DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            MatCompareHelper<DT_U8, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S8:
        {
            MatCompareHelper<DT_S8, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::U16:
        {
            MatCompareHelper<DT_U16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S16:
        {
            MatCompareHelper<DT_S16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            MatCompareHelper<MI_F16, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::U32:
        {
            MatCompareHelper<DT_U32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::S32:
        {
            MatCompareHelper<DT_S32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::F32:
        {
            MatCompareHelper<DT_F32, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        case ElemType::F64:
        {
            MatCompareHelper<DT_F64, CmpFunc>(ctx, src, ref, result, tolerate, step, cmp_eps);
            break;
        }
        default:
        {
            AURA_LOGE(ctx, AURA_TAG, "iaura elem type not support");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

/**
 * @brief Compare multiple pairs of matrices with element-wise comparison using a specified comparison function.
 *
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The vector of source matrices.
 * @param ref The vector of reference matrices.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if the comparison is successful for all pairs; otherwise, an error status.
 */
template<typename CmpFunc = AbsDiff>
Status MatCompare(Context *ctx, const std::vector<Mat> &src, const std::vector<Mat> &ref,
                  MatCmpResult &result, DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    if (src.size() < 1 || ref.size() < 1 || src.size() != ref.size())
    {
        AURA_LOGE(ctx, AURA_TAG, "number of src mat is not equal to number of ref mat\n");
        return Status::ERROR;
    }
    result.Clear();

    size_t num_mat = src.size();
    std::vector<MatCmpResult> v_result(num_mat);

    Status ret = Status::OK;
    for (size_t i = 0; i < num_mat; i++)
    {
        ret |= MatCompare<CmpFunc>(ctx, src[i], ref[i], v_result[i], tolerate, step, cmp_eps);
    }

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "MatCompare failed \n");
        return ret;
    }

    // merge compare result
    result = v_result[0];
    for (size_t i = 1; i < num_mat; i++)
    {
        result.status &= v_result[i].status;
        result.total  += v_result[i].total;
        for (size_t j = 0; j < result.hist.size(); j++)
        {
            result.hist[j].second += v_result[i].hist[j].second;
        }
        result.detail.insert(result.detail.end(), v_result[i].detail.begin(), v_result[i].detail.end());
    }

    return Status::OK;
}

/**
 * @brief Compare multiple pairs of matrices with element-wise comparison using a specified comparison function.
 *
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The vector of source matrices.
 * @param ref The vector of reference matrices.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if the comparison is successful for all pairs; otherwise, an error status.
 */
template<typename CmpFunc = AbsDiff>
Status MatCompare(Context *ctx, const std::vector<Mat*> &src, const std::vector<Mat*> &ref,
                  MatCmpResult &result, DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    if (src.size() < 1 || ref.size() < 1 || src.size() != ref.size())
    {
        AURA_LOGE(ctx, AURA_TAG, "number of src mat is not equal to number of ref mat\n");
        return Status::ERROR;
    }
    result.Clear();

    size_t num_mat = src.size();
    std::vector<MatCmpResult> v_result(num_mat);

    Status ret = Status::OK;
    for (size_t i = 0; i < num_mat; i++)
    {
        ret |= MatCompare<CmpFunc>(ctx, *src[i], *ref[i], v_result[i], tolerate, step, cmp_eps);
    }

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "MatCompare failed \n");
        return ret;
    }

    // merge compare result
    result = v_result[0];
    for (size_t i = 1; i < num_mat; i++)
    {
        result.status &= v_result[i].status;
        result.total  += v_result[i].total;
        for (size_t j = 0; j < result.hist.size(); j++)
        {
            result.hist[j].second += v_result[i].hist[j].second;
        }
        result.detail.insert(result.detail.end(), v_result[i].detail.begin(), v_result[i].detail.end());
    }

    return Status::OK;
}

/**
 * @brief Traits class for extracting the element type of an array or pointer type.
 *
 * @tparam Tp The array or pointer type.
 */
template<typename Tp>
struct ArrayCompareTraits
{
    using type = typename std::conditional<std::is_pointer<Tp>::value, typename std::remove_pointer<Tp>::type,
                                           typename std::iterator_traits<Tp>::value_type>::type;
};

/**
 * @brief Compare two arrays with element-wise comparison using a specified comparison function.
 *
 * @tparam Iter The iterator type for the arrays.
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param src The iterator to the source array.
 * @param ref The iterator to the reference array.
 * @param num The number of elements in the arrays.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename Iter, typename CmpFunc = AbsDiff,
         typename std::enable_if<is_arithmetic<typename ArrayCompareTraits<Iter>::type>::value>::type* = DT_NULL>
Status ArrayCompare(Context *ctx, Iter src, Iter ref, DT_S32 num, ArrayCmpResult &result,
                    DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    result.Clear();

    tolerate = Abs(tolerate);
    step     = Min(Abs(step), tolerate);
    cmp_eps  = Max(Min(cmp_eps, tolerate), DBL_EPSILON);

    if (num <= 0)
    {
        AURA_LOGE(ctx, AURA_TAG, "compare number must be positive \n");
        return Status::ERROR;
    }
    CmpFunc compare_func;
    result.cmp_method = CmpFunc::ToString();

    // compare
    DT_S64 interval = (0 == step) ? 0 : Ceil(tolerate / step);

    std::vector<DT_S32> diff_count(interval + 2, 0);
    DT_F64 max_diff = 0;
    DT_S32 max_diff_idx = 0;

    Iter it_src = src;
    Iter it_ref = ref;
    for (DT_S32 cnt = 0; cnt < num; it_src++, it_ref++, cnt++)
    {
        DT_F64 v_src = SaturateCast<DT_F64>(*it_src);
        DT_F64 v_ref = SaturateCast<DT_F64>(*it_ref);
        DT_F64 diff = compare_func(v_src, v_ref);
        if (diff > cmp_eps)
        {
            if (result.is_detail)
            {
                if (diff > tolerate)
                {
                    result.detail.emplace_back(ArrayCmpPos(cnt, v_src, v_ref, diff));
                }
            }

            if (diff > max_diff)
            {
                max_diff     = diff;
                max_diff_idx = cnt;
            }

            DT_S32 diff_idx = (0 == step) ? (diff > tolerate) : Min(Ceil(diff / step), interval + 1);
            diff_count[diff_idx]++;
        }
        else
        {
            diff_count[0]++;
        }
    }

    AURA_LOGI(ctx, AURA_TAG, "ArrayCompare error distribution: \n");

    if (max_diff > tolerate)
    {
        AURA_LOGI(ctx, AURA_TAG, "Max diff(%.4f) Pos(%d)\n", max_diff, max_diff_idx);
    }

    DT_F64 cur_tol = 0;
    DT_S32 cur_sum = 0;
    DT_F64 cur_pct = 0;

    for (DT_S32 i = 0; i <= interval; i++)
    {
        cur_tol = Min(i * step, tolerate);
        cur_sum += diff_count[i];
        cur_pct = 1. * cur_sum / num;
        result.hist.emplace_back(std::make_pair(cur_tol, cur_sum));
        AURA_LOGI(ctx, AURA_TAG, "diff <= %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);
    }

    cur_tol = tolerate;
    cur_sum = num - cur_sum;
    cur_pct = 1. * cur_sum / num;
    AURA_LOGI(ctx, AURA_TAG, "diff > %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);

    result.total = num;
    if (max_diff > tolerate)
    {
        result.status = DT_FALSE;
    }
    else
    {
        result.status = DT_TRUE;
    }

    return Status::OK;
}

/**
 * @brief Compare two vectors of Scalar objects with element-wise comparison using a specified comparison function.
 *
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param scalar0 The vector of Scalar objects for comparison.
 * @param scalar1 The vector of Scalar objects for comparison.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename CmpFunc = AbsDiff>
Status ScalarCompare(Context *ctx, const std::vector<Scalar> &scalar0, const std::vector<Scalar> &scalar1,
                     ScalarCmpResult &result, DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    if (scalar0.size() < 1 || scalar1.size() < 1 || scalar0.size() != scalar1.size())
    {
        AURA_LOGE(ctx, AURA_TAG, "scalar size error \n");
        return Status::ERROR;
    }

    result.Clear();

    tolerate = Abs(tolerate);
    step     = Min(Abs(step), tolerate);
    cmp_eps  = Max(Min(cmp_eps, tolerate), DBL_EPSILON);

    CmpFunc compare_func;
    result.cmp_method = CmpFunc::ToString();

    // compare
    DT_S32 num_scalar = static_cast<DT_S32>(scalar0.size());
    DT_S64 interval = (0 == step) ? 0 : Ceil(tolerate / step);

    std::vector<DT_S32> diff_count(interval + 2, 0);
    DT_F64 max_diff = 0;
    DT_S32 max_diff_idx[2] = {0};

    for (DT_S32 idx_scalar = 0; idx_scalar < num_scalar; idx_scalar++)
    {
        for (DT_S32 i = 0; i < 4; i++)
        {
            DT_F64 v0 = scalar0[idx_scalar].m_val[i];
            DT_F64 v1 = scalar1[idx_scalar].m_val[i];
            DT_F64 diff = compare_func(v0, v1);
            if (diff > cmp_eps)
            {
                if (result.is_detail)
                {
                    if (diff > tolerate)
                    {
                        result.detail.emplace_back(ScalarCmpPos(idx_scalar, i, v0, v1, diff));
                    }
                }

                if (diff > max_diff)
                {
                    max_diff        = diff;
                    max_diff_idx[0] = idx_scalar;
                    max_diff_idx[1] = i;
                }

                DT_S32 diff_idx = (0 == step) ? (diff > tolerate) : Min(Ceil(diff / step), interval + 1);
                diff_count[diff_idx]++;
            }
            else
            {
                diff_count[0]++;
            }
        }
    }

    AURA_LOGI(ctx, AURA_TAG, "ArrayCompare error distribution: \n");

    if (max_diff > tolerate)
    {
        AURA_LOGI(ctx, AURA_TAG, "Max diff(%.4f) Pos(%d, %d)\n", max_diff, max_diff_idx[0], max_diff_idx[1]);
    }

    DT_F64 cur_tol = 0;
    DT_S32 cur_sum = 0;
    DT_F64 cur_pct = 0;
    DT_S32 num_tot = num_scalar * 4;

    for (DT_S32 i = 0; i <= interval; i++)
    {
        cur_tol = Min(i * step, tolerate);
        cur_sum += diff_count[i];
        cur_pct = 1. * cur_sum / num_tot;
        result.hist.emplace_back(std::make_pair(cur_tol, cur_sum));
        AURA_LOGI(ctx, AURA_TAG, "diff <= %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);
    }

    cur_tol = tolerate;
    cur_sum = num_tot - cur_sum;
    cur_pct = 1. * cur_sum / num_tot;
    AURA_LOGI(ctx, AURA_TAG, "diff > %f : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);

    result.total = num_tot;
    if (max_diff > tolerate)
    {
        result.status = DT_FALSE;
    }
    else
    {
        result.status = DT_TRUE;
    }

    return Status::OK;
}

/**
 * @brief Compare two Scalar objects with element-wise comparison using a specified comparison function.
 *
 * @tparam CmpFunc The comparison function to use for element-wise comparison.
 *
 * @param ctx The pointer to the Context object.
 * @param scalar0 The first Scalar object for comparison.
 * @param scalar1 The second Scalar object for comparison.
 * @param result The result structure to store the comparison details.
 * @param tolerate The tolerance for the comparison.
 * @param step The step for calculating the histogram of differences.
 * @param cmp_eps The epsilon value for comparison.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename CmpFunc = AbsDiff>
Status ScalarCompare(Context *ctx, const Scalar &scalar0, const Scalar &scalar1,
                     ScalarCmpResult &result, DT_F64 tolerate = 1, DT_F64 step = 1, DT_F64 cmp_eps = 1e-6)
{
    std::vector<Scalar> v0{scalar0};
    std::vector<Scalar> v1{scalar1};

    return ScalarCompare<CmpFunc>(ctx, v0 , v1, result, tolerate, step, cmp_eps);
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_COMPARE_HPP__
