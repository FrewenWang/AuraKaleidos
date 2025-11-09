/** @brief      : houghlines impl header for aura
 *  @file       : houghlines_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_HOUGH_LINES_IMPL_HPP__
#define AURA_OPS_MISC_HOUGH_LINES_IMPL_HPP__

#include "aura/ops/misc/houghlines.hpp"

namespace aura
{

class HoughLinesImpl : public OpImpl
{
public:
    HoughLinesImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<Scalar> &lines, LinesType line_type, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                           DT_F64 srn = 0, DT_F64 stn = 0, DT_F64 min_theta = 0, DT_F64 max_theta = AURA_PI);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    LinesType m_line_type;
    DT_F64    m_rho;
    DT_F64    m_theta;
    DT_S32    m_threshold;
    DT_F64    m_srn;
    DT_F64    m_stn;
    DT_F64    m_min_theta;
    DT_F64    m_max_theta;

    const Array *m_src;
    std::vector<Scalar> *m_lines;
};

class HoughLinesNone : public HoughLinesImpl
{
public:
    HoughLinesNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<Scalar> &lines, LinesType line_type, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                   DT_F64 srn = 0, DT_F64 stn = 0, DT_F64 min_theta = 0, DT_F64 max_theta = AURA_PI) override;

    Status Run() override;
};

class HoughLinesPImpl : public OpImpl
{
public:
    HoughLinesPImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<Scalari> &lines, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                           DT_F64 min_line_length, DT_F64 max_gap);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_F64 m_rho;
    DT_F64 m_theta;
    DT_S32 m_threshold;
    DT_F64 m_min_line_length;
    DT_F64 m_max_gap;

    const Array *m_src;
    std::vector<Scalari> *m_lines;
};

class HoughLinesPNone : public HoughLinesPImpl
{
public:
    HoughLinesPNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<Scalari> &lines, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                   DT_F64 min_line_length, DT_F64 max_gap) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_MISC_HOUGH_LINES_IMPL_HPP__