/** @brief      : find_contours impl for aura
 *  @file       : find_contours_impl.hpp
 *  @author     : wangyisi@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Aug. 28, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_CONTOURS_FIND_CONTOURS_IMPL_HPP__
#define AURA_OPS_CONTOURS_FIND_CONTOURS_IMPL_HPP__

#include "aura/ops/misc/find_contours.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class FindContoursImpl : public OpImpl
{
public:
    FindContoursImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy,
                           ContoursMode mode, ContoursMethod method, Point2i offset); 

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    ContoursMode   m_mode;
    ContoursMethod m_method;
    Point2i        m_offset;

    const Array *m_src;
    std::vector<std::vector<Point2i>> *m_contours;
    std::vector<Scalari> *m_hierarchy;
};

class FindContoursNone : public FindContoursImpl
{
public:
    FindContoursNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy,
                   ContoursMode mode = ContoursMode::RETR_EXTERNAL, ContoursMethod method = ContoursMethod::CHAIN_APPROX_SIMPLE, 
                   Point2i offset = Point2i()) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_CONTOURS_FIND_CONTOURS_IMPL_HPP__