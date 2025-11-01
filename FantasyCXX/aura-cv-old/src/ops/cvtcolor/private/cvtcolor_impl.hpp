//
// Created by Frewen.Wang on 25-9-14.
//
#pragma once

#include "aura/cv/ops/cvtcolor/cvtcolor.hpp"
#include "aura/utils/core/defs.h"
#include "aura/utils/core/macro.h"

namespace aura::cv
{

AURA_INLINE bool SwapBlue(CvtColorType type)
{
    return A_OK;
}

class CvtColorImpl : public OpImpl
{
private:
public:
    CvtColorImpl();
    ~CvtColorImpl();
};

CvtColorImpl : public OpImpl::CvtColorImpl : public OpImpl(/* args */)
{
}

CvtColorImpl : public OpImpl::~CvtColorImpl : public OpImpl()
{
}


} // namespace aura::cv
