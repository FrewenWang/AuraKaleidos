// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTIONDEPTHWISE_IOV_X86_64_H
#define LAYER_CONVOLUTIONDEPTHWISE_IOV_X86_64_H

#include "../../../../../../third_party/ncnn/20190320/src/layer/convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_iov_x86_64 : public ConvolutionDepthWise
{
public:
    ConvolutionDepthWise_iov_x86_64();
    virtual ~ConvolutionDepthWise_iov_x86_64();

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    std::vector<ncnn::Layer*> group_ops;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_X86_H
