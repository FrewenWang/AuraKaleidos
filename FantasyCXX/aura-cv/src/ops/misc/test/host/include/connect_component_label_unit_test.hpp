/** @brief      : connect component unit test head for aura
 *  @file       : connect component_unit_test.hpp
 *  @author     : wangshiyu7@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 25, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_CONNECT_COMPONENT_UINT_TEST_HPP__
#define AURA_OPS_CONNECT_COMPONENT_UINT_TEST_HPP__

#include "aura/ops/misc/connect_component_label.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

#include <set>
#include <queue>

using namespace aura;

static Status GenerateCustomedMask(aura::Mat &mask, DT_F32 density)
{
    if (density <= 0.f)
    {
        memset(mask.GetData(), 0, mask.GetTotalBytes());
        return Status::OK;
    }
    else if(density >= 0.99f)
    {
        memset(mask.GetData(), 255, mask.GetTotalBytes());
        return Status::OK;
    }
    memset(mask.GetData(), 0, mask.GetTotalBytes());

    const DT_S32 height = mask.GetSizes().m_height;
    const DT_S32 width  = mask.GetSizes().m_width;
    const DT_S32 seed   = 2023;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<DT_F32> dis(0.0f, 1.0f);

    for (DT_S32 y = 0; y < height; y++)
    {
        for (DT_S32 x = 0; x < width; x++)
        {
            if (dis(gen) < density)
            {
                mask.At<DT_U8>(y, x) = 255;
            }
            else
            {
                mask.At<DT_U8>(y, x) = 0;
            }
        }
    }

    return Status::OK;
}

template <typename LabelType>
static Status LabelCheck(Context *ctx, const Mat &img, const Mat &label, aura::ConnectivityType connectivity_type,
                         MatCmpResult &result, DT_S32 tolerate = 0)
{
    result.Clear();
    result.cmp_method = "connectivity BFS";
    
    tolerate = Abs(tolerate);
    DT_S32 max_diff = 0, diff_count = 0;
    DT_S32 dx[8] = {0, 0, 1, -1, 1, -1,  1, -1};
    DT_S32 dy[8] = {1,-1, 0,  0, 1, -1, -1,  1};
    DT_S32 search_areas = aura::ConnectivityType::CROSS == connectivity_type ? 4 : 8;

    DT_S32 tot_pixel = 0;
    auto size = label.GetSizes();
    tot_pixel += size.Total();

    result.total = tot_pixel;

    Mat visited(ctx, ElemType::U8, size);
    Sizes3 max_pos = {0, 0, 0};

    // 1.check labeled
    for (auto y = 0; y < size.m_height; ++y)
    {
        for (auto x = 0; x < size.m_width; ++x)
        {
            // diff pixel
            if ((label.Ptr<LabelType>(y)[x] ^ img.Ptr<DT_U8>(y)[x]) < 0)
            {
                diff_count++;
                max_diff = label.Ptr<LabelType>(y)[x];
                max_pos.m_height  = y;
                max_pos.m_width   = x;
                max_pos.m_channel = 1;
                goto EXIT;
            }
        }
    }

    // 2.check connected
    for (auto y = 0; y < size.m_height; ++y)
    {
        for (auto x = 0; x < size.m_width; ++x)
        {
            if (0 == visited.Ptr<DT_U8>(y)[x] && label.Ptr<LabelType>(y)[x])
            {
                std::queue<std::pair<DT_S32, DT_S32>> neighbor_pixels;
                std::vector<LabelType> connected_component;
                std::vector<std::pair<DT_S32, DT_S32>> connected_component_pos;
                neighbor_pixels.push(std::pair<DT_S32, DT_S32>(y, x));
                visited.Ptr<DT_U8>(y)[x] = 255;
                while (!neighbor_pixels.empty())
                {
                    std::pair<DT_S32, DT_S32> cur_pixel = neighbor_pixels.front();
                    DT_S32 cur_y = cur_pixel.first;
                    DT_S32 cur_x = cur_pixel.second;
                    connected_component.emplace_back(label.Ptr<LabelType>(cur_y)[cur_x]);
                    connected_component_pos.emplace_back(std::pair<DT_S32, DT_S32>(cur_y, cur_x));
                    neighbor_pixels.pop();
                    for (DT_S32 direction = 0; direction < search_areas; direction++) 
                    {
                        DT_S32 new_x = cur_x + dx[direction];
                        DT_S32 new_y = cur_y + dy[direction];
                        if ((new_x == cur_x && new_y == cur_y) ||
                            (!(new_x >= 0 && new_x < size.m_width && new_y >= 0 && new_y < size.m_height)))
                        {
                            continue;
                        }
                        if (label.Ptr<LabelType>(new_y)[new_x] > 0 &&
                            0 == visited.Ptr<DT_U8>(new_y)[new_x])
                        {
                            visited.Ptr<DT_U8>(new_y)[new_x] = 255;
                            neighbor_pixels.push(std::pair<DT_S32, DT_S32>(new_y, new_x));
                        }
                    }
                }

                std::set<LabelType> labels_set(connected_component.begin(), connected_component.end());
                if (labels_set.size() != 1) // connected component must be same label value
                {
                    auto minmax = std::minmax_element(connected_component.begin(), connected_component.end());
                    max_diff = *minmax.second - *minmax.first;
                    diff_count += connected_component.size();
                    max_pos.m_height  = y;
                    max_pos.m_width   = x;
                    max_pos.m_channel = 1;

                    for (auto pos : connected_component_pos)
                    {
                        std::cout << pos.first << " " << pos.second  << "=" << label.Ptr<LabelType>(pos.first)[pos.second] << std::endl;
                    }
                    goto EXIT;
                }
            }
        }
    }

EXIT:
    AURA_LOGI(ctx, AURA_TAG, "MatCompare error distribution: \n");

    if (max_diff > tolerate || max_diff != 0)
    {
        AURA_LOGI(ctx, AURA_TAG, "Max diff(%d) Pos(%d %d %d) \n", max_diff, max_pos.m_height, max_pos.m_width, max_pos.m_channel);
        result.status = DT_FALSE;
    }
    else
    {
        result.status = DT_TRUE;
    }

    DT_S32 cur_tol = tolerate;
    DT_S32 cur_sum = tot_pixel - diff_count;
    DT_F64 cur_pct = 1. * cur_sum / tot_pixel;
    result.hist.emplace_back(std::make_pair(cur_tol, cur_sum));
    AURA_LOGI(ctx, AURA_TAG, "diff <= %d : %.4f%% %d\n", cur_tol, 100. * cur_pct, cur_sum);

    return Status::OK;
}

struct ConnectComponentLabelTestParam
{
    ConnectComponentLabelTestParam()
    {}

    ConnectComponentLabelTestParam(aura::CCLAlgo algo_type, aura::ConnectivityType connectivity_type,
                                   aura::EquivalenceSolver solver_type) : algo_type(algo_type), connectivity_type(connectivity_type),
                                                                          solver_type(solver_type)
    {}

    friend std::ostream& operator << (std::ostream &os, ConnectComponentLabelTestParam connectcomponent_test_param)
    {
        os << "algo_type:" << connectcomponent_test_param.algo_type << " connectivity_type:" << connectcomponent_test_param.connectivity_type \
           << "solver_type:" << connectcomponent_test_param.solver_type << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    aura::CCLAlgo           algo_type;
    aura::ConnectivityType  connectivity_type;
    aura::EquivalenceSolver solver_type;
};

AURA_TEST_PARAM(MiscConnectComponentLabelParam,
                aura::MatSize,                  mat_sizes,
                aura::ElemType,                 label_elem_type,
                DT_F32,                         mask_density,
                ConnectComponentLabelTestParam, param,
                aura::OpTarget,                 target);

#if !defined(AURA_BUILD_XPLORER)
static DT_VOID CvConvertAlgoType(aura::Context *ctx, aura::CCLAlgo algo_type, aura::ConnectivityType connectivity_type,
                                 DT_S32 &connectivity, cv::ConnectedComponentsAlgorithmsTypes &cv_algo_type)
{
    if (aura::ConnectivityType::CROSS == connectivity_type)
    {
        connectivity = 4;
        switch (algo_type)
        {
            case (aura::CCLAlgo::SAUF):
            {
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_WU;
                break;
            }
            case (aura::CCLAlgo::BBDT):
            {
                AURA_LOGD(ctx, AURA_TAG, "BBDT only support connectivity = 4, set to SAUF\n");
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_DEFAULT;
                return;
            }
            default:
            {
                AURA_LOGD(ctx, AURA_TAG, "SPAGHETTI not support yet in opencv, set to Default\n");
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_DEFAULT;
                break;
            }
        }
    }
    else if (aura::ConnectivityType::SQUARE == connectivity_type)
    {
        connectivity = 8;
        switch (algo_type)
        {
            case (aura::CCLAlgo::SAUF):
            {
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_WU;
                break;
            }
            case (aura::CCLAlgo::BBDT):
            {
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_GRANA;
                break;
            }
            default:
            {
                AURA_LOGD(ctx, AURA_TAG, "SPAGHETTI not support yet in opencv, set to BBDT\n");
                cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_GRANA;
                break;
            }
        }
    }
    else
    {
        AURA_LOGD(ctx, AURA_TAG, "unsupport connectivity type\n");
    }
}
#endif

static aura::Status CvConnectComponentLabel(aura::Context *ctx, aura::Mat &img, aura::Mat &label, aura::CCLAlgo algo_type,
                                            aura::ConnectivityType connectivity_type)
{
    if (label.GetElemType() == aura::ElemType::S32)
    {
#if !defined(AURA_BUILD_XPLORER)
        DT_S32 cv_connectivity = 0;
        cv::ConnectedComponentsAlgorithmsTypes cv_algo_type = cv::ConnectedComponentsAlgorithmsTypes::CCL_DEFAULT;

        CvConvertAlgoType(ctx, algo_type, connectivity_type, cv_connectivity, cv_algo_type);

        cv::Mat cv_img = aura::MatToOpencv(img);
        cv::Mat cv_label = aura::MatToOpencv(label);

        cv::connectedComponents(cv_img, cv_label, cv_connectivity, 4, cv_algo_type);
#else
        AURA_UNUSED(img);
        AURA_UNUSED(algo_type);
        AURA_UNUSED(connectivity_type);
#endif
        return aura::Status::OK;
    }
    else
    {
        AURA_LOGD(ctx, AURA_TAG, "CV ccl not support\n");
        return aura::Status::ERROR;
    }
}

class ConnectComponentLabelTest : public aura::TestBase<MiscConnectComponentLabelParam::TupleTable, MiscConnectComponentLabelParam::Tuple>
{
public:
    ConnectComponentLabelTest(aura::Context *ctx, MiscConnectComponentLabelParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {}

#if defined(AURA_ENABLE_OPENCL)
    Status CheckParam(DT_S32 index) override
    {
        MiscConnectComponentLabelParam run_param(GetParam((index)));

        if (TargetType::OPENCL == run_param.target.m_type)
        {
            // bypass opencl testcase on mtk
            if (m_ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type != GpuType::ADRENO)
            {
                return Status::ERROR;
            }
            
            std::shared_ptr<cl::Device> cl_device = m_ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
            std::string device_version = cl_device->getInfo<CL_DEVICE_VERSION>();
            DT_S32 adreno_num = std::stoi(device_version.substr(device_version.find_last_of(" ")));

            // adreno650 can not support atomic operate, bypass
            if (adreno_num <= 650)
            {
                return Status::ERROR;
            }
        }

        return Status::OK;
    }
#endif // AURA_ENABLE_OPENCL

    DT_S32 RunOne(DT_S32 index, aura::TestCase *test_case, DT_S32 stress_count) override
    {
        aura::Status ret = aura::Status::OK;

        // get next param set
        MiscConnectComponentLabelParam run_param(GetParam((index)));

        // create iauras
        aura::Mat src_img   = m_factory.GetEmptyMat(ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        aura::Mat dst_label = m_factory.GetEmptyMat(run_param.label_elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        aura::Mat ref_label = m_factory.GetEmptyMat(run_param.label_elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        memset(src_img.GetData(), 0, src_img.GetTotalBytes());
        memset(dst_label.GetData(), 0, dst_label.GetTotalBytes());
        memset(ref_label.GetData(), 0, ref_label.GetTotalBytes());
        DT_S32 loop_count = stress_count ? stress_count : 10;

        aura::TestTime time_val;
        aura::MatCmpResult cmp_result;
        aura::TestResult result;
        std::ostringstream ss;
        ss << run_param.param.algo_type << " | " << run_param.param.solver_type << " | " << run_param.param.connectivity_type;

        result.param  = ss.str();
        result.input  = run_param.mat_sizes.ToString() + " " + std::to_string(run_param.mask_density);
        result.output = run_param.mat_sizes.ToString() + " " + aura::ElemTypesToString(run_param.label_elem_type);

        // ready input Mat
        ret = GenerateCustomedMask(src_img, run_param.mask_density);
        if (ret != aura::Status::OK)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "GenerateCustomedMask failed in ConnectComponentLabelTest\n");
            goto EXIT;
        }

        // run interface
        ret = aura::Executor(loop_count, 2, time_val, aura::IConnectComponentLabel, m_ctx, src_img, dst_label, run_param.param.algo_type,
                             run_param.param.connectivity_type, run_param.param.solver_type, run_param.target);
        if (aura::Status::OK == ret)
        {
            result.perf_result[aura::TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = aura::TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = aura::TestStatus::FAILED;
            result.accu_status = aura::TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (aura::TargetType::NONE == run_param.target.m_type)
        {
            ret = aura::Executor(loop_count, 2, time_val, CvConnectComponentLabel, m_ctx, src_img, ref_label,
                                 run_param.param.algo_type, run_param.param.connectivity_type);
            result.accu_benchmark = "OpenCV::ConnectComponentLabel";

            if (ret != aura::Status::OK)
            {
                result.accu_status = aura::TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = aura::IConnectComponentLabel(m_ctx, src_img, ref_label, aura::CCLAlgo::SAUF, run_param.param.connectivity_type,
                                               run_param.param.solver_type, aura::TargetType::NONE);
            result.accu_benchmark = "aura::ConnectComponentLabel(target::none)";

            if (ret != aura::Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = aura::TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (aura::TargetType::NONE != run_param.target.m_type)
        {
            switch (dst_label.GetElemType())
            {
                case ElemType::U32:
                {
                    LabelCheck<DT_U32>(m_ctx, src_img, dst_label, run_param.param.connectivity_type, cmp_result, 0);
                    break;
                }

                case ElemType::S32:
                {
                    LabelCheck<DT_S32>(m_ctx, src_img, dst_label, run_param.param.connectivity_type, cmp_result, 0);
                    break;
                }

                default:
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "label elem type not support");
                    goto EXIT;
                }
            }
        }
        else
        {
            aura::MatCompare(m_ctx, dst_label, ref_label, cmp_result, 0);
        }
        result.accu_status = cmp_result.status ? aura::TestStatus::PASSED : aura::TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutMats(src_img, dst_label, ref_label);
        return 0;
    }

private:
    aura::Context *m_ctx;
    MatFactory     m_factory;
};

#endif // AURA_OPS_CONNECT_COMPONENT_UINT_TEST_HPP__