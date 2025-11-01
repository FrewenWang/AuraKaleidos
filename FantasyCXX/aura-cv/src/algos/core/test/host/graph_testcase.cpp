#include "aura/algos/core.h"
#include "aura/tools/unit_test.h"
#include "aura/ops/filter.h"
#include "aura/ops/matrix.h"
#include "aura/tools/json.h"

using namespace aura;

class AlgoSample0Impl : public AlgoImpl
{
public:
    AlgoSample0Impl(Context *ctx) : AlgoImpl(ctx, "AlgoSample0")
    {
        m_graph.MakeNode<Arithmetic>("arithmetic_0", OpTarget::Opencl());
        Node *node = m_graph.MakeNode<Arithmetic>("arithmetic_1", OpTarget::None());
        m_graph.MakeNodes(node, {"arithmetic_2", "arithmetic_3"});
    }

    Status SetArgs(const Array *src0, const Array *src1, const Array *src2, Array *dst)
    {
        if (MI_NULL == src0 || MI_NULL == src1 || MI_NULL == src2 || MI_NULL == dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "nullptr");
            return Status::ERROR;
        }

        if (!src0->IsValid() || !src1->IsValid() ||
            !src2->IsValid() || !dst->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            return Status::ERROR;
        }

        m_src0 = src0;
        m_src1 = src1;
        m_src2 = src2;
        m_dst = dst;
        return Status::OK;
    }

    Status Initialize() override
    {
        AURA_LOGD(m_ctx, AURA_TAG, "AlgoSample0 Initialize\n");
        return Status::OK;
    }

    Status Run() override
    {
        Status ret = Status::OK;

        if (MI_NULL == m_src0 || MI_NULL == m_src1 || MI_NULL == m_src2 || MI_NULL == m_dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        Mat *dst_sub0 = m_graph.CreateMat("dst_sub0", m_src0->GetElemType(), m_src0->GetSizes());
        Mat *dst_sub1 = m_graph.CreateMat("dst_sub1", m_src0->GetElemType(), m_src0->GetSizes());
        Mat *dst_add  = m_graph.CreateMat("dst_add", m_src0->GetElemType(), m_src0->GetSizes());

        if (MI_NULL == dst_sub0 || MI_NULL == dst_sub1 || MI_NULL == dst_add)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            m_graph.DeleteArray(&dst_sub0, &dst_sub1, &dst_add);
            return Status::ERROR;
        }

        if (!dst_sub0->IsValid() || !dst_sub1->IsValid() || !dst_add->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            goto EXIT;
        }

        m_graph["arithmetic_0"].AsyncCall<Arithmetic>(m_src1, m_src1, dst_sub0, ArithmOpType::SUB);
        ret  = m_graph["arithmetic_1"].SyncCall<Arithmetic>(m_src2, m_src2, dst_sub1, ArithmOpType::SUB);
        ret |= m_graph.Barrier();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

        ret |= m_graph["arithmetic_2"].SyncCall<Arithmetic>(dst_sub0, dst_sub1, dst_add, ArithmOpType::ADD);

        ret |= m_graph["arithmetic_3"].SyncInitialize<Arithmetic>(m_src0, dst_add, m_dst, ArithmOpType::ADD);
        ret |= m_graph["arithmetic_3"].SyncRun();
        ret |= m_graph["arithmetic_3"].SyncDeInitialize();

EXIT:
        m_graph.DeleteArray(&dst_sub0, &dst_sub1, &dst_add);
        AURA_RETURN(m_ctx, ret);
    }

    Status DeInitialize() override
    {
        AURA_LOGD(m_ctx, AURA_TAG, "AlgoSample0 DeInitialize\n");
        return Status::OK;
    }

    AURA_VOID Dump(const std::string &prefix) const override
    {
        JsonWrapper json_wrapper(m_ctx, prefix, m_name);
        std::vector<std::string> names  = {"src0", "src1", "src2", "dst"};
        std::vector<const Array*> arrays = {m_src0, m_src1, m_src2, m_dst};

        if (json_wrapper.SetArray(names, arrays) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray srcs failed");
            return;
        }

        AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_src2, m_dst);
    }

    std::vector<const Array*> GetInputArrays() const override
    {
        return {m_src0, m_src1, m_src2};
    }

    std::vector<const Array*> GetOutputArrays() const override
    {
        return {m_dst};
    }

private:
    const Array *m_src0;
    const Array *m_src1;
    const Array *m_src2;
    Array *m_dst;
};

class AlgoSample0 : public Algo
{
public:
    AlgoSample0(Context *ctx) : Algo(ctx)
    {
        m_impl.reset(new AlgoSample0Impl(m_ctx));
    }

    Status SetArgs(const Array *src0, const Array *src1, const Array *src2, Array *dst)
    {
        if (MI_NULL == src0 || MI_NULL == src1 || MI_NULL == src2 || MI_NULL == dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        if (!src0->IsValid() || !src1->IsValid() ||
            !src2->IsValid() || !dst->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            return Status::ERROR;
        }

        AlgoSample0Impl *algo_sample0_impl = dynamic_cast<AlgoSample0Impl*>(m_impl.get());
        if (MI_NULL == algo_sample0_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        Status ret = algo_sample0_impl->SetArgs(src0, src1, src2, dst);
        AURA_RETURN(m_ctx, ret);
    }
};

class AlgoSampleImpl : public AlgoImpl
{
public:
    AlgoSampleImpl(Context *ctx) : AlgoImpl(ctx, "AlgoSample")
    {
        m_graph.MakeNode<Gaussian>("gaussian_0", OpTarget::Hvx());
        m_graph.MakeNode<Gaussian>("gaussian_1", OpTarget::Opencl());
        m_graph.MakeNode<AlgoSample0>("algo_sample0");
        m_graph.MakeNode<Function>("add_weighted0", OpTarget::None());
        m_graph.MakeNode<Function>("add_weighted1", OpTarget::None());
    }

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma)
    {
        if (MI_NULL == src || MI_NULL == dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "nullptr");
            return Status::ERROR;
        }

        if (!src->IsValid() || !dst->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            return Status::ERROR;
        }

        m_src = src;
        m_dst = dst;
        m_ksize = ksize;
        m_sigma = sigma;
        return Status::OK;
    }

    Status Initialize() override
    {
        AURA_LOGD(m_ctx, AURA_TAG, "AlgoSample Initialize\n");
        return Status::OK;
    }

    Status Run() override
    {
        Status ret = Status::OK;

        if (MI_NULL == m_src || MI_NULL == m_dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        Mat *dst0 = m_graph.CreateMat("dst0", m_src->GetElemType(), m_src->GetSizes());
        Mat *dst1 = m_graph.CreateMat("dst1", m_src->GetElemType(), m_src->GetSizes());
        Mat *dst2 = m_graph.CreateMat("dst2", m_src->GetElemType(), m_src->GetSizes());
        Mat *dst3 = m_graph.CreateMat("dst3", m_src->GetElemType(), m_src->GetSizes());
        std::vector<const Mat*> add_weighted_srcs = {dst0, dst1};

        if (MI_NULL == dst0 || MI_NULL == dst1 || MI_NULL == dst2 || MI_NULL == dst3)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        auto func = [&](MI_S32 loop_cnt) -> Status
        {
            for (MI_S32 i = 0; i < loop_cnt; i++)
            {
                ret = m_graph["gaussian_0"].SyncCall<Gaussian>(m_src, dst0, m_ksize, m_sigma);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "SyncCall failed");
                    return Status::ERROR;
                }
            }
            return Status::OK;
        };

        if (!dst0->IsValid() || !dst1->IsValid() || !dst2->IsValid() || !dst3->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            goto EXIT;
        }

        ////////////////////////////////////
        m_graph.AsyncRun(func, 2);
        ret = m_graph["gaussian_1"].SyncCall<Gaussian>(m_src, dst1, m_ksize, m_sigma);
        ret |= m_graph.Barrier();

        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

        m_graph["add_weighted0"].SetInputArrays(add_weighted_srcs);
        m_graph["add_weighted0"].SetOutputArrays(dst2);
        m_graph["add_weighted0"].BindDump(&AlgoSampleImpl::AddWeightedDump, this, std::cref(add_weighted_srcs), dst2, 0.5, 0.4, 3);
        m_graph["add_weighted0"].AsyncCall<Function>(&AlgoSampleImpl::AddWeighted, this, std::cref(add_weighted_srcs), dst2, 0.5, 0.4, 3);

        m_graph["add_weighted1"].SetInputArrays(add_weighted_srcs);
        m_graph["add_weighted1"].SetOutputArrays(dst3);
        m_graph["add_weighted1"].BindDump(&AlgoSampleImpl::AddWeightedDump, this, std::cref(add_weighted_srcs), dst3, 0.4, 0.2, 2);
        ret = m_graph["add_weighted1"].SyncCall<Function>(&AlgoSampleImpl::AddWeighted, this, std::cref(add_weighted_srcs), dst3, 0.4, 0.2, 2);
        ret |= m_graph.Barrier();

        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

        ///////////////////////////////////////
        m_graph["algo_sample0"].AsyncInitialize<AlgoSample0>(m_src, dst2, dst3, m_dst);
        ret = m_graph.Barrier();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

        m_graph["algo_sample0"].AsyncRun();
        ret = m_graph.Barrier();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

        m_graph["algo_sample0"].AsyncDeInitialize();
        ret = m_graph.Barrier();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Barrier failed");
            goto EXIT;
        }

EXIT:
        m_graph.DeleteArray(&dst0, &dst1, &dst2, &dst3);
        AURA_RETURN(m_ctx, ret);
    }

    Status DeInitialize() override
    {
        AURA_LOGD(m_ctx, AURA_TAG, "AlgoSample DeInitialize\n");
        return Status::OK;
    }

    std::string ToString() const override
    {
        std::string op_string;
        op_string += "ksize : " + std::to_string(m_ksize) + " sigma : " + std::to_string(m_sigma);
        return op_string;
    }

    AURA_VOID Dump(const std::string &prefix) const override
    {
        JsonWrapper json_wrapper(m_ctx, prefix, m_name);
        std::vector<std::string> names  = {"src", "dst"};
        std::vector<const Array*> arrays = {m_src, m_dst};

        if (json_wrapper.SetArray(names, arrays) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray srcs failed");
            return;
        }

        AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_sigma);
    }

    std::vector<const Array*> GetInputArrays() const override
    {
        return {m_src};
    }

    std::vector<const Array*> GetOutputArrays() const override
    {
        return {m_dst};
    }

private:
    Status AddWeighted(const std::vector<const Mat*> &srcs, Mat *dst, MI_F32 alpha, MI_F32 beta, MI_F32 gamma)
    {
        if (srcs.size() < 2)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the srcs size must be greater than 2");
            return Status::ERROR;
        }

        const MI_U8 *src0_data = srcs[0]->Ptr<MI_U8>(0);
        const MI_U8 *src1_data = srcs[1]->Ptr<MI_U8>(0);
        MI_U8 *dst_data = dst->Ptr<MI_U8>(0);

        for (MI_S64 i = 0; i < dst->GetTotalBytes(); i++)
        {
            dst_data[i] = SaturateCast<MI_U8>(src0_data[i] * alpha + src1_data[i] * beta + gamma);
        }

        return Status::OK;
    }

    AURA_VOID AddWeightedDump(const std::vector<const Mat*> &srcs, Mat *dst, MI_F64 alpha, MI_F64 beta,
                            MI_F64 gamma, const std::string &prefix, const std::string &name)
    {
        JsonWrapper json_wrapper(m_ctx, prefix, name);

        std::vector<std::string> src_names  = {"src0", "src1"};

        if (json_wrapper.SetArray(src_names, srcs) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray srcs failed");
            return;
        }

        if (json_wrapper.SetArray("dst", dst) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray dst failed");
            return;
        }

        AURA_JSON_SERIALIZE(m_ctx, json_wrapper, srcs, dst, alpha, beta, gamma);
    }

private:
    const Array *m_src;
    Array *m_dst;
    MI_S32 m_ksize;
    MI_F32 m_sigma;
};

class AlgoSample : public Algo
{
public:
    AlgoSample(Context *ctx) : Algo(ctx)
    {
        m_impl.reset(new AlgoSampleImpl(m_ctx));
    }

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma)
    {
        if (MI_NULL == src || MI_NULL == dst)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        if (!src->IsValid() || !dst->IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid array");
            return Status::ERROR;
        }

        AlgoSampleImpl *algo_sample_impl = dynamic_cast<AlgoSampleImpl*>(m_impl.get());
        if (MI_NULL == algo_sample_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
            return Status::ERROR;
        }

        Status ret = algo_sample_impl->SetArgs(src, dst, ksize, sigma);
        AURA_RETURN(m_ctx, ret);
    }
};

static AURA_VOID AlgoSampleDeInit(Context *ctx, Graph **graph)
{
    Delete<Graph>(ctx, graph);
}

static Graph *AlgoSampleInit(Context *ctx)
{
    std::unordered_map<std::string, std::vector<std::string>> props;
    Graph *graph = Create<Graph>(ctx, props);
    if (MI_NULL == graph)
    {
        AURA_LOGE(ctx, AURA_TAG, "nullptr");
    }

    graph->MakeNode<AlgoSample>("algo_sample");
    if (graph->Finalize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "Finalize failed\n");
        AlgoSampleDeInit(ctx, &graph);
    }

    return graph;
}

static Status AlgoSampleRun(Context *ctx, Graph *graph, const Mat &src, Mat &dst, MI_S32 loop_id)
{
    Status ret = Status::ERROR;
    MatCmpResult cmp_result;

    ret = graph->SetTimeout(1000);
    ret = graph->SetOutputPath("/data/local/tmp", "graph_test_loop_" + std::to_string(loop_id));
    graph->AddExternalArray("src", &src);
    graph->AddExternalArray("dst", &dst);

    ret |= (*graph)["algo_sample"].SyncCall<AlgoSample>(&src, &dst, 3, 1.f);

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "graph execute failed\n");
        goto EXIT;
    }

    if (MatCompare(ctx, src, dst, cmp_result) == Status::OK)
    {
        ret = cmp_result.status ? Status::OK : Status::ERROR;
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "mat compare fail\n");
        ret = Status::ERROR;
    }

EXIT:
    ret |= graph->SaveProfiling();
    AURA_RETURN(ctx, ret);
}

NEW_TESTCASE(algos_core_graph_test)
{
    Status ret = Status::ERROR;
    Context *ctx = UnitTest::GetInstance()->GetContext();
    MatFactory factory(ctx);

    Graph *graph = AlgoSampleInit(ctx);
    if (MI_NULL == graph)
    {
        AURA_LOGE(ctx, AURA_TAG, "graph is null\n");
        goto EXIT;
    }

    for (MI_S32 i = 0; i < 10; i++)
    {
        Mat src = factory.GetRandomMat(0, 255, ElemType::U8, Sizes3(2048, 4096));
        Mat dst(ctx, ElemType::U8, Sizes3(2048, 4096));

        ret = AlgoSampleRun(ctx, graph, src, dst, i);

        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "AlgoSampleRun failed\n");
            break;
        }

        factory.PutMats(src);
    }

    AlgoSampleDeInit(ctx, &graph);

EXIT:
    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "%s\n", ctx->GetLogger()->GetErrorString().c_str());
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}