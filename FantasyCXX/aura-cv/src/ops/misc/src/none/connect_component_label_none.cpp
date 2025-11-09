#include "connect_component_label_impl.hpp"
#include "aura/runtime/worker_pool.h"

#include <unordered_map>

#define UPPER_BOUND_4_CONNECTIVITY(height, width)                  (((height * width + 1) >> 1) + 1)
#define UPPER_BOUND_8_CONNECTIVITY(height, width)                  (((height + 1) >> 1) * ((width + 1) >> 1) + 1)
#define ROUND_UP(a, b)                                             (a + b - 1 - (a + b -1) % b)
#define CHUNK_LABEL_INIT_4C(y, width)                              (((y * width) >> 1) + 1)
#define CHUNK_LABEL_INIT_8C(y, width)                              ((y >> 1) * ((width + 1) >> 1) + 1)

namespace aura
{

class UnionFindSolver
{
public:
    UnionFindSolver(Context *ctx, const DT_U32 max_length, const DT_S32 mem_type = AURA_MEM_HEAP)
    {
        m_ctx    = ctx;
        m_buffer = ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(ctx, mem_type, max_length * sizeof(DT_U32), 0));
        if (!m_buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "alloc buffer failed");
            AURA_FREE(ctx, m_buffer.m_origin);
            m_buffer.Clear();
        }
        else
        {
            m_table = m_buffer.GetData<DT_U32*>();
            m_table[0] = 0;
            m_length = 1;
        }
    }

    virtual ~UnionFindSolver()
    {
        AURA_FREE(m_ctx, m_buffer.m_origin);
        m_buffer.Clear();
    }

    virtual DT_U32 NewLabel()
    {
        m_table[m_length] = m_length;
        return m_length++;
    }

    virtual DT_U32 NewChunkLabel(DT_U32 label)
    {
        m_table[label] = label;
        return label;
    }

    DT_U32 GetLabel(DT_U32 index)
    {
        return m_table[index];
    }

    DT_U32 FindRoot(DT_U32 root)
    {
        while (m_table[root] < root)
        {
            root = m_table[root];
        }
        return root;
    }

    virtual DT_U32 Merge(DT_U32 i, DT_U32 j)
    {
        // FindRoot(i)
        while (m_table[i] < i)
        {
            i = m_table[i];
        }

        // FindRoot(j)
        while (m_table[j] < j)
        {
            j = m_table[j];
        }

        // Merge to the smaller
        if (i < j)
        {
            return m_table[j] = i;
        }
        return m_table[i] = j;
    }

    DT_U32 Flatten()
    {
        DT_U32 k = 1;
        for (DT_U32 i = 1; i < m_length; ++i)
        {
            if (m_table[i] < i)
            {
                m_table[i] = m_table[m_table[i]];
            }
            else
            {
                m_table[i] = k++;
            }
        }
        return k;
    }

    DT_VOID FlattenChunk(const DT_U32 start_row, const DT_U32 elem_nums)
    {
        for (DT_U32 i = start_row; i < (start_row + elem_nums); ++i)
        {
            if (m_table[i] < i)
            {
                m_table[i] = m_table[m_table[i]];
            }
            else
            {
                m_table[i] = m_length++;
            }
        }
    }

protected:
    Context *m_ctx;
    Buffer   m_buffer;
    DT_U32  *m_table;
    DT_U32   m_length;
};

class UFPCSolver : public UnionFindSolver
{
public:
    UFPCSolver(Context *ctx, const DT_U32 max_length, const DT_S32 mem_type = AURA_MEM_HEAP) : UnionFindSolver(ctx, max_length, mem_type)
    {}

    DT_U32 Merge(DT_U32 i, DT_U32 j) override
    {
        // FindRoot(i)
        DT_U32 root = i;
        while (m_table[root] < root)
        {
            root = m_table[root];
        }
        if (i != j)
        {
            // FindRoot(j)
            DT_U32 root_j = j;
            while (m_table[root_j] < root_j)
            {
                root_j = m_table[root_j];
            }

            if (root > root_j)
            {
                root = root_j;
            }

            // SetRoot(j, root);
            while (m_table[j] < j)
            {
                DT_U32 t = m_table[j];
                m_table[j] = root;
                j = t;
            }
            m_table[j] = root;
        }

        // SetRoot(i, root);
        while (m_table[i] < i)
        {
            DT_U32 t = m_table[i];
            m_table[i] = root;
            i = t;
        }

        m_table[i] = root;
        return root;
    }

};

class RemSpliceSolver : public UnionFindSolver
{
public:
    RemSpliceSolver(Context *ctx, const DT_U32 max_length, const DT_S32 mem_type = AURA_MEM_HEAP) : UnionFindSolver(ctx, max_length, mem_type)
    {}

    DT_U32 Merge(DT_U32 i, DT_U32 j) override
    {
        DT_U32 root_i = i, root_j = j;

        while (m_table[root_i] != m_table[root_j])
        {
            if (m_table[root_i] > m_table[root_j])
            {
                if (root_i == m_table[root_i])
                {
                    m_table[root_i] = m_table[root_j];
                    return m_table[root_i];
                }
                DT_U32 z = m_table[root_i];
                m_table[root_i] = m_table[root_j];
                root_i = z;
            }
            else
            {
                if (root_j == m_table[root_j])
                {
                    m_table[root_j] = m_table[root_i];
                    return m_table[root_i];
                }
                DT_U32 z = m_table[root_j];
                m_table[root_j] = m_table[root_i];
                root_j = z;
            }
        }

        return m_table[root_i];
    }
};

class TTASolver : public UnionFindSolver
{
public:
    TTASolver(Context *ctx, const DT_U32 max_length, const DT_S32 mem_type = AURA_MEM_HEAP) : UnionFindSolver(ctx, max_length, mem_type)
    {
        m_buffer_new = ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(ctx, mem_type, max_length * sizeof(DT_U32) * 2, 0));
        if (!m_buffer_new.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "alloc buffer failed");
            AURA_FREE(ctx, m_buffer_new.m_origin);
            m_buffer_new.Clear();
        }
        else
        {
            m_table    = m_buffer.GetData<DT_U32*>();
            m_next     = m_buffer_new.GetData<DT_U32*>();
            m_tail     = m_next + max_length;
            m_table[0] = 0;	 // First label is for background pixels
            m_length   = 1;
        }
    }

    virtual ~TTASolver()
    {
        AURA_FREE(m_ctx, m_buffer_new.m_origin);
        m_buffer_new.Clear();
    }

    DT_U32 NewLabel() override
    {
        m_table[m_length] = m_length;
        m_next[m_length]  = UINT_MAX;
        m_tail[m_length]  = m_length;
        return m_length++;
    }

    DT_U32 NewChunkLabel(DT_U32 label) override
    {
        m_table[label] = label;
        m_next[label]  = UINT_MAX;
        m_tail[label]  = label;
        return label;
    }

    DT_U32 Merge(DT_U32 u, DT_U32 v) override
    {
        // FindRoot(u);
        u = m_table[u];
        // FindRoot(v);
        v = m_table[v];

        if (u < v)
        {
            DT_U32 i = v;
            while (i != UINT_MAX)
            {
                m_table[i] = u;
                i = m_next[i];
            }

            m_next[m_tail[u]] = v;
            m_tail[u] = m_tail[v];
            return u;
        }
        else if (u > v)
        {
            DT_U32 i = u;
            while (i != UINT_MAX)
            {
                m_table[i] = v;
                i = m_next[i];
            }

            m_next[m_tail[v]] = u;
            m_tail[v] = m_tail[u];
            return v;
        }

        return u;  // equal to v
    }

private:
    Buffer  m_buffer_new;
    DT_U32 *m_next;
    DT_U32 *m_tail;

};

static std::shared_ptr<UnionFindSolver> CreateSolver(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length)
{
    std::shared_ptr<UnionFindSolver> impl;

    switch (solver_type)
    {
        case EquivalenceSolver::UNION_FIND:
        {
            impl.reset(new UnionFindSolver(ctx, max_length));
            break;
        }

        case EquivalenceSolver::UNION_FIND_PATH_COMPRESS:
        {
            impl.reset(new UFPCSolver(ctx, max_length));
            break;
        }

        case EquivalenceSolver::REM_SPLICING:
        {
            impl.reset(new RemSpliceSolver(ctx, max_length));
            break;
        }

        case EquivalenceSolver::THREE_TABLE_ARRAYS:
        {
            impl.reset(new TTASolver(ctx, max_length));
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

template <typename LabelType>
class ScanPlusUnionFindNone
{
public:
    ScanPlusUnionFindNone(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length)
    {
        m_solver = CreateSolver(ctx, solver_type, max_length);
    }

    Status operator()(Context *ctx, const Mat &img, Mat &label, ConnectivityType type = ConnectivityType::CROSS)
    {
        Status ret = Status::ERROR;

        // fisrt forward scan
        if (ConnectivityType::CROSS == type)
        {
            ret = this->FirstScan4C(img, label);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "FirstScan4C fail");
                goto EXIT;
            }
        }
        else if (ConnectivityType::SQUARE == type)
        {
            ret = this->FirstScan8C(img, label);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "FirstScan8C fail");
                goto EXIT;
            }
        }
        else
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupport Connectivity Type");
            goto EXIT;
        }

        // Second backward scan
        this->m_solver->Flatten();

        ret = this->SecondScan(label);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan fail");
            goto EXIT;
        }

    EXIT:
        AURA_RETURN(ctx, ret);
    }

protected:
    std::shared_ptr<UnionFindSolver> m_solver;

    virtual Status FirstScan4C(const Mat &img, Mat &label)
    {
        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;
        //   +-+
        //   |q|
        // +-+-+
        // |s|x|
        // +-+-+
        for (DT_S32 y = 0; y < height; ++y)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p0 = label.Ptr<LabelType>(y - 1);
            for (DT_S32 x = 0; x < width; ++x)
            {
#define CONDITION_Q (y > 0 && src_p0[x] > 0)
#define CONDITION_S (x > 0 && src_c0[x - 1] > 0)
#define CONDITION_X (src_c0[x] > 0)

#define ACTION_1 // nothing to do 
#define ACTION_2 dst_c0[x] = this->m_solver->NewLabel(); // new label
#define ACTION_3 dst_c0[x] = dst_p0[x];         // x <- q
#define ACTION_4 dst_c0[x] = dst_c0[x - 1];     // x <- s
#define ACTION_5 dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x - 1]); // x <- p + s

#include "connect_component_label/sauf_4c_decision_tree.hpp"
            }
        }
#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
        return Status::OK;
    }

    Status SecondScan(Mat &label)
    {
        const DT_S32 height = label.GetSizes().m_height;
        for (DT_S32 y = 0; y < height; ++y)
        {
            LabelType *dst_start_row     = label.Ptr<LabelType>(y);
            LabelType *const dst_end_row = dst_start_row + label.GetRowStep();
            for (; dst_start_row != dst_end_row; ++dst_start_row)
            {
                *dst_start_row = static_cast<LabelType>(this->m_solver->GetLabel(*dst_start_row));
            }
        }
        return Status::OK;
    }

private:
    Status FirstScan8C(const Mat &img, Mat &label)
    {
        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;

        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+
        for (DT_S32 y = 0; y < height; ++y)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p0 = label.Ptr<LabelType>(y - 1);
            for (DT_S32 x = 0; x < width; ++x)
            {
#define CONDITION_P    (x > 0 && y > 0 && src_p0[x - 1] > 0)
#define CONDITION_Q    (y > 0 && src_p0[x] > 0)
#define CONDITION_R    (x < width - 1 && y > 0 && src_p0[x + 1] > 0)
#define CONDITION_S    (x > 0 && src_c0[x - 1] > 0)
#define CONDITION_X    (src_c0[x] > 0)

#define ACTION_1 // nothing to do 
#define ACTION_2 dst_c0[x] = this->m_solver->NewLabel(); // new label
#define ACTION_3 dst_c0[x] = dst_p0[x - 1];     // x <- p
#define ACTION_4 dst_c0[x] = dst_p0[x];         // x <- q
#define ACTION_5 dst_c0[x] = dst_p0[x + 1];     // x <- y
#define ACTION_6 dst_c0[x] = dst_c0[x - 1];     // x <- s
#define ACTION_7 dst_c0[x] = this->m_solver->Merge(dst_p0[x - 1], dst_p0[x + 1]); // x <- p + y
#define ACTION_8 dst_c0[x] = this->m_solver->Merge(dst_c0[x - 1], dst_p0[x + 1]); // x <- s + y

#include "connect_component_label/sauf_8c_decision_tree.hpp"
            }
        }
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X
        return Status::OK;
    }

};

template <typename LabelType>
class BlockBasedDecisionTreeNone
{
public:
    BlockBasedDecisionTreeNone(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length)
    {
        m_solver = CreateSolver(ctx, solver_type, max_length);
    }

    virtual Status operator()(Context *ctx, const Mat &img, Mat &label)
    {
        Status ret = Status::ERROR;

        // first forward scan
        ret = this->FirstScan(img, label);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FirstScan fail");
            goto EXIT;
        }

        // Second backward scan
        this->m_solver->Flatten();

        ret = this->SecondScan(img, label);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan fail");
            goto EXIT;
        }

    EXIT:
        AURA_RETURN(ctx, ret);
    }

protected:
    std::shared_ptr<UnionFindSolver> m_solver;

    virtual Status FirstScan(const Mat &img, Mat &label)
    {
        const DT_S32 height = label.GetSizes().m_height;
        const DT_S32 width  = label.GetSizes().m_width;

        DT_S32 y = 0;

        // fisrt forward scan
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q y|s t|
        // +---+---+
        for (y = 0; y < height; y += 2)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
            const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);
            for (DT_S32 x = 0; x < width; x += 2)
            {
#define CONDITION_B ((x - 1) >= 0 && (y - 2) >= 0 && src_p1[x - 1] > 0)
#define CONDITION_C ((y - 2) >= 0 && src_p1[x] > 0)
#define CONDITION_D ((x + 1) < width && (y - 2) >= 0 && src_p1[x + 1] > 0)
#define CONDITION_E ((x + 2) < width && (y - 2) >= 0 && src_p1[x + 2] > 0)

#define CONDITION_G ((x - 2) >= 0 && (y - 1) >= 0 && src_p0[x - 2] > 0)
#define CONDITION_H ((x - 1) >= 0 && (y - 1) >= 0 && src_p0[x - 1] > 0)
#define CONDITION_I ((y - 1) >= 0 && src_p0[x] > 0)
#define CONDITION_J ((x + 1) < width && (y - 1) >= 0 && src_p0[x + 1] > 0)
#define CONDITION_K ((x + 2) < width && (y - 1) >= 0 && src_p0[x + 2] > 0)

#define CONDITION_M ((x - 2) >= 0 && src_c0[x - 2] > 0)
#define CONDITION_N ((x - 1) >= 0 && src_c0[x - 1] > 0)
#define CONDITION_O (src_c0[x] > 0)
#define CONDITION_P ((x + 1) < width && src_c0[x + 1] > 0)

#define CONDITION_R ((x - 1) >= 0 && (y + 1) < height && src_n0[x - 1] > 0)
#define CONDITION_S ((y + 1) < height && src_n0[x] > 0)
#define CONDITION_T ((x + 1) < width && (y + 1) < height && src_n0[x + 1] > 0)

#define ACTION_1  dst_c0[x] = 0; continue; 
#define ACTION_2  dst_c0[x] = this->m_solver->NewLabel(); continue;
#define ACTION_3  dst_c0[x] = dst_p1[x - 2]; continue;
#define ACTION_4  dst_c0[x] = dst_p1[x];     continue;
#define ACTION_5  dst_c0[x] = dst_p1[x + 2]; continue;
#define ACTION_6  dst_c0[x] = dst_c0[x - 2]; continue;
#define ACTION_7  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]);     continue;
#define ACTION_8  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]); continue;
#define ACTION_9  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_c0[x - 2]); continue;
#define ACTION_10 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]); continue;
#define ACTION_11 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_c0[x - 2]); continue;
#define ACTION_12 dst_c0[x] = this->m_solver->Merge(dst_p1[x + 2], dst_c0[x - 2]); continue;
#define ACTION_13
#define ACTION_14 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_c0[x - 2]); continue;
#define ACTION_15 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]), dst_c0[x - 2]); continue;
#define ACTION_16 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]), dst_c0[x - 2]); continue;
                    
#include "connect_component_label/bbdt_8c_decision_tree.hpp"
            }
        }
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
        return Status::OK;
    }

    Status SecondScan(const Mat &img, Mat &label)
    {
        const DT_S32 height  = label.GetSizes().m_height;
        const DT_S32 width   = label.GetSizes().m_width;
        const DT_S32 e_rows  = height & 0xfffffffe;
        const DT_BOOL o_rows = (height & 1) == 1;
        const DT_S32 e_cols  = width & 0xfffffffe;
        const DT_BOOL o_cols = (width & 1) == 1;
        DT_S32 y = 0;
        
        for (; y < e_rows; y += 2)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_n0 = src_c0 + img.GetRowStep();
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_n0 = dst_c0 + label.GetRowStep();
            DT_S32 x = 0;
            for (; x < e_cols; x += 2)
            {
                LabelType label_val = dst_c0[x];
                if (label_val > 0)
                {
                    label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                    if (src_c0[x] > 0)
                    {
                        dst_c0[x] = label_val;
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                    if (src_c0[x + 1] > 0)
                    {
                        dst_c0[x + 1] = label_val;
                    }
                    else
                    {
                        dst_c0[x + 1] = 0;
                    }
                    if (src_n0[x] > 0)
                    {
                        dst_n0[x] = label_val;
                    }
                    else
                    {
                        dst_n0[x] = 0;
                    }
                    if (src_n0[x + 1] > 0)
                    {
                        dst_n0[x + 1] = label_val;
                    }
                    else
                    {
                        dst_n0[x + 1] = 0;
                    }
                }
                else
                {
                    dst_c0[x] = 0;
                    dst_c0[x + 1] = 0;
                    dst_n0[x] = 0;
                    dst_n0[x + 1] = 0;
                }
            }

            // Last column if the number of columns is odd
            if (o_cols)
            {
                DT_S32 label_val = dst_c0[x];
                if (label_val > 0)
                {
                    label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                    if (src_c0[x] > 0)
                    {
                        dst_c0[x] = label_val;
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                    if (src_n0[x] > 0)
                    {
                        dst_n0[x] = label_val;
                    }
                    else
                    {
                        dst_n0[x] = 0;
                    }
                }
                else
                {
                    dst_c0[x] = 0;
                    dst_n0[x] = 0;
                }
            }
        }

        // Last row if the number of rows is odd
        if (o_rows)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            LabelType *const dst_c0 = label.Ptr<LabelType>(y);
            DT_S32 x = 0;
            for (; x < e_cols; x += 2)
            {
                DT_S32 label_val = dst_c0[x];
                if (label_val > 0)
                {
                    label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                    if (src_c0[x] > 0)
                    {
                        dst_c0[x] = label_val;
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                    if (src_c0[x + 1] > 0)
                    {
                        dst_c0[x + 1] = label_val;
                    }
                    else
                    {
                        dst_c0[x + 1] = 0;
                    }
                }
                else
                {
                    dst_c0[x] = 0;
                    dst_c0[x + 1] = 0;
                }
            }

            // Last column if the number of columns is odd
            if (o_cols)
            {
                DT_S32 label_val = dst_c0[x];
                if (label_val > 0)
                {
                    label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                    if (src_c0[x] > 0)
                    {
                        dst_c0[x] = label_val;
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                }
                else
                {
                    dst_c0[x] = 0;
                }
            }
        }

        return Status::OK;
    }

};

template <typename LabelType>
class Spaghetti4CNone : public ScanPlusUnionFindNone<LabelType>
{
public:
    Spaghetti4CNone(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length) : ScanPlusUnionFindNone<LabelType>(ctx, solver_type, max_length)
    {}

private:
    Status FirstScan4C(const Mat &img, Mat &label) override
    {
        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width = img.GetSizes().m_width;

#define CONDITION_Q (src_p0[x] > 0)
#define CONDITION_S (src_c0[x - 1] > 0)
#define CONDITION_X (src_c0[x] > 0)

#define ACTION_1 dst_c0[x] = 0;
#define ACTION_2 dst_c0[x] = this->m_solver->NewLabel();
#define ACTION_3 dst_c0[x] = dst_p0[x]; // x <- q
#define ACTION_4 dst_c0[x] = dst_c0[x - 1];// x <- s
#define ACTION_5 dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x - 1]); // x <- q + s

        // first row
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(0);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(0);
            DT_S32 x = -1;

#include "connect_component_label/spaghetti_4c_firstline_graph.hpp"
        }

        // rest rows
        for (DT_S32 y = 1; y < height; ++y)
        {
            const DT_U8 *src_c0     = img.Ptr<DT_U8>(y);
            const DT_U8 *src_p0     = img.Ptr<DT_U8>(y - 1);
            LabelType *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType *const dst_p0 = label.Ptr<LabelType>(y - 1);
            DT_S32 x = -1;

#include "connect_component_label/spaghetti_4c_singleline_graph.hpp"
        }
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X
        return Status::OK;
    }

};

template <typename LabelType>
class Spaghetti8CNone : public BlockBasedDecisionTreeNone<LabelType>
{
public:
    Spaghetti8CNone(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length) : BlockBasedDecisionTreeNone<LabelType>(ctx, solver_type, max_length)
    {}

private:
    Status FirstScan(const Mat &img, Mat &label) override
    {
        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;

        const DT_S32  e_rows = height & 0xfffffffe;
        const DT_BOOL o_rows = (height & 1) == 1;

        // first scan
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q y|s t|
        // +---+---+
#define CONDITION_B (src_p1[x - 1] > 0)
#define CONDITION_C (src_p1[x] > 0)
#define CONDITION_D (src_p1[x + 1] > 0)
#define CONDITION_E (src_p1[x + 2] > 0)
#define CONDITION_G (src_p0[x - 2] > 0)
#define CONDITION_H (src_p0[x - 1] > 0)
#define CONDITION_I (src_p0[x] > 0)
#define CONDITION_J (src_p0[x + 1] > 0)
#define CONDITION_K (src_p0[x + 2] > 0)
#define CONDITION_M (src_c0[x - 2] > 0)
#define CONDITION_N (src_c0[x - 1] > 0)
#define CONDITION_O (src_c0[x] > 0)
#define CONDITION_P (src_c0[x + 1] > 0)
#define CONDITION_R (src_n0[x - 1] > 0)
#define CONDITION_S (src_n0[x] > 0)
#define CONDITION_T (src_n0[x + 1] > 0)

#define ACTION_1  dst_c0[x] = 0;
#define ACTION_2  dst_c0[x] = this->m_solver->NewLabel();
#define ACTION_3  dst_c0[x] = dst_p1[x - 2];
#define ACTION_4  dst_c0[x] = dst_p1[x];
#define ACTION_5  dst_c0[x] = dst_p1[x + 2];
#define ACTION_6  dst_c0[x] = dst_c0[x - 2];
#define ACTION_7  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]);
#define ACTION_8  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]);
#define ACTION_9  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_c0[x - 2]);
#define ACTION_10 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]);
#define ACTION_11 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_c0[x - 2]);
#define ACTION_12 dst_c0[x] = this->m_solver->Merge(dst_p1[x + 2], dst_c0[x - 2]);
#define ACTION_13 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_p1[x + 2]);
#define ACTION_14 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_c0[x - 2]);
#define ACTION_15 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]), dst_c0[x - 2]);
#define ACTION_16 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]), dst_c0[x - 2]);

        if (1 == height)
        {
            // Single line
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(0);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(0);
            DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_singleline_graph.hpp"
        }
        else
        {
            // First couple of lines
            {
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(0);
                const DT_U8 *const src_n0 = img.Ptr<DT_U8>(1);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(0);
                DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_firstline_graph.hpp"
            }

            for (DT_S32 y = 2; y < e_rows; y += 2)
            {
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
                const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
                const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);

                DT_S32 x = -2;
                goto tree_0;

#include "connect_component_label/spaghetti_8c_graph.hpp"
            }

            // Last line (in case the rows are odd)
            if (o_rows)
            {
                DT_S32 y = height - 1;
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
                const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);
                DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_lastline_graph.hpp"
            }
        }

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
        return Status::OK;
    }

};

template <typename LabelType>
class ScanPlusUnionFindParallel
{
public:
    ScanPlusUnionFindParallel(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length)
    {
        m_solver = CreateSolver(ctx, solver_type, max_length);
    }

    Status operator()(Context *ctx, const Mat &img, Mat &label, ConnectivityType type)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        Status ret = Status::ERROR;

        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;

        DT_S32 step = height >> 5;
        step = Clamp<DT_S32>(step, 16, 256);
        step = ((step & 1) == 1) ? (step + 1) : step; // make sure even
        std::vector<DT_S32> chunks(ROUND_UP(height, step));

        if (ConnectivityType::CROSS == type)
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&ScanPlusUnionFindParallel::FirstScan4C,
                                  this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                  std::placeholders::_4, std::placeholders::_5, std::placeholders::_6), std::cref(img),
                                  std::ref(label), (DT_S32*)(chunks.data()), step);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "FirstScan4C parallel_for fail");
                goto EXIT;
            }

            ret = this->MergeLabels4C(label, (DT_S32*)(chunks.data()));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "MergeLabels4C parallel_for fail");
                goto EXIT;
            }

            for (DT_S32 i = 0; i < height; i = chunks[i])
            {
                this->m_solver->FlattenChunk(CHUNK_LABEL_INIT_4C(i, width), chunks[i + 1]);
            }
        }
        else if (ConnectivityType::SQUARE == type)
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&ScanPlusUnionFindParallel::FirstScan8C,
                                  this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                  std::placeholders::_4, std::placeholders::_5, std::placeholders::_6), std::cref(img),
                                  std::ref(label), (DT_S32*)(chunks.data()), step);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "FirstScan8C parallel_for fail");
                goto EXIT;
            }

            ret = this->MergeLabels8C(label, (DT_S32*)(chunks.data()));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "MergeLabels8C parallel_for fail");
                goto EXIT;
            }

            for (DT_S32 i = 0; i < height; i = chunks[i])
            {
                this->m_solver->FlattenChunk(CHUNK_LABEL_INIT_8C(i, width), chunks[i + 1]);
            }
        }
        else
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupport Connectivity Type");
            goto EXIT;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&ScanPlusUnionFindParallel::SecondScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4), std::ref(label), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan parallel_for fail");
            goto EXIT;
        }

    EXIT:
        AURA_RETURN(ctx, ret);
    }

protected:
    std::shared_ptr<UnionFindSolver> m_solver;

    virtual Status FirstScan4C(const Mat &img, Mat &label, DT_S32 *chunks, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk)
    {
        const DT_S32 width     = img.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        chunks[start_row] = end_row;
        const LabelType label_init = CHUNK_LABEL_INIT_4C(start_row, width);
        LabelType label_last = label_init;

        for (DT_S32 y = start_row; y != end_row; ++y)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p0 = label.Ptr<LabelType>(y - 1);
            for (DT_S32 x = 0; x < width; ++x)
            {
#define CONDITION_Q (y > start_row && src_p0[x] > 0)
#define CONDITION_S (x > 0 && src_c0[x - 1] > 0)
#define CONDITION_X (src_c0[x] > 0)

#define ACTION_1 // nothing to do 
#define ACTION_2 dst_c0[x] = (LabelType)this->m_solver->NewChunkLabel(label_last++);
#define ACTION_3 dst_c0[x] = dst_p0[x]; // x <- q
#define ACTION_4 dst_c0[x] = dst_c0[x - 1]; // x <- s
#define ACTION_5 dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x - 1]); // x <- p + s

#include "connect_component_label/sauf_4c_decision_tree.hpp"
            }
        }

        chunks[start_row + 1] = (DT_S32)label_last - (DT_S32)label_init;

#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
        return Status::OK;
    }

    Status MergeLabels4C(Mat &label, const DT_S32 *chunks)
    {
        const DT_S32 height = label.GetSizes().m_height;
        const DT_S32 width  = label.GetSizes().m_width;

        for (DT_S32 y = chunks[0]; y < height; y = chunks[y])
        {
            LabelType *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType *const dst_p0 = label.Ptr<LabelType>(y - 1);

            for (DT_S32 x = 0; x < width; ++x)
            {
                if (dst_c0[x] > 0 && dst_p0[x] > 0)
                {
                    dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x]);
                }
            }
        }
        return Status::OK;
    }

    Status SecondScan(Mat &label, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk)
    {
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, label.GetSizes().m_height);

        for (DT_S32 y = start_row; y < end_row; ++y)
        {
            LabelType *dst_start_row     = label.Ptr<LabelType>(y);
            LabelType *const dst_end_row = dst_start_row + label.GetRowStep();
            for (; dst_start_row != dst_end_row; ++dst_start_row)
            {
                *dst_start_row = static_cast<LabelType>(this->m_solver->GetLabel(*dst_start_row));
            }
        }
        return Status::OK;
    }

private:
    Status FirstScan8C(const Mat &img, Mat &label, DT_S32 *chunks, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk)
    {
        const DT_S32 width     = img.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        chunks[start_row] = end_row;
        const LabelType label_init = CHUNK_LABEL_INIT_8C(start_row, width);
        LabelType label_last = label_init;

        for (DT_S32 y = start_row; y < end_row; ++y)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p0 = label.Ptr<LabelType>(y - 1);
            for (DT_S32 x = 0; x < width; ++x)
            {
#define CONDITION_P    (x > 0 && y > start_row && src_p0[x - 1] > 0)
#define CONDITION_Q    (y > start_row && src_p0[x] > 0)
#define CONDITION_R    (x < (width - 1) && y > start_row && src_p0[x + 1] > 0)
#define CONDITION_S    (x > 0 && src_c0[x - 1] > 0)
#define CONDITION_X    (src_c0[x] > 0)

#define ACTION_1 // nothing to do 
#define ACTION_2 dst_c0[x] = (LabelType)this->m_solver->NewChunkLabel(label_last++);
#define ACTION_3 dst_c0[x] = dst_p0[x - 1];  // x <- p
#define ACTION_4 dst_c0[x] = dst_p0[x];      // x <- q
#define ACTION_5 dst_c0[x] = dst_p0[x + 1];  // x <- y
#define ACTION_6 dst_c0[x] = dst_c0[x - 1];  // x <- s
#define ACTION_7 dst_c0[x] = this->m_solver->Merge(dst_p0[x - 1], dst_p0[x + 1]); // x <- p + y
#define ACTION_8 dst_c0[x] = this->m_solver->Merge(dst_c0[x - 1], dst_p0[x + 1]); // x <- s + y

#include "connect_component_label/sauf_8c_decision_tree.hpp"
            }
        }
        chunks[start_row + 1] = (DT_S32)label_last - (DT_S32)label_init;
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X
        return Status::OK;
    }

    Status MergeLabels8C(Mat &label, const DT_S32 *chunks)
    {
        const DT_S32 height = label.GetSizes().m_height;
        const DT_S32 width  = label.GetSizes().m_width;

        for (DT_S32 y = chunks[0]; y < height; y = chunks[y])
        {
            LabelType *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType *const dst_p0 = label.Ptr<LabelType>(y - 1);

            for (DT_S32 x = 0; x < width; ++x)
            {
                if (dst_c0[x] > 0)
                {
                    if (x > 0 && dst_p0[x - 1] > 0)
                    {
                        dst_c0[x] = this->m_solver->Merge(dst_p0[x - 1], dst_c0[x]);
                    }
                    if (x < (width - 1) && dst_p0[x + 1] > 0)
                    {
                        dst_c0[x] = this->m_solver->Merge(dst_p0[x + 1], dst_c0[x]);
                    }
                    if (dst_p0[x] > 0)
                    {
                        dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x]);
                    }
                }
            }
        }
        return Status::OK;
    }

};

template <typename LabelType>
class BlockBasedDecisionTreeParallel
{
public:
    BlockBasedDecisionTreeParallel(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length)
    {
        m_solver = CreateSolver(ctx, solver_type, max_length);
    }

    virtual Status operator()(Context *ctx, const Mat &img, Mat &label)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        Status ret = Status::ERROR;

        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;

        DT_S32 step = height >> 5;
        step = Clamp<DT_S32>(step, 16, 256);
        step = ((step & 1) == 1) ? (step + 1) : step; // make sure even
        std::vector<DT_S32> chunks(ROUND_UP(height, step));

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&BlockBasedDecisionTreeParallel::FirstScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4, std::placeholders::_5, std::placeholders::_6), std::cref(img),
                              std::ref(label), (DT_S32*)(chunks.data()), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FirstScan parallel_for fail");
            goto EXIT;
        }

        ret = this->MergeLabels(img, label, (DT_S32*)(chunks.data()));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MergeLabels parallel_for fail");
            goto EXIT;
        }

        for (DT_S32 i = 0; i < height; i = chunks[i])
        {
            this->m_solver->FlattenChunk(CHUNK_LABEL_INIT_8C(i, width), chunks[i + 1]);
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&BlockBasedDecisionTreeParallel::SecondScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4, std::placeholders::_5), std::cref(img), std::ref(label), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan parallel_for fail");
            goto EXIT;
        }

    EXIT:
        AURA_RETURN(ctx, ret);
    }

protected:
    std::shared_ptr<UnionFindSolver> m_solver;

    virtual Status FirstScan(const Mat &img, Mat &label, DT_S32 *chunks, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk)
    {
        const DT_S32 width     = img.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        DT_S32 y = start_row;
        chunks[start_row] = end_row;
        const LabelType label_init = CHUNK_LABEL_INIT_8C(start_row, width);
        LabelType label_last = label_init;

        for (; y < end_row; y += 2)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
            const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);
            for (DT_S32 x = 0; x < width; x += 2)
            {
#define CONDITION_B ((x - 1) >= 0 && (y - 2) >= start_row && src_p1[x - 1] > 0)
#define CONDITION_C ((y - 2) >= start_row && src_p1[x] > 0)
#define CONDITION_D ((x + 1) < width && (y - 2) >= start_row && src_p1[x + 1] > 0)
#define CONDITION_E ((x + 2) < width && (y - 2) >= start_row && src_p1[x + 2] > 0)

#define CONDITION_G ((x - 2) >= 0 && (y - 1) >= start_row && src_p0[x - 2] > 0)
#define CONDITION_H ((x - 1) >= 0 && (y - 1) >= start_row && src_p0[x - 1] > 0)
#define CONDITION_I ((y - 1) >= start_row && src_p0[x] > 0)
#define CONDITION_J ((x + 1) < width && (y - 1) >= start_row && src_p0[x + 1] > 0)
#define CONDITION_K ((x + 2) < width && (y - 1) >= start_row && src_p0[x + 2] > 0)

#define CONDITION_M ((x - 2) >= 0 && src_c0[x - 2] > 0)
#define CONDITION_N ((x - 1) >= 0 && src_c0[x - 1] > 0)
#define CONDITION_O (src_c0[x] > 0)
#define CONDITION_P ((x + 1) < width && src_c0[x + 1] > 0)

#define CONDITION_R ((x - 1) >= 0 && (y + 1) < end_row && src_n0[x - 1] > 0)
#define CONDITION_S ((y + 1) < end_row && src_n0[x] > 0)
#define CONDITION_T ((x + 1) < width && (y + 1) < end_row && src_n0[x + 1] > 0)

#define ACTION_1  dst_c0[x] = 0; continue; 
#define ACTION_2  dst_c0[x] = (LabelType)this->m_solver->NewChunkLabel(label_last++); continue;
#define ACTION_3  dst_c0[x] = dst_p1[x - 2]; continue;
#define ACTION_4  dst_c0[x] = dst_p1[x];     continue;
#define ACTION_5  dst_c0[x] = dst_p1[x + 2]; continue;
#define ACTION_6  dst_c0[x] = dst_c0[x - 2]; continue;
#define ACTION_7  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]);     continue;
#define ACTION_8  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]); continue;
#define ACTION_9  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_c0[x - 2]); continue;
#define ACTION_10 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]); continue;
#define ACTION_11 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_c0[x - 2]); continue;
#define ACTION_12 dst_c0[x] = this->m_solver->Merge(dst_p1[x + 2], dst_c0[x - 2]); continue;
#define ACTION_13
#define ACTION_14 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_c0[x - 2]); continue;
#define ACTION_15 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]), dst_c0[x - 2]); continue;
#define ACTION_16 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]), dst_c0[x - 2]); continue;
   
#include "connect_component_label/bbdt_8c_decision_tree.hpp"
            }
        }
        chunks[start_row + 1] = (DT_S32)label_last - (DT_S32)label_init;
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
        return Status::OK;
    }

    Status MergeLabels(const Mat &img, Mat &label, const DT_S32 *chunks)
    {
        const DT_S32 height = label.GetSizes().m_height;
        const DT_S32 width  = label.GetSizes().m_width;

        for (DT_S32 y = chunks[0]; y < height; y = chunks[y])
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);

            for (DT_S32 x = 0; x < width; x += 2)
            {
                if (dst_c0[x] > 0)
                {
                    if (x > 1 && dst_p1[x - 2] > 0 && src_c0[x] > 0 && src_p0[x - 1] > 0)
                    {
                        dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_c0[x]);
                    }
                    if (dst_p1[x] > 0)
                    {
                        if (x < (width - 1))
                        {
                            if ((src_c0[x] > 0 && src_p0[x]     > 0) || (src_c0[x + 1] > 0 && src_p0[x]     > 0) ||
                                (src_c0[x] > 0 && src_p0[x + 1] > 0) || (src_c0[x + 1] > 0 && src_p0[x + 1] > 0))
                            {
                                dst_c0[x] = this->m_solver->Merge(dst_p1[x], dst_c0[x]);
                            }
                        }
                        else // x == width -1
                        {
                            if ((src_c0[x] > 0 && src_p0[x] > 0))
                            {
                                dst_c0[x] = this->m_solver->Merge(dst_p1[x], dst_c0[x]);
                            }
                        }
                    }

                    if (x < (width - 2) && dst_p1[x + 2] > 0 && src_c0[x + 1] > 0 && src_p0[x + 2] > 0)
                    {
                        dst_c0[x] = this->m_solver->Merge(dst_p1[x + 2], dst_c0[x]);
                    }
                }
            }
        }
        return Status::OK;
    }

    Status SecondScan(const Mat &img, Mat &label, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk)
    {
        const DT_S32 height    = label.GetSizes().m_height;
        const DT_S32 width     = label.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        // only both even
        for (DT_S32 y = start_row; y < Min(end_row, height & -2); y += 2)
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
            const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType   *const dst_n0 = label.Ptr<LabelType>(y + 1);
            for (DT_S32 x = 0; x < (width & -2); x += 2)
            {
                LabelType label_val = dst_c0[x];
                if (label_val > 0)
                {
                    label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                    if (src_c0[x] > 0)
                    {
                        dst_c0[x] = label_val;
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                    if (src_c0[x + 1] > 0)
                    {
                        dst_c0[x + 1] = label_val;
                    }
                    else
                    {
                        dst_c0[x + 1] = 0;
                    }
                    if (src_n0[x] > 0)
                    {
                        dst_n0[x] = label_val;
                    }
                    else
                    {
                        dst_n0[x] = 0;
                    }
                    if (src_n0[x + 1] > 0)
                    {
                        dst_n0[x + 1] = label_val;
                    }
                    else
                    {
                        dst_n0[x + 1] = 0;
                    }
                }
                else
                {
                    dst_c0[x + 1] = 0;
                    dst_n0[x] = 0;
                    dst_n0[x + 1] = 0;
                }
            }
        }

        if (height == end_row) // process rest when last thread
        {
            if (height & 1) // handle rest odd row
            {
                if (width & 1) // handle rest odd row and col
                {
                    for (DT_S32 y = 0; y < (height & -2); y += 2)
                    {
                        const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                        const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
                        LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                        LabelType   *const dst_n0 = label.Ptr<LabelType>(y + 1);
                        LabelType label_val = dst_c0[width - 1];
                        if (label_val > 0)
                        {
                            label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                            if (src_c0[width - 1] > 0)
                            {
                                dst_c0[width - 1] = label_val;
                            }
                            else
                            {
                                dst_c0[width - 1] = 0;
                            }
                            if (src_n0[width - 1] > 0)
                            {
                                dst_n0[width - 1] = label_val;
                            }
                            else
                            {
                                dst_n0[width - 1] = 0;
                            }
                        }
                        else
                        {
                            dst_n0[width - 1] = 0;
                        }
                    }

                    const DT_U8 *const src_c0 = img.Ptr<DT_U8>(height - 1);
                    LabelType   *const dst_c0 = label.Ptr<LabelType>(height - 1);
                    for (DT_S32 x = 0; x < (width & -2); x += 2)
                    {
                        LabelType label_val = dst_c0[x];
                        if (label_val > 0)
                        {
                            label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                            if (src_c0[x] > 0)
                            {
                                dst_c0[x] = label_val;
                            }
                            else
                            {
                                dst_c0[x] = 0;
                            }
                            if (src_c0[x + 1] > 0)
                            {
                                dst_c0[x + 1] = label_val;
                            }
                            else
                            {
                                dst_c0[x + 1] = 0;
                            }
                        }
                        else
                        {
                            dst_c0[x + 1] = 0;
                        }
                    }

                    const DT_S32 x = width - 1;
                    LabelType label_val = label.Ptr<LabelType>(height - 1)[x];
                    if (label_val > 0)
                    {
                        label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                        if (src_c0[x] > 0)
                        {
                            dst_c0[x] = label_val;
                        }
                        else
                        {
                            dst_c0[x] = 0;
                        }
                    }
                    else
                    {
                        dst_c0[x] = 0;
                    }
                }
                else // handle only rest odd row
                {
                    const DT_U8 *const src_c0 = img.Ptr<DT_U8>(height - 1);
                    LabelType   *const dst_c0 = label.Ptr<LabelType>(height - 1);
                    for (DT_S32 x = 0; x < width; x += 2)
                    {
                        LabelType label_val = dst_c0[x];
                        if (label_val > 0)
                        {
                            label_val = this->m_solver->GetLabel(label_val);
                            if (src_c0[x] > 0)
                            {
                                dst_c0[x] = label_val;
                            }
                            else
                            {
                                dst_c0[x] = 0;
                            }
                            if (src_c0[x + 1] > 0)
                            {
                                dst_c0[x + 1] = label_val;
                            }
                            else
                            {
                                dst_c0[x + 1] = 0;
                            }
                        }
                        else
                        {
                            dst_c0[x] = 0;
                            dst_c0[x + 1] = 0;
                        }
                    }
                }
            }
            else
            {
                if (width & 1) // handle only rest odd cols
                {
                    const DT_S32 x = width - 1;
                    for (DT_S32 y = 0; y < height; y += 2)
                    {
                        const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                        const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
                        LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                        LabelType   *const dst_n0 = label.Ptr<LabelType>(y + 1);
                        LabelType label_val = dst_c0[x];
                        if (label_val > 0)
                        {
                            label_val = static_cast<LabelType>(this->m_solver->GetLabel(label_val));
                            if (src_c0[x] > 0)
                            {
                                dst_c0[x] = label_val;
                            }
                            else
                            {
                                dst_c0[x] = 0;
                            }
                            if (src_n0[x] > 0)
                            {
                                dst_n0[x] = label_val;
                            }
                            else
                            {
                                dst_n0[x] = 0;
                            }
                        }
                        else
                        {
                            dst_n0[x] = 0;
                        }
                    }
                }
            }
        }
        return Status::OK;
    }

};

template <typename LabelType>
class Spaghetti4CParallel : public ScanPlusUnionFindParallel<LabelType>
{
public:
    Spaghetti4CParallel(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length) : ScanPlusUnionFindParallel<LabelType>(ctx, solver_type, max_length)
    {}

    Status operator()(Context *ctx, const Mat &img, Mat &label)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        Status ret = Status::ERROR;

        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width  = img.GetSizes().m_width;

        DT_S32 step = height >> 5;
        step = Clamp<DT_S32>(step, 16, 256);
        step = ((step & 1) == 1) ? (step + 1) : step; // make sure even
        std::vector<DT_S32> chunks(ROUND_UP(height, step));

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&Spaghetti4CParallel::FirstScan4C,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4, std::placeholders::_5, std::placeholders::_6), std::cref(img),
                              std::ref(label), (DT_S32*)(chunks.data()), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FirstScan4C parallel_for fail");
            goto EXIT;
        }
        ret = this->MergeLabels4C(label, (DT_S32*)(chunks.data()));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MergeLabels4C parallel_for fail");
            goto EXIT;
        }

        for (DT_S32 i = 0; i < height; i = chunks[i])
        {
            this->m_solver->FlattenChunk(CHUNK_LABEL_INIT_4C(i, width), chunks[i + 1]);
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&Spaghetti4CParallel::SecondScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4),std::ref(label), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan parallel_for fail");
            goto EXIT;
        }

    EXIT:
        AURA_RETURN(ctx, ret);
    }

private:
    Status FirstScan4C(const Mat &img, Mat &label, DT_S32 *chunks, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk) override
    {
        const DT_S32 width     = img.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        chunks[start_row] = end_row;
        const LabelType label_init = CHUNK_LABEL_INIT_4C(start_row, width);
        LabelType label_last = label_init;

#define CONDITION_Q (src_p0[x] > 0)
#define CONDITION_S (src_c0[x - 1] > 0)
#define CONDITION_X (src_c0[x] > 0)

#define ACTION_1 dst_c0[x] = 0;
#define ACTION_2 dst_c0[x] = (LabelType)this->m_solver->NewChunkLabel(label_last++);
#define ACTION_3 dst_c0[x] = dst_p0[x];     // x <- q
#define ACTION_4 dst_c0[x] = dst_c0[x - 1]; // x <- s
#define ACTION_5 dst_c0[x] = this->m_solver->Merge(dst_p0[x], dst_c0[x - 1]); // x <- q + s

        // first row
        {
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(start_row);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(start_row);
            DT_S32 x = -1;

#include "connect_component_label/spaghetti_4c_firstline_graph.hpp"
        }

        // rest rows
        for (DT_S32 y = 1 + start_row; y < end_row; ++y)
        {
            const DT_U8 *src_c0     = img.Ptr<DT_U8>(y);
            const DT_U8 *src_p0     = img.Ptr<DT_U8>(y - 1);
            LabelType *const dst_c0 = label.Ptr<LabelType>(y);
            LabelType *const dst_p0 = label.Ptr<LabelType>(y - 1);
            DT_S32 x = -1;

#include "connect_component_label/spaghetti_4c_singleline_graph.hpp"
        }
        chunks[start_row + 1] = (DT_S32)label_last - (DT_S32)label_init;
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5

#undef CONDITION_Q
#undef CONDITION_S
#undef CONDITION_X
        return Status::OK;
    }

};

template <typename LabelType>
class Spaghetti8CParallel : public BlockBasedDecisionTreeParallel<LabelType>
{
public:
    Spaghetti8CParallel(Context *ctx, EquivalenceSolver solver_type, DT_U32 max_length) : BlockBasedDecisionTreeParallel<LabelType>(ctx, solver_type, max_length)
    {}

    Status operator()(Context *ctx, const Mat &img, Mat &label) override
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        Status ret = Status::ERROR;

        const DT_S32 height = img.GetSizes().m_height;
        const DT_S32 width = img.GetSizes().m_width;

        DT_S32 step = height >> 5;
        step = Clamp<DT_S32>(step, 16, 256);
        step = ((step & 1) == 1) ? (step + 1) : step; // make sure even
        std::vector<DT_S32> chunks(ROUND_UP(height, step));

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&Spaghetti8CParallel::FirstScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4, std::placeholders::_5, std::placeholders::_6), std::cref(img),
                              std::ref(label), (DT_S32*)(chunks.data()), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FirstScan parallel_for fail");
            goto EXIT;
        }

        ret = this->MergeLabels(img, label, (DT_S32*)(chunks.data()));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MergeLabels parallel_for fail");
            goto EXIT;
        }

        for (DT_S32 i = 0; i < height; i = chunks[i])
        {
            this->m_solver->FlattenChunk(CHUNK_LABEL_INIT_8C(i, width), chunks[i + 1]);
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), (height + step - 1) / step, std::bind(&Spaghetti8CParallel::SecondScan,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                              std::placeholders::_4,std::placeholders::_5), std::cref(img), std::ref(label), step);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "SecondScan parallel_for fail");
            goto EXIT;
        }

EXIT:
        AURA_RETURN(ctx, ret);
    }

private:
    Status FirstScan(const Mat &img, Mat &label, DT_S32 *chunks, DT_S32 step, DT_S32 start_blk, DT_S32 end_blk) override
    {
        const DT_S32 width     = img.GetSizes().m_width;
        const DT_S32 start_row = start_blk * step;
        const DT_S32 end_row   = Min(end_blk * step, img.GetSizes().m_height);

        const LabelType label_init = CHUNK_LABEL_INIT_8C(start_row, width);
        LabelType label_last       = label_init;
        DT_S32 step_true           = end_row - start_row;
        DT_S32 e_rows              = step_true & 0xfffffffe;
        DT_BOOL o_rows             = (step_true & 1) == 1;
        chunks[start_row]          = end_row;

#define CONDITION_B (src_p1[x - 1] > 0)
#define CONDITION_C (src_p1[x] > 0)
#define CONDITION_D (src_p1[x + 1] > 0)
#define CONDITION_E (src_p1[x + 2] > 0)
#define CONDITION_G (src_p0[x - 2] > 0)
#define CONDITION_H (src_p0[x - 1] > 0)
#define CONDITION_I (src_p0[x] > 0)
#define CONDITION_J (src_p0[x + 1] > 0)
#define CONDITION_K (src_p0[x + 2] > 0)
#define CONDITION_M (src_c0[x - 2] > 0)
#define CONDITION_N (src_c0[x - 1] > 0)
#define CONDITION_O (src_c0[x] > 0)
#define CONDITION_P (src_c0[x + 1] > 0)
#define CONDITION_R (src_n0[x - 1] > 0)
#define CONDITION_S (src_n0[x] > 0)
#define CONDITION_T (src_n0[x + 1] > 0)

#define ACTION_1  dst_c0[x] = 0;
#define ACTION_2  dst_c0[x] = (LabelType)this->m_solver->NewChunkLabel(label_last++);
#define ACTION_3  dst_c0[x] = dst_p1[x - 2];
#define ACTION_4  dst_c0[x] = dst_p1[x];
#define ACTION_5  dst_c0[x] = dst_p1[x + 2];
#define ACTION_6  dst_c0[x] = dst_c0[x - 2];
#define ACTION_7  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]);
#define ACTION_8  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]);
#define ACTION_9  dst_c0[x] = this->m_solver->Merge(dst_p1[x - 2], dst_c0[x - 2]);
#define ACTION_10 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]);
#define ACTION_11 dst_c0[x] = this->m_solver->Merge(dst_p1[x],     dst_c0[x - 2]);
#define ACTION_12 dst_c0[x] = this->m_solver->Merge(dst_p1[x + 2], dst_c0[x - 2]);
#define ACTION_13 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_p1[x + 2]);
#define ACTION_14 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x]),     dst_c0[x - 2]);
#define ACTION_15 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x - 2], dst_p1[x + 2]), dst_c0[x - 2]);
#define ACTION_16 dst_c0[x] = this->m_solver->Merge(this->m_solver->Merge(dst_p1[x],     dst_p1[x + 2]), dst_c0[x - 2]);
        if (1 == step_true)
        {
            // Single line
            const DT_U8 *const src_c0 = img.Ptr<DT_U8>(start_row);
            LabelType   *const dst_c0 = label.Ptr<LabelType>(start_row);
            DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_singleline_graph.hpp"
        }
        else
        {
            // First couple of lines
            {
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(start_row);
                const DT_U8 *const src_n0 = img.Ptr<DT_U8>(start_row + 1);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(start_row);
                DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_firstline_graph.hpp"
            }

            for (DT_S32 y = 2 + start_row; y < e_rows + start_row; y += 2)
            {
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
                const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
                const DT_U8 *const src_n0 = img.Ptr<DT_U8>(y + 1);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);

                DT_S32 x = -2;
                goto tree_0;

#include "connect_component_label/spaghetti_8c_graph.hpp"
            }

            // Last line (in case the rows are odd)
            if (o_rows)
            {
                DT_S32 y = end_row - 1;
                const DT_U8 *const src_c0 = img.Ptr<DT_U8>(y);
                const DT_U8 *const src_p0 = img.Ptr<DT_U8>(y - 1);
                const DT_U8 *const src_p1 = img.Ptr<DT_U8>(y - 2);
                LabelType   *const dst_c0 = label.Ptr<LabelType>(y);
                LabelType   *const dst_p1 = label.Ptr<LabelType>(y - 2);
                DT_S32 x = -2;
#include "connect_component_label/spaghetti_8c_lastline_graph.hpp"
            }
        }
        chunks[start_row + 1] = (DT_S32)label_last - (DT_S32)label_init;
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
        return Status::OK;
    }

};

template <typename LabelType>
static Status ScanPlusUnionFindImpl(Context *ctx, const Mat &img, Mat &label, EquivalenceSolver solver_type, ConnectivityType type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    const DT_S32 height = img.GetSizes().m_height;
    const DT_S32 width  = img.GetSizes().m_width;
    const DT_S32 upper_bound = ConnectivityType::CROSS == type ? UPPER_BOUND_4_CONNECTIVITY(height, width) : UPPER_BOUND_8_CONNECTIVITY(height, width);

    if (target.m_data.none.enable_mt)
    {
        ret = ScanPlusUnionFindParallel<LabelType>(ctx, solver_type, upper_bound)(ctx, img, label, type);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ScanPlusUnionFindParallel fail");
            return ret;
        }
    }
    else
    {
        ret = ScanPlusUnionFindNone<LabelType>(ctx, solver_type, upper_bound)(ctx, img, label, type);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ScanPlusUnionFindNone fail");
            return ret;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename LabelType>
static Status BlockBasedDecisionTreeImpl(Context *ctx, const Mat &img, Mat &label, EquivalenceSolver solver_type, ConnectivityType type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    const DT_S32 height = img.GetSizes().m_height;
    const DT_S32 width  = img.GetSizes().m_width;
    if (type != ConnectivityType::SQUARE)
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(ctx, "block based CCL algos, only support SQUARE ConnectivityType");
        return ret;
    }

    if (target.m_data.none.enable_mt)
    {
        ret = BlockBasedDecisionTreeParallel<LabelType>(ctx, solver_type, UPPER_BOUND_8_CONNECTIVITY(height, width))(ctx, img, label);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "BlockBasedDecisionTreeParallel fail");
            return ret;
        }
    }
    else
    {
        ret = BlockBasedDecisionTreeNone<LabelType>(ctx, solver_type, UPPER_BOUND_8_CONNECTIVITY(height, width))(ctx, img, label);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "BlockBasedDecisionTree fail");
            return ret;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename LabelType>
static Status SpaghettiImpl(Context *ctx, const Mat &img, Mat &label, EquivalenceSolver solver_type, ConnectivityType type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    const DT_S32 height = img.GetSizes().m_height;
    const DT_S32 width  = img.GetSizes().m_width;
    if (target.m_data.none.enable_mt)
    {
        switch (type)
        {
            case ConnectivityType::CROSS:
            {
                ret = Spaghetti4CParallel<LabelType>(ctx, solver_type, UPPER_BOUND_4_CONNECTIVITY(height, width))(ctx, img, label);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Spaghetti4CParallel fail");
                    return ret;
                }
                break;
            }
            case ConnectivityType::SQUARE:
            {
                ret = Spaghetti8CParallel<LabelType>(ctx, solver_type, UPPER_BOUND_8_CONNECTIVITY(height, width))(ctx, img, label);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Spaghetti8CParallel fail");
                    return ret;
                }
                break;
            }
            default:
            {
                ret = Status::ERROR;
                AURA_ADD_ERROR_STRING(ctx, "unsupport Connectivity Type for Spaghetti algo parallel");
                break;
            }
        }
    }
    else
    {
        switch (type)
        {
            case ConnectivityType::CROSS:
            {
                ret = Spaghetti4CNone<LabelType>(ctx, solver_type, UPPER_BOUND_4_CONNECTIVITY(height, width))(ctx, img, label);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Spaghetti4C fail");
                    return ret;
                }
                break;
            }
            case ConnectivityType::SQUARE:
            {
                ret = Spaghetti8CNone<LabelType>(ctx, solver_type, UPPER_BOUND_8_CONNECTIVITY(height, width))(ctx, img, label);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "Spaghetti8C fail");
                    return ret;
                }
                break;
            }
            default:
            {
                ret = Status::ERROR;
                AURA_ADD_ERROR_STRING(ctx, "unsupport Connectivity Type for Spaghetti algo");
                break;
            }
        }
    }

    AURA_RETURN(ctx, ret);
}

#undef UPPER_BOUND_4_CONNECTIVITY
#undef UPPER_BOUND_8_CONNECTIVITY

ConnectComponentLabelNone::ConnectComponentLabelNone(Context *ctx, const OpTarget &target) : ConnectComponentLabelImpl(ctx, target),
                                                                                             m_solver_type(EquivalenceSolver::UNION_FIND_PATH_COMPRESS)
{}

Status ConnectComponentLabelNone::SetArgs(const Array *src, Array *dst, CCLAlgo algo_type, ConnectivityType connectivity_type,
                                          EquivalenceSolver solver_type)
{
    if (ConnectComponentLabelImpl::SetArgs(src, dst, algo_type, connectivity_type, solver_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConnectComponentLabelImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (algo_type != CCLAlgo::SAUF && algo_type != CCLAlgo::BBDT && algo_type != CCLAlgo::SPAGHETTI)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "unsupported algorithms");
        return Status::ERROR;
    }

    if (solver_type != EquivalenceSolver::UNION_FIND   && solver_type != EquivalenceSolver::UNION_FIND_PATH_COMPRESS &&
        solver_type != EquivalenceSolver::REM_SPLICING && solver_type != EquivalenceSolver::THREE_TABLE_ARRAYS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "unsupported equivalence solver");
        return Status::ERROR;
    }

    m_solver_type = solver_type;

    return Status::OK;
}

Status ConnectComponentLabelNone::Run()
{
    const Mat *img = dynamic_cast<const Mat*>(m_src);
    Mat *label = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == img) || (DT_NULL == label))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (EquivalenceSolver::THREE_TABLE_ARRAYS == m_solver_type && m_target.m_data.none.enable_mt &&
        (label->GetElemType() == ElemType::U8 || label->GetElemType() == ElemType::U16))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "TTASolver not support U8 and U16 when multi-thread");
        return ret;
    }

    std::unordered_map<std::string, std::pair<std::string,
                      Status (*)(Context*, const Mat&, Mat&, EquivalenceSolver, ConnectivityType, const OpTarget&)>> ccl_algos
    {
        {"0_1", std::make_pair("SAUF_U8",       ScanPlusUnionFindImpl<DT_U8>)},
        {"0_3", std::make_pair("SAUF_U16",      ScanPlusUnionFindImpl<DT_U16>)},
        {"0_5", std::make_pair("SAUF_U32",      ScanPlusUnionFindImpl<DT_U32>)},
        {"0_6", std::make_pair("SAUF_S32",      ScanPlusUnionFindImpl<DT_S32>)},
        {"1_1", std::make_pair("BBDT_U8",       BlockBasedDecisionTreeImpl<DT_U8>)},
        {"1_3", std::make_pair("BBDT_U16",      BlockBasedDecisionTreeImpl<DT_U16>)},
        {"1_5", std::make_pair("BBDT_U32",      BlockBasedDecisionTreeImpl<DT_U32>)},
        {"1_6", std::make_pair("BBDT_S32",      BlockBasedDecisionTreeImpl<DT_S32>)},
        {"2_1", std::make_pair("SPAGHETTI_U8",  SpaghettiImpl<DT_U8>)},
        {"2_3", std::make_pair("SPAGHETTI_U16", SpaghettiImpl<DT_U16>)},
        {"2_5", std::make_pair("SPAGHETTI_U32", SpaghettiImpl<DT_U32>)},
        {"2_6", std::make_pair("SPAGHETTI_S32", SpaghettiImpl<DT_S32>)},
    };

    std::string ccl_func_nums = std::to_string((DT_S32)m_algo_type) + "_" + std::to_string((DT_S32)label->GetElemType());
    if (ccl_algos.count(ccl_func_nums) > 0)
    {
        ret = (ccl_algos[ccl_func_nums].second)(m_ctx, *img, *label, m_solver_type, m_connectivity_type, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, " ccl algos execute fail");
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, " current ccl algo is unsupported yet");
        ret = Status::ERROR;
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura