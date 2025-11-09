#include "find_contours_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/ops/misc.h"
#include "aura/runtime/logger.h"

#include <queue>

#define AURA_CONTOURS_STORAGE_BLOCK_SIZE   ((1 << 16) - 128)
#define AURA_CONTOURS_STRUCT_ALIGN         ((DT_S32)sizeof(DT_F64))
#define AURA_CONTOURS_SEQ_CHAIN_CONTOUR    (41)
#define AURA_CONTOURS_SEQ_FLAG_HOLE        (2 << 14)
#define AURA_CONTOURS_SEQ_POLYGON          (16384)
#define AURA_CONTOURS_MAGIC_MASK           (0xFFFF0000)
#define AURA_CONTOURS_SEQ_MAGIC_VAL        (0x42990000)

namespace aura
{

constexpr DT_S32 FIND_CONTOURS_MAX_SIZE = 16;
static const Point2i code_deltas[8] = {{ 1, 0}, { 1, -1}, {0, -1}, {-1, -1},
                                       {-1, 0}, {-1,  1}, {0,  1}, { 1,  1}};

struct ContoursTreeNode
{
    DT_S32            color;
    DT_S32            flags;
    DT_S32            header_size;
    ContoursTreeNode *h_prev;
    ContoursTreeNode *h_next;
    ContoursTreeNode *v_prev;
    ContoursTreeNode *v_next;
};

struct MemBlock
{
    MemBlock *prev;
    MemBlock *next;
};

struct ContoursMemStorage
{
    MemBlock           *bottom;
    DT_S32              block_size;
    MemBlock           *top;
    DT_S32              free_space;
    ContoursMemStorage *parent;
};

struct SeqBlock
{
    SeqBlock *prev;
    SeqBlock *next;
    DT_S32    start_index;
    DT_S32    count;
    DT_S8    *data;
};

struct ContoursSeq : public ContoursTreeNode
{
    DT_S32              total;
    DT_S32              elem_size;
    DT_S8              *block_max;
    DT_S8              *ptr;
    DT_S32              delta_elems;
    ContoursMemStorage *storage;
    SeqBlock           *free_blocks;
    SeqBlock           *first;
};

struct ContourInfo
{
    DT_S32       flags;
    ContourInfo *next;
    ContourInfo *parent;
    ContoursSeq *contour;
    Point2i      origin;
};

struct ContourScanner
{
    ContoursMemStorage *storage1;
    ContoursMemStorage *storage2;
    DT_S8              *img0;
    DT_S8              *img;
    DT_S32              img_step;
    Sizes               img_size;
    Point2i             offset;
    Point2i             pt;
    Point2i             lnbd;
    DT_S32              nbd;
    ContourInfo         l_cinfo;
    ContourInfo         frame_info;
    ContoursSeq         frame;
    ContoursMethod      approx_method;
    ContoursMode        mode;
    DT_S32              seq_type1;
    DT_S32              header_size1;
    DT_S32              elem_size1;
};

struct SeqOperator
{
    ContoursSeq *seq;
    SeqBlock    *block;
    DT_S8       *ptr;
    DT_S8       *block_max;
};

using ContourScannerPointer = ContourScanner*;
using MemStorage = ContoursMemStorage*;

constexpr DT_S32 ALIGNED_SEQ_BLOCK_SIZE = (DT_S32)AURA_ALIGN(sizeof(SeqBlock), AURA_CONTOURS_STRUCT_ALIGN);

static Status FlushSeqWriter(Context *ctx, SeqOperator *writer)
{
    if (DT_NULL == writer)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr..");
        return Status::ERROR;
    }

    ContoursSeq *seq = writer->seq;
    seq->ptr = writer->ptr;

    if (writer->block)
    {
        DT_S32 total          = 0;
        SeqBlock *first_block = writer->seq->first;
        SeqBlock *block       = first_block;

        writer->block->count = (DT_S32)((writer->ptr - writer->block->data) / seq->elem_size);

        if (writer->block->count <= 0)
        {
            AURA_ADD_ERROR_STRING(ctx, "writer count should > 0 ...");
            return Status::ERROR;
        }

        do
        {
            total += block->count;
            block  = block->next;
        } while (block != first_block);

        writer->seq->total = total;
    }
    return Status::OK;
}

static Status SetSeqBlockSize(Context *ctx, ContoursSeq *seq, DT_S32 delta_elements)
{
    if ((DT_NULL == seq) || (DT_NULL == seq->storage) || (delta_elements < 0))
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr or delta_elements out of range ...");
        return Status::ERROR;
    }

    DT_S32 block_size_aligned = (seq->storage->block_size - sizeof(MemBlock) - sizeof(SeqBlock)) & (-AURA_CONTOURS_STRUCT_ALIGN);

    if (0 == delta_elements)
    {
        delta_elements = Max((1 << 10) / seq->elem_size, (DT_S32)1);
    }
    if (delta_elements * seq->elem_size > block_size_aligned)
    {
        delta_elements = block_size_aligned / seq->elem_size;
        if (0 == delta_elements)
        {
            AURA_ADD_ERROR_STRING(ctx, "Storage block size is too small to fit the sequence elements..");
            return Status::ERROR;
        }
    }

    seq->delta_elems = delta_elements;
    return Status::OK;
}

static Status GoNextMemBlock(Context *ctx, ContoursMemStorage *storage)
{
    if (!storage)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        return Status::ERROR;
    }
    Status ret = Status::ERROR;

    if (!storage->top || !storage->top->next)
    {
        MemBlock *block;

        if (!(storage->parent))
        {
            block = (MemBlock*)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, storage->block_size, 32);
            if (DT_NULL == block)
            {
                AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed...");
                return Status::ERROR;
            }
        }
        else
        {
            ContoursMemStorage *parent = storage->parent;
            ret = GoNextMemBlock(ctx, parent);
            if (ret != Status::OK)
            {
                 AURA_ADD_ERROR_STRING(ctx, "GoNextMemBlock failed..");
                 return Status::ERROR;
            }

            if (DT_NULL == parent->top)
            {
                parent->top = parent->bottom;
                parent->free_space = parent->top ? parent->block_size - sizeof(MemBlock) : 0;
            }

            block = parent->top;
            if (block == parent->top)
            {
                if (parent->bottom != block)
                {
                    AURA_ADD_ERROR_STRING(ctx, "parent bottom shouble be same as block..");
                    return Status::ERROR;
                }
                parent->top = parent->bottom = 0;
                parent->free_space = 0;
            }
            else
            {
                parent->top->next = block->next;
                if (block->next)
                {
                    block->next->prev = parent->top;
                }
            }
        }

        block->next = 0;
        block->prev = storage->top;

        if (storage->top)
        {
            storage->top->next = block;
        }
        else
        {
            storage->top = storage->bottom = block;
        }
    }

    if (storage->top->next)
    {
        storage->top = storage->top->next;
    }
    storage->free_space = storage->block_size - sizeof(MemBlock);
    if (storage->free_space % AURA_CONTOURS_STRUCT_ALIGN != 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "free_space should be aligned..");
        return Status::ERROR;
    }

    return Status::OK;
}

static DT_VOID* MemStorageAlloc(Context *ctx, ContoursMemStorage *storage, size_t size, Status &ret)
{
    if (DT_NULL == storage)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        ret = Status::ERROR;
        return DT_NULL;
    }

    if (size > INT_MAX)
    {
        AURA_ADD_ERROR_STRING(ctx, "Too large memory block is requested..");
        ret = Status::ERROR;
        return DT_NULL;
    }

    if (0 != (storage->free_space % AURA_CONTOURS_STRUCT_ALIGN))
    {
        AURA_ADD_ERROR_STRING(ctx, "free space not aligned...");
        ret = Status::ERROR;
        return DT_NULL;
    }

    if ((size_t)storage->free_space < size)
    {
        size_t max_free_space = (storage->block_size - sizeof(MemBlock)) & (-AURA_CONTOURS_STRUCT_ALIGN);
        if (max_free_space < size)
        {
            AURA_ADD_ERROR_STRING(ctx, "requested size is negative or too big..");
            ret = Status::ERROR;
            return DT_NULL;
        }

        ret = GoNextMemBlock(ctx, storage);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "GoNextMemBlock failed..");
            ret = Status::ERROR;
            return DT_NULL;
        }
    }

    DT_S8 *ptr = (DT_S8*)storage->top + storage->block_size - storage->free_space;
    if (0 != (size_t)ptr % AURA_CONTOURS_STRUCT_ALIGN)
    {
        AURA_ADD_ERROR_STRING(ctx, "ptr not aligned..");
        ret = Status::ERROR;
        return DT_NULL;
    }

    storage->free_space = (storage->free_space - (DT_S32)size) & (-AURA_CONTOURS_STRUCT_ALIGN);

    ret = Status::OK;
    return ptr;
}

static Status GrowSeq(Context *ctx, ContoursSeq *seq, DT_S32 in_front_of)
{
    if (DT_NULL == seq)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    SeqBlock *block = seq->free_blocks;

    if (!block)
    {
        DT_S32 elem_size = seq->elem_size;
        DT_S32 delta_elems = seq->delta_elems;
        ContoursMemStorage *storage = seq->storage;

        if (seq->total >= delta_elems * 4)
        {
            Status ret = SetSeqBlockSize(ctx, seq, delta_elems * 2);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "The sequence has DT_NULL storage pointer..");
                return Status::ERROR;
            }
        }

        if (DT_NULL == storage)
        {
            AURA_ADD_ERROR_STRING(ctx, "The sequence has DT_NULL storage pointer..");
            return Status::ERROR;
        }

        if ((size_t)(((DT_S8*)storage->top + storage->block_size - storage->free_space) - seq->block_max) < AURA_CONTOURS_STRUCT_ALIGN &&
            storage->free_space >= seq->elem_size && !in_front_of)
        {
            DT_S32 delta = storage->free_space / elem_size;

            delta = Min(delta, delta_elems) * elem_size;
            seq->block_max += delta;
            storage->free_space = ((DT_S32)(((DT_S8*)storage->top + storage->block_size) -
                                              seq->block_max)) & (-AURA_CONTOURS_STRUCT_ALIGN);
            return Status::OK;
        }
        else
        {
            DT_S32 delta = elem_size * delta_elems + ALIGNED_SEQ_BLOCK_SIZE;
            if (storage->free_space < delta)
            {
                DT_S32 small_block_size = Max((DT_S32)1, delta_elems / 3)*elem_size + ALIGNED_SEQ_BLOCK_SIZE;

                if (storage->free_space >= small_block_size + AURA_CONTOURS_STRUCT_ALIGN)
                {
                    delta = (storage->free_space - ALIGNED_SEQ_BLOCK_SIZE) / seq->elem_size;
                    delta = delta * seq->elem_size + ALIGNED_SEQ_BLOCK_SIZE;
                }
                else
                {
                    ret = GoNextMemBlock(ctx, storage);
                    if (ret != Status::OK || storage->free_space < delta)
                    {
                        AURA_ADD_ERROR_STRING(ctx, "free space < delta");
                        return Status::ERROR;
                    }
                }
            }

            block = (SeqBlock*)MemStorageAlloc(ctx, storage, delta, ret);

            if ((DT_NULL == block) || (ret != Status::OK))
            {
                AURA_ADD_ERROR_STRING(ctx, "MemStorageAlloc failed..");
                return Status::ERROR;
            }

            block->data = (DT_S8*)AURA_ALIGN((size_t)(block + 1), AURA_CONTOURS_STRUCT_ALIGN);
            block->count = delta - ALIGNED_SEQ_BLOCK_SIZE;
            block->prev = block->next = 0;
        }
    }
    else
    {
        seq->free_blocks = block->next;
    }

    if (!(seq->first))
    {
        seq->first = block;
        block->prev = block->next = block;
    }
    else
    {
        block->prev = seq->first->prev;
        block->next = seq->first;
        block->prev->next = block->next->prev = block;
    }

    if (block->count % seq->elem_size != 0 || block->count <= 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "block count and size wrong..");
        return Status::ERROR;
    }

    if (!in_front_of)
    {
        seq->ptr = block->data;
        seq->block_max = block->data + block->count;
        block->start_index = block == block->prev ? 0 : (block->prev->start_index + block->prev->count);
    }
    else
    {
        DT_S32 delta = block->count / seq->elem_size;
        block->data += block->count;

        if (block != block->prev)
        {
            if (0 != seq->first->start_index)
            {
                AURA_ADD_ERROR_STRING(ctx, "start_index != 0...");
                return Status::ERROR;
            }
            seq->first = block;
        }
        else
        {
            seq->block_max = seq->ptr = block->data;
        }

        block->start_index = 0;

        for (;;)
        {
            block->start_index += delta;
            block = block->next;
            if (block == seq->first)
            {
                break;
            }
        }
    }

    block->count = 0;
    return Status::OK;
}

template<typename Tp> AURA_INLINE Status WriteSeqElem(Context *ctx, Tp &elem, SeqOperator &writer)
{
    if (writer.seq->elem_size != sizeof(elem))
    {
        AURA_ADD_ERROR_STRING(ctx, "ele size invalid..");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    if (writer.ptr >= writer.block_max)
    {
        ContoursSeq *seq = writer.seq;
        ret = FlushSeqWriter(ctx, &writer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FlushSeqWriter failed..");
            AURA_RETURN(ctx, ret);
        }

        ret = GrowSeq(ctx, seq, 0);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "GrowSeq failed..");
            AURA_RETURN(ctx, ret);
        }

        writer.block     = seq->first->prev;
        writer.ptr       = seq->ptr;
        writer.block_max = seq->block_max;
    }

    if (writer.ptr > writer.block_max - sizeof(elem))
    {
        return Status::ERROR;
    }
    memcpy(writer.ptr, &elem, sizeof(elem));
    writer.ptr += sizeof(elem);

    return Status::OK;
}

static Status CreateSeq(Context *ctx, DT_S32 seq_flags, size_t header_size, size_t elem_size,
                        ContoursMemStorage *storage, ContoursSeq **seq)
{
    if ((DT_NULL == ctx) || (DT_NULL == storage))
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if ((elem_size <= 0))
    {
        AURA_ADD_ERROR_STRING(ctx, "header_size bad size..");
        return Status::ERROR;
    }

    *seq = (ContoursSeq*)MemStorageAlloc(ctx, storage, header_size, ret);
    if ((DT_NULL == seq) || (ret != Status::OK))
    {
        AURA_ADD_ERROR_STRING(ctx, "MemStorageAlloc failed..");
        return Status::ERROR;
    }

    memset(*seq, 0, header_size);
    (*seq)->header_size = (DT_S32)header_size;
    (*seq)->flags       = (seq_flags & ~AURA_CONTOURS_MAGIC_MASK) | AURA_CONTOURS_SEQ_MAGIC_VAL;
    (*seq)->elem_size   = (DT_S32)elem_size;
    (*seq)->storage     = storage;
    ret = SetSeqBlockSize(ctx, *seq, (DT_S32)((1 << 10) / elem_size));
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "The sequence has DT_NULL storage pointer..");
        AURA_RETURN(ctx, ret);
    }

    AURA_RETURN(ctx, ret);
}

static DT_VOID StartReadSeq(const ContoursSeq *seq, SeqOperator *reader)
{
    reader->seq = 0;
    reader->block = 0;
    reader->ptr = reader->block_max = 0;
    reader->seq = (ContoursSeq*)seq;

    if (seq->first)
    {
        reader->ptr       = seq->first->data;
        reader->block     = seq->first;
        reader->block_max = reader->block->data + reader->block->count * seq->elem_size;
    }
}

static Status SetSeqReaderPos(Context *ctx, SeqOperator *reader)
{
    DT_S32 total = reader->seq->total;
    DT_S32 elem_size = reader->seq->elem_size;

    if (total < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid total size");
        return Status::ERROR;
    }
    DT_S32 index = total, count = 0;

    SeqBlock *block = reader->seq->first;
    if (index >= (count = block->count))
    {
        if (index + index <= total)
        {
            do
            {
                block = block->next;
                index -= count;
            }
            while (index >= (count = block->count));
        }
        else
        {
            do
            {
                block = block->prev;
                total -= block->count;
            } while (index < total);

            index -= total;
        }
    }

    reader->ptr = block->data + index * elem_size;

    if (reader->block != block)
    {
        reader->block = block;
        reader->block_max = block->data + block->count * elem_size;
    }

    return Status::OK;
}

static DT_VOID DestroyMemStorage(Context *ctx, ContoursMemStorage *storage)
{
    if (DT_NULL == storage)
    {
        return;
    }
    MemBlock *dst_top = (storage->parent) ? storage->parent->top : 0;

    for (MemBlock *block = storage->bottom; block != 0;)
    {
        MemBlock *temp = block;

        block = block->next;
        if (storage->parent)
        {
            if (dst_top)
            {
                temp->prev = dst_top;
                temp->next = dst_top->next;
                temp->next->prev = (temp->next) ? temp : temp->next->prev;
                dst_top = dst_top->next = temp;
            }
            else
            {
                dst_top = storage->parent->bottom = storage->parent->top = temp;
                temp->prev = temp->next = 0;
                storage->free_space = storage->block_size - sizeof(*temp);
            }
        }
        else
        {
            AURA_FREE(ctx, temp);
            temp = DT_NULL;
        }
    }

    storage->top = 0;
    storage->bottom = 0;
    storage->free_space = 0;
}

static Status StartFindContoursNone(Context *ctx, const Mat &src, ContoursMemStorage *storage, ContoursMode mode,
                                    ContoursMethod method, Point2i offset, ContourScannerPointer &scanner, Mat &thres_dst)
{
    if ((DT_NULL == ctx) || (DT_NULL == storage))
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr..");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    scanner = (ContourScannerPointer)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(*scanner), 32);
    if (DT_NULL == scanner)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed...");
        return Status::ERROR;
    }
    memset((DT_VOID*)scanner, 0, sizeof(*scanner));

    DT_S32 step = src.GetRowStep();

    if ((step < 0) || (src.GetSizes().m_height < 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src step or size wrong...");
        AURA_FREE(ctx, scanner);
        scanner = NULL;
        return Status::ERROR;
    }

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img_step = step;
    scanner->img_size.m_width  = src.GetSizes().m_width - 1;
    scanner->img_size.m_height = src.GetSizes().m_height - 1;
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt.m_x = scanner->pt.m_y = 1;
    scanner->lnbd.m_x = 0;
    scanner->lnbd.m_y = 1;
    scanner->nbd = 2;
    scanner->frame_info.contour  = &(scanner->frame);
    scanner->frame_info.next     = 0;
    scanner->frame_info.parent   = 0;
    scanner->frame.flags = AURA_CONTOURS_SEQ_FLAG_HOLE;
    scanner->approx_method = method;
    scanner->header_size1  = sizeof(ContoursSeq);
    scanner->elem_size1    = sizeof(DT_S32) * 2;
    scanner->seq_type1     = AURA_CONTOURS_SEQ_POLYGON;

    ret = IThreshold(ctx, src, thres_dst, 0.f, 1.f, AURA_THRESH_BINARY, OpTarget::None());
    if (ret != Status::OK)
    {
        AURA_FREE(ctx, scanner);
        scanner = NULL;
        AURA_ADD_ERROR_STRING(ctx, "Threshold run failed");
        return Status::ERROR;
    }

    DT_U8 *p_dst  = (DT_U8*)thres_dst.GetData();
    scanner->img0 = (DT_S8*)p_dst;
    scanner->img  = (DT_S8*)(p_dst + step);

    AURA_RETURN(ctx, ret);
}

static Status FetchContour(Context *ctx, DT_S8 *ptr, DT_S32 step, Point2i pt, ContoursSeq *contour, ContoursMethod method_in)
{
    if ((DT_NULL == ptr) || (DT_NULL == contour))
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    const DT_S8 nbd = 2;
    SeqOperator writer;
    DT_S8  *i0 = ptr;
    DT_S8  *i1 = DT_NULL;
    DT_S8  *i3 = DT_NULL;
    DT_S8  *i4 = DT_NULL;
    DT_S32 prev_s = -1;
    DT_S32 s      = 0;
    DT_S32 s_end  = 0;
    DT_S32 method = static_cast<DT_S32>(method_in);

    DT_S32 deltas[FIND_CONTOURS_MAX_SIZE] =
    {
        1, -step + 1, -step, -step - 1, -1, step - 1, step, step + 1,
        1, -step + 1, -step, -step - 1, -1, step - 1, step, step + 1,
    };

    writer.seq       = contour;
    writer.block     = contour->first ? contour->first->prev : 0;
    writer.ptr       = contour->ptr;
    writer.block_max = contour->block_max;

    if (method < 0)
    {
        *((Point2i*)contour) = pt;
    }

    s_end = s = (((contour)->flags & AURA_CONTOURS_SEQ_FLAG_HOLE) != 0) ? 0 : 4;

    do
    {
        s  = (s - 1) & 7;
        i1 = i0 + deltas[s];
    } while (*i1 == 0 && s != s_end);

    if (s == s_end)
    {
        *i0 = (DT_S8)(nbd | -128);
        if (method >= 0)
        {
            ret = WriteSeqElem<Point2i>(ctx, pt, writer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "WriteSeqElem failed");
                return Status::ERROR;
            }
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        for (;;)
        {
            if (DT_NULL == i3)
            {
                AURA_ADD_ERROR_STRING(ctx, "i3 null...");
                return Status::ERROR;
            }

            s_end = s;
            s = std::min(s, FIND_CONTOURS_MAX_SIZE - 1);

            while (s < (FIND_CONTOURS_MAX_SIZE - 1))
            {
                i4 = i3 + deltas[++s];
                if (DT_NULL == i4)
                {
                    AURA_ADD_ERROR_STRING(ctx, "i4 null...");
                    return Status::ERROR;
                }

                if (*i4 != 0)
                {
                    break;
                }
            }

            s &= 7;

            if ((DT_U32)(s - 1) < (DT_U32)s_end)
            {
                *i3 = (DT_S8)(nbd | -128);
            }
            else if (*i3 == 1)
            {
                *i3 = nbd;
            }

            if (method < 0)
            {
                DT_U8 _s = (DT_U8)s;
                ret = WriteSeqElem<DT_U8>(ctx, _s, writer);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "WriteSeqElem failed");
                    return Status::ERROR;
                }
            }
            else
            {
                if ((s != prev_s) || (0 == method))
                {
                    ret = WriteSeqElem<Point2i>(ctx, pt, writer);
                    if (ret != Status::OK)
                    {
                        AURA_ADD_ERROR_STRING(ctx, "WriteSeqElem failed");
                        return Status::ERROR;
                    }
                    prev_s = s;
                }
                pt = Point2i(pt.m_x + code_deltas[s].m_x, pt.m_y + code_deltas[s].m_y);
            }

            if ((i4 == i0) && (i3 == i1))
            {
                break;
            }

            i3 = i4;
            s = (s + 4) & 7;
        }
    }

    ret = FlushSeqWriter(ctx, &writer);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "FlushSeqWriter fail...");
        return Status::ERROR;
    }

    ContoursSeq* seq = writer.seq;

    if (writer.block && writer.seq->storage)
    {
        ContoursMemStorage *storage = seq->storage;
        DT_S8 *storage_block_max = (DT_S8*) storage->top + storage->block_size;

        if ((DT_U32)((storage_block_max - storage->free_space) - seq->block_max) < AURA_CONTOURS_STRUCT_ALIGN)
        {
            storage->free_space = ((DT_S32)(storage_block_max - seq->ptr)) & (-AURA_CONTOURS_STRUCT_ALIGN);
            seq->block_max = seq->ptr;
        }
    }

    writer.ptr = 0;

    if (!((writer.seq->total == 0 && writer.seq->first == 0) ||
           writer.seq->total > writer.seq->first->count      ||
          (writer.seq->first->prev == writer.seq->first && writer.seq->first->next == writer.seq->first)))
    {
        AURA_ADD_ERROR_STRING(ctx, "writer seq wrong!");
        return Status::ERROR;
    }
    return Status::OK;
}

static Status EndProcessContour(Context *ctx, ContourScannerPointer scanner)
{
    if (scanner->l_cinfo.contour)
    {
        ContoursTreeNode *node   = (ContoursTreeNode*)scanner->l_cinfo.contour;
        ContoursTreeNode *parent = (ContoursTreeNode*)scanner->l_cinfo.parent->contour;
        if (parent->v_next == node)
        {
            AURA_ADD_ERROR_STRING(ctx, "parent->v_next should not be same as node..");
            return Status::ERROR;
        }

        node->v_prev = scanner->l_cinfo.parent->contour != &(scanner->frame) ? parent : 0;
        node->h_next = parent->v_next;
        parent->v_next = node;
    }
    memset((DT_VOID*)&(scanner->l_cinfo), 0, sizeof(scanner->l_cinfo));

    return Status::OK;
}

// main impl
static Status FindNextContourNone(Context *ctx, ContourScannerPointer scanner, ContoursSeq **contour)
{
    if (scanner->img_step < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "img step must >= 0");
        AURA_RETURN(ctx, Status::ERROR);
    }
    Status ret = Status::ERROR;

    ret = EndProcessContour(ctx, scanner);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "EndProcessContour fail..");
        return Status::ERROR;
    }

    DT_S8 *img0       = scanner->img0;
    DT_S8 *img        = scanner->img;
    DT_S32 step       = scanner->img_step;
    DT_S32 x          = scanner->pt.m_x;
    DT_S32 y          = scanner->pt.m_y;
    DT_S32 width      = scanner->img_size.m_width;
    DT_S32 height     = scanner->img_size.m_height;
    ContoursMode mode = scanner->mode;
    Point2i lnbd      = scanner->lnbd;
    DT_S32 prev       = img[x - 1];
    const DT_S32 new_mask = -2;

    for (; y < height; y++, img += step)
    {
        DT_S32 *img_i  = DT_NULL;
        DT_S32 p       = 0;

        for (; x < width; x++)
        {
            {
                for (; (x < width) && ((p = img[x]) == prev); x++)
                {
                    ;
                }
            }
            if (x >= width)
            {
                break;
            }

            {
                ContoursSeq *seq = 0;
                DT_S32 is_hole = 0;
                Point2i origin = {0, 0};

                if ((!img_i && !(prev == 0 && p == 1)) ||
                    (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)))
                {
                    if ((!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                    {
                        goto RESUMESCAN;
                    }

                    lnbd.m_x = (prev & new_mask) ? (x - 1) : lnbd.m_x;
                    is_hole = 1;
                }

                if ((ContoursMode::RETR_EXTERNAL == mode) && (is_hole || img0[lnbd.m_y * static_cast<size_t>(step) + lnbd.m_x] > 0))
                {
                    goto RESUMESCAN;
                }

                origin = Point2i(x - is_hole, y);
                lnbd.m_x = x - is_hole;

                ret = CreateSeq(ctx, scanner->seq_type1, scanner->header_size1,
                                scanner->elem_size1, scanner->storage1, &seq);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "CreateSeq fail..");
                    return Status::ERROR;
                }
                seq->flags |= is_hole ? AURA_CONTOURS_SEQ_FLAG_HOLE : 0;

                ContourInfo *l_cinfo = &(scanner->l_cinfo);
                Point2i temp = {origin.m_x + scanner->offset.m_x, origin.m_y + scanner->offset.m_y};

                ret = FetchContour(ctx, img + x - is_hole, step, temp, seq, scanner->approx_method);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "FetchContour fail..");
                    return Status::ERROR;
                }

                l_cinfo->contour = seq;
                l_cinfo->origin  = origin;
                l_cinfo->parent  = &(scanner->frame_info);
                l_cinfo->contour->v_prev = l_cinfo->parent->contour;

                if (0 == scanner->frame_info.contour)
                {
                    l_cinfo->contour = 0;
                    if (scanner->storage1 == scanner->storage2)
                    {
                        if (DT_NULL == scanner->storage1->top)
                        {
                            scanner->storage1->top = scanner->storage1->bottom;
                            scanner->storage1->free_space = scanner->storage1->top ? scanner->storage1->block_size - sizeof(MemBlock) : 0;
                        }
                    }
                    else
                    {
                        if (scanner->storage1->parent)
                        {
                            DestroyMemStorage(ctx, scanner->storage1);
                        }
                        else
                        {
                            scanner->storage1->top = scanner->storage1->bottom;
                            scanner->storage1->free_space = scanner->storage1->bottom ? scanner->storage1->block_size - sizeof(MemBlock) : 0;
                        }
                    }
                    p = img[x];
                    goto RESUMESCAN;
                }

                scanner->l_cinfo = *l_cinfo;
                scanner->pt.m_x  = !img_i ? x + 1 : x + 1 - is_hole;
                scanner->pt.m_y  = y;
                scanner->lnbd    = lnbd;
                scanner->img     = (DT_S8*)img;
                *contour         = l_cinfo->contour;
                AURA_RETURN(ctx, ret);
            }

RESUMESCAN:
            {
                prev = p;
                lnbd.m_x = (prev & -2) ? x : lnbd.m_x;
            }
        }

        lnbd.m_x = 0;
        lnbd.m_y = y + 1;
        x = 1;
        prev = 0;
    }

    *contour = DT_NULL;
    AURA_RETURN(ctx, ret);
}

static Status EndFindContoursNone(Context *ctx, ContourScannerPointer *p_scanner, ContoursSeq **first_contour)
{
    Status ret = Status::ERROR;
    ContourScannerPointer scanner = *p_scanner;

    ret = EndProcessContour(ctx, scanner);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "EndProcessContour fail..");
        return Status::ERROR;
    }

    if (scanner->storage1 != scanner->storage2)
    {
        DestroyMemStorage(ctx, scanner->storage1);
        AURA_FREE(ctx, scanner->storage1);
        scanner->storage1 = DT_NULL;
    }

    *first_contour = (ContoursSeq*)(scanner->frame.v_next);

    AURA_FREE(ctx, *p_scanner);
    *p_scanner = DT_NULL;
    return Status::OK;
}

static Status TreeToNodeSeq(Context *ctx, const DT_VOID *first, DT_S32 header_size,
                            ContoursMemStorage *storage, ContoursSeq **allseq)
{
    if (!storage)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        return Status::ERROR;
    }

    Status ret = CreateSeq(ctx, 0, header_size, sizeof(first), storage, allseq);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "CreateSeq fail..");
        return Status::ERROR;
    }

    if (first)
    {
        const DT_VOID *node_it = (DT_VOID*)first;
        DT_S32 level = 0, max_level = INT_MAX;

        for (;;)
        {
            ContoursTreeNode* prev_node = DT_NULL;
            ContoursTreeNode *node = prev_node = (ContoursTreeNode*)node_it;

            if (node)
            {
                if (node->v_next && level + 1 < max_level)
                {
                    node = node->v_next;
                    level++;
                }
                else
                {
                    while (0 == node->h_next)
                    {
                        node = node->v_prev;
                        if (--level < 0)
                        {
                            node = 0;
                            break;
                        }
                    }
                    node = node && max_level != 0 ? node->h_next : 0;
                }
            }

            node_it  = node;

            if (!prev_node)
            {
                break;
            }
            if (DT_NULL == *allseq)
            {
                continue;
            }
            DT_S8 *ptr = (*allseq)->ptr;

            if (ptr >= (*allseq)->block_max)
            {
                GrowSeq(ctx, *allseq, 0);
                ptr = (*allseq)->ptr;
                if ((ptr + (*allseq)->elem_size) > (*allseq)->block_max)
                {
                    continue;
                }
            }

            memcpy(ptr, &prev_node, (*allseq)->elem_size);
            (*allseq)->first->prev->count++;
            (*allseq)->total++;
            (*allseq)->ptr = ptr + (*allseq)->elem_size;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status CvtSeqToArray(Context *ctx, const ContoursSeq *seq, std::vector<std::vector<Point2i>> &array, DT_S32 index)
{
    if (DT_NULL == seq || 0 == seq->total)
    {
        AURA_ADD_ERROR_STRING(ctx, "nullptr...");
        return Status::ERROR;
    }

    SeqOperator reader;
    DT_CHAR *dst = (DT_CHAR*)(array[index].data());
    DT_S32 elem_size = seq->elem_size;

    DT_S32 total = seq->total * elem_size;

    StartReadSeq(seq, &reader);
    if (SetSeqReaderPos(ctx, &reader) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SetSeqReaderPos fail...");
        return Status::ERROR;
    }

    do
    {
        DT_S32 count = Min(total, (DT_S32)(reader.block_max - reader.ptr));

        memcpy(dst, reader.ptr, count);

        dst += count;
        reader.block     = reader.block->next;
        reader.ptr       = reader.block->data;
        reader.block_max = reader.ptr + reader.block->count * elem_size;
        total -= count;
    } while (total > 0);

    return Status::OK;
}

static Status FindContoursU8C1None(Context *ctx, const Mat &src, std::vector<std::vector<Point2i>> &contours,
                                   std::vector<Scalari> &hierarchy, ContoursMode mode, ContoursMethod method, const Point2i &offset)
{
    if (DT_NULL == ctx)
    {
        AURA_ADD_ERROR_STRING(ctx, "ctx nullptr..");
        return Status::ERROR;
    }

    AURA_UNUSED(hierarchy);

    if (src.GetElemType() != ElemType::U8 || src.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src must be U8C1 type");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    // 1.make border
    Sizes3 border_sizes = src.GetSizes() + Sizes3(2, 2, 0);
    Point2i offset_0    = {-1, -1};
    Point2i offset_in   = {offset_0.m_x + offset.m_x, offset_0.m_y + offset.m_y};

    Mat src_border = Mat(ctx, ElemType::U8, border_sizes, AURA_MEM_DEFAULT);
    ret = IMakeBorder(ctx, src, src_border, 1, 1, 1, 1, BorderType::CONSTANT, Scalar(0), OpTarget::None());
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "make border fail..");
        AURA_RETURN(ctx, ret);
    }

    // 2.allocate contours mem
    ContoursMemStorage *mem_storage = (ContoursMemStorage*)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(ContoursMemStorage), 32);
    if (DT_NULL == mem_storage)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed...");
        return Status::ERROR;
    }
    memset(mem_storage, 0, sizeof(*mem_storage));
    mem_storage->block_size = AURA_ALIGN(AURA_CONTOURS_STORAGE_BLOCK_SIZE, AURA_CONTOURS_STRUCT_ALIGN);
    MemStorage storage(mem_storage);
    ContoursSeq *p_contours = DT_NULL, *p_start_node = DT_NULL;

    // 3.main algo body
    ContourScannerPointer scanner = DT_NULL;
    ContoursSeq *contour = DT_NULL;
    hierarchy.clear();

    Mat dst_thresd = src_border.Clone();
    if (!dst_thresd.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "dst_thresd mat is invalid.");
        return Status::ERROR;
    }

    ret = StartFindContoursNone(ctx, src_border, storage, mode, method, offset_in, scanner, dst_thresd);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "StartFindContoursNone fail..");
        goto EXIT;
    }

    do
    {
        ret = FindNextContourNone(ctx, scanner, &contour);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "FineNextContourNone fail..");
            goto EXIT;
        }
    } while (DT_NULL != contour);

    ret = EndFindContoursNone(ctx, &scanner, &p_contours);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "EndFindContoursNone fail..");
        goto EXIT;
    }

    // 4.convert seqs to vector
    ret = TreeToNodeSeq(ctx, p_contours, sizeof(ContoursSeq), storage, &p_start_node);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "TreeToNodeSeq fail..");
        goto EXIT;
    }

    if (p_start_node)
    {
        contours.resize(p_start_node->total);
        hierarchy.reserve(p_start_node->total);

        SeqOperator seq_operator;
        StartReadSeq(p_start_node, &seq_operator);

        for (DT_S32 i = 0; i < p_start_node->total; i++)
        {
            ContoursSeq *c = *(ContoursSeq **)(seq_operator.ptr);
            c->color = i;
            contours[i].resize(c->total);

            ret = CvtSeqToArray(ctx, c, contours, i);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CvtSeqToArray fail..");
                goto EXIT;
            }

            if ((seq_operator.ptr += sizeof(ContoursSeq*)) >= seq_operator.block_max)
            {
                seq_operator.block = seq_operator.block->next;
                seq_operator.ptr = seq_operator.block->data;
            }

            DT_S32 h_next = c->h_next ? ((c->h_next)->color) : -1;
            DT_S32 h_prev = c->h_prev ? ((c->h_prev)->color) : -1;
            DT_S32 v_next = c->v_next ? ((c->v_next)->color) : -1;
            DT_S32 v_prev = c->v_prev ? ((c->v_prev)->color) : -1;

            hierarchy.emplace_back(h_next, h_prev, v_next, v_prev);
        }
    }

EXIT:
    DestroyMemStorage(ctx, storage);
    AURA_FREE(ctx, storage);
    storage = DT_NULL;
    AURA_RETURN(ctx, ret);
}

FindContoursNone::FindContoursNone(Context *ctx, const OpTarget &target) : FindContoursImpl(ctx, target)
{}

Status FindContoursNone::SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy,
                                 ContoursMode mode, ContoursMethod method, Point2i offset)
{
    if (FindContoursImpl::SetArgs(src, contours, hierarchy, mode, method, offset) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FindContoursImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (m_src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    if (m_src->GetElemType() != ElemType::U8 || m_src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be U8C1 type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status FindContoursNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    Status ret = FindContoursU8C1None(m_ctx, *src, *m_contours, *m_hierarchy, m_mode, m_method, m_offset);

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura