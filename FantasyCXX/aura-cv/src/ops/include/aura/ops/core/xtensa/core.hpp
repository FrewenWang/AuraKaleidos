#ifndef AURA_OPS_CORE_XTENSA_CORE_HPP__
#define AURA_OPS_CORE_XTENSA_CORE_HPP__

#include "aura/runtime/xtensa.h"

#include <new>

#define AURA_XTENSA_RPC_FUNC_REGISTER(op, func)

/**
 * @brief The external interface of the operator.
 */
#if !defined(AURA_VDSP_OP_HPP)
#define AURA_VDSP_OP_HPP()                                      \
    aura::xtensa::Status Initialize();                          \
                                                                \
    aura::xtensa::Status DeInitialize();                        \
                                                                \
    aura::xtensa::Status Run();
#endif

/**
 * @brief The implementation of the operator's external interface.
 */
#if !defined(AURA_VDSP_OP_CPP)
#define AURA_VDSP_OP_CPP(CLASSNAME)                                                             \
aura::xtensa::Status CLASSNAME##Vdsp::Initialize()                                              \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        if (aura::xtensa::ExecuteMode::TILE == m_mode)                                          \
        {                                                                                       \
            CLASSNAME##Tile *tile_impl = static_cast<CLASSNAME##Tile*>(m_impl);                 \
            if (DT_NULL == tile_impl)                                                           \
            {                                                                                   \
                AURA_XTENSA_LOG("tile_impl is null ptr");                                       \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            return tile_impl->Initialize();                                                     \
        }                                                                                       \
        else if (aura::xtensa::ExecuteMode::FRAME == m_mode)                                    \
        {                                                                                       \
            CLASSNAME##Frame *frame_impl = static_cast<CLASSNAME##Frame*>(m_impl);              \
            if (DT_NULL == frame_impl)                                                          \
            {                                                                                   \
                AURA_XTENSA_LOG("frame_impl is null ptr");                                      \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            return frame_impl->Initialize();                                                    \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            AURA_XTENSA_LOG("invalid mode");                                                    \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}                                                                                               \
                                                                                                \
aura::xtensa::Status CLASSNAME##Vdsp::DeInitialize()                                            \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        aura::xtensa::Status ret = aura::xtensa::Status::ERROR;                                 \
        if (aura::xtensa::ExecuteMode::TILE == m_mode)                                          \
        {                                                                                       \
            CLASSNAME##Tile *tile_impl = static_cast<CLASSNAME##Tile*>(m_impl);                 \
            if (DT_NULL == tile_impl)                                                           \
            {                                                                                   \
                AURA_XTENSA_LOG("tile_impl is null ptr");                                       \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            ret = tile_impl->DeInitialize();                                                    \
        }                                                                                       \
        else if (aura::xtensa::ExecuteMode::FRAME == m_mode)                                    \
        {                                                                                       \
            CLASSNAME##Frame *frame_impl = static_cast<CLASSNAME##Frame*>(m_impl);              \
            if (DT_NULL == frame_impl)                                                          \
            {                                                                                   \
                AURA_XTENSA_LOG("frame_impl is null ptr");                                      \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            ret = frame_impl->DeInitialize();                                                   \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            AURA_XTENSA_LOG("invalid mode");                                                    \
        }                                                                                       \
                                                                                                \
        if (aura::xtensa::VdspOp::DeInitialize() != aura::xtensa::Status::OK)                   \
        {                                                                                       \
            AURA_XTENSA_LOG("VdspOp::DeInitialize failed");                                     \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
                                                                                                \
        return ret;                                                                             \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}                                                                                               \
                                                                                                \
aura::xtensa::Status CLASSNAME##Vdsp::Run()                                                     \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        if (aura::xtensa::ExecuteMode::TILE == m_mode)                                          \
        {                                                                                       \
            CLASSNAME##Tile *tile_impl = static_cast<CLASSNAME##Tile*>(m_impl);                 \
            if (DT_NULL == tile_impl)                                                           \
            {                                                                                   \
                AURA_XTENSA_LOG("tile_impl is null ptr");                                       \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            return tile_impl->Run();                                                            \
        }                                                                                       \
        else if (aura::xtensa::ExecuteMode::FRAME == m_mode)                                    \
        {                                                                                       \
            CLASSNAME##Frame *frame_impl = static_cast<CLASSNAME##Frame*>(m_impl);              \
            if (DT_NULL == frame_impl)                                                          \
            {                                                                                   \
                AURA_XTENSA_LOG("frame_impl is null ptr");                                      \
                return aura::xtensa::Status::ERROR;                                             \
            }                                                                                   \
                                                                                                \
            return frame_impl->Run();                                                           \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            AURA_XTENSA_LOG("invalid mode");                                                    \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}
#endif

/**
 * @brief The implementation of the cascade operator's external interface.
 */
#if !defined(AURA_VDSP_CASCADE_CPP)
#define AURA_VDSP_CASCADE_CPP(CLASSNAME)                                                        \
Status CLASSNAME##Vdsp::Initialize()                                                            \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        CLASSNAME##Cascade *cascade_impl = static_cast<CLASSNAME##Cascade*>(m_impl);            \
        if (DT_NULL == cascade_impl)                                                            \
        {                                                                                       \
            AURA_XTENSA_LOG("cascade_impl is null ptr");                                        \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
                                                                                                \
        return cascade_impl->Initialize();                                                      \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}                                                                                               \
                                                                                                \
Status CLASSNAME##Vdsp::DeInitialize()                                                          \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        CLASSNAME##Cascade *cascade_impl = static_cast<CLASSNAME##Cascade*>(m_impl);            \
        if (DT_NULL == cascade_impl)                                                            \
        {                                                                                       \
            AURA_XTENSA_LOG("cascade_impl is null ptr");                                        \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
                                                                                                \
        Status ret = cascade_impl->DeInitialize();                                              \
                                                                                                \
        if (aura::xtensa::VdspOp::DeInitialize() != aura::xtensa::Status::OK)                   \
        {                                                                                       \
            AURA_XTENSA_LOG("VdspOp::DeInitialize failed");                                     \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
                                                                                                \
        return ret;                                                                             \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}                                                                                               \
                                                                                                \
Status CLASSNAME##Vdsp::Run()                                                                   \
{                                                                                               \
    if (m_impl)                                                                                 \
    {                                                                                           \
        CLASSNAME##Cascade *cascade_impl = static_cast<CLASSNAME##Cascade*>(m_impl);            \
        if (DT_NULL == cascade_impl)                                                            \
        {                                                                                       \
            AURA_XTENSA_LOG("cascade_impl is null ptr");                                        \
            return aura::xtensa::Status::ERROR;                                                 \
        }                                                                                       \
                                                                                                \
        return cascade_impl->Run();                                                             \
    }                                                                                           \
                                                                                                \
    return aura::xtensa::Status::ERROR;                                                         \
}
#endif

/**
 * @brief The implementation of the operator's run interface for different modes.
 */
#if !defined(AURA_VDSP_OP_MODE_CPP)
#define AURA_VDSP_OP_MODE_CPP(CLASSNAME, MODE)                                                                                                \
aura::xtensa::Status CLASSNAME##MODE::Run()                                                                                                   \
{                                                                                                                                             \
    if (0 == m_tile_num)                                                                                                                      \
    {                                                                                                                                         \
        AURA_XTENSA_LOG("m_tile_num is 0");                                                                                                   \
        return aura::xtensa::Status::ERROR;                                                                                                   \
    }                                                                                                                                         \
                                                                                                                                              \
    xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);                                                                                 \
    if (DT_NULL == xv_tm)                                                                                                                     \
    {                                                                                                                                         \
        AURA_XTENSA_LOG("xv_tm is null ptr");                                                                                                 \
        return aura::xtensa::Status::ERROR;                                                                                                   \
    }                                                                                                                                         \
                                                                                                                                              \
    if (sizeof(RefTile) != sizeof(aura::xtensa::RefTileWrapper))                                                                              \
    {                                                                                                                                         \
        AURA_XTENSA_LOG("sizeof(RefTile) != sizeof(RefTileWrapper)");                                                                         \
        return aura::xtensa::Status::ERROR;                                                                                                   \
    }                                                                                                                                         \
                                                                                                                                              \
    RefTile *xv_ref_tile = reinterpret_cast<RefTile*>(&m_ref_tile);                                                                           \
    if (DT_NULL == xv_ref_tile)                                                                                                               \
    {                                                                                                                                         \
        AURA_XTENSA_LOG("xv_ref_tile is null ptr");                                                                                           \
        return aura::xtensa::Status::ERROR;                                                                                                   \
    }                                                                                                                                         \
                                                                                                                                              \
    if (xvExecuteFullIauraKernel(xv_tm, xv_ref_tile, m_tile_num * sizeof(xvTile), this, Prepare, Execute, 0, 1, XV_AUTOSIZE_FLAG) != 0)       \
    {                                                                                                                                         \
        AURA_XTENSA_LOG("xvExecuteFullIauraKernel failed!");                                                                                  \
        return aura::xtensa::Status::ERROR;                                                                                                   \
    }                                                                                                                                         \
                                                                                                                                              \
    return aura::xtensa::Status::OK;                                                                                                          \
}
#endif

/**
 * @brief The implementation of the operator's run interface for frame.
 */
#if !defined(AURA_VDSP_OP_FRAME_CPP)
#  define AURA_VDSP_OP_FRAME_CPP(CLASSNAME) AURA_VDSP_OP_MODE_CPP(CLASSNAME, Frame)
#endif

/**
 * @brief The implementation of the operator's run interface for cascade.
 */
#if !defined(AURA_VDSP_OP_CASCADE_CPP)
#  define AURA_VDSP_OP_CASCADE_CPP(CLASSNAME) AURA_VDSP_OP_MODE_CPP(CLASSNAME, Cascade)
#endif

namespace aura
{
namespace xtensa
{

/**
 * @brief  Enumerated type for xtensa execution method.
 */
enum class ExecuteMode
{
    INVALID = 0,
    TILE,             /*!< The tile execution mode*/
    FRAME             /*!< The frame execution mode*/
};

/**
 * @brief Base tile implementation class.
 *
 * This class provides a base tile implementation for operators.
 */
class VdspOpTile
{
public:
    /**
     * @brief Constructor.
     *
     * @param tm The pointer to the TileManager object.
     */
    VdspOpTile(TileManager tm) : m_tm(tm), m_flag(DT_FALSE)
    {}

    /**
     * @brief Initialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize()
    {
        return Status::OK;
    }

    /**
     * @brief Run operator implementation, subclasses should override this function.
     *
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status Run()
    {
        return Status::ERROR;
    }

    /**
     * @brief DeInitialize operator implementation, subclasses should override this function.
     *
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status DeInitialize()
    {
        return Status::ERROR;
    }

protected:
    TileManager m_tm;         /*!< The pointer to xvTileManager object. */
    DT_BOOL     m_flag;       /*!< The flag to set args. */
};

/**
 * @brief Base frame implementation class.
 *
 * This class provides a base frame implementation for operators.
 */
class VdspOpFrame
{
public:
    /**
     * @brief Constructor.
     *
     * @param tm The pointer to the TileManager object.
     * @param tile_num The number of tile.
     */
    VdspOpFrame(TileManager tm, DT_S32 tile_num) : m_tm(tm), m_tile_num(tile_num)
    {}

    /**
     * @brief Initialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize()
    {
        return Status::OK;
    }

    /**
     * @brief Run operator implementation, subclasses should override this function.
     *
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status Run()
    {
        return Status::ERROR;
    }

    /**
     * @brief DeInitialize operator implementation, subclasses should override this function.
     *
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status DeInitialize()
    {
        for (DT_S32 i = 0; i < m_elem_types.size(); i++)
        {
            m_elem_types[i] = ElemType::INVALID;
        }

        for (DT_S32 i = 0; i < m_channels.size(); i++)
        {
            m_channels[i] = 0;
        }

        for (DT_S32 i = 0; i < m_frames.size(); i++)
        {
            Memset(&m_frames[i], 0, sizeof(FrameWrapper));
        }

        m_frames.clear();
        m_channels.clear();
        m_elem_types.clear();

        Memset(&m_ref_tile,  0, sizeof(RefTileWrapper));

        return Status::OK;
    }

protected:
    TileManager    m_tm;           /*!< The pointer to tileManager object. */
    DT_U32         m_tile_num;     /*!< The number of tile. */
    RefTileWrapper m_ref_tile;     /*!< The refTile object for full iaura exector. */

    vector<ElemType>     m_elem_types;
    vector<DT_S32>       m_channels;
    vector<FrameWrapper> m_frames;
};

class VdspOp;

/**
 * @brief A base class for cascade operations with pad support.
 * This class provides a common interface for creating and managing cascade operation with pad functionality.
 */
class VdspNode
{
public:
    /**
     * @brief Default constructor for creating a VdspNode object.
     */
    VdspNode() : m_op(DT_NULL), m_enable_pad(DT_FALSE), m_edge_size(aura::Sizes(0, 0))
    {}

    /**
     * @brief Constructor for creating a VdspNode object with pad enabled.
     */
    VdspNode(VdspOp *op) : m_op(op), m_edge_size(aura::Sizes(0, 0))
    {}

    /**
     * @brief Destructor for cleaning up resources associated with the VdspNode object.
     */
    ~VdspNode()
    {
        m_enable_pad   = DT_FALSE;
        m_border_type  = BorderType::CONSTANT;
        m_border_value = 0;
        m_extra_size   = aura::Sizes(0, 0);
        m_edge_size    = aura::Sizes(0, 0);

        m_src_tiles.clear();
        m_dst_tiles.clear();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(VdspNode);

    /**
     * @brief Configures edge parameters for padding.
     *
     * @param edge_size Sizes for edge padding.
     * @param border_type Type of border padding to apply.
     * @param border_value Value to use for border padding.
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status SetEdgeArgs(const aura::Sizes &edge_size = aura::Sizes(0, 0),
                       const BorderType border_type = BorderType::CONSTANT,
                       const Scalar &border_value = Scalar(0))
    {
        if (!IsValid())
        {
            AURA_XTENSA_LOG("Invalid node");
            return Status::ERROR;
        }

        if (m_edge_size != aura::Sizes(0, 0))
        {
            m_enable_pad = DT_TRUE;
        }

        m_edge_size    = edge_size;
        m_border_type  = border_type;
        m_border_value = border_value;

        return Status::OK;
    }

    /**
     * @brief Gets the edge size for the operation padding.
     * 
     * @return const aura::Sizes& The edge size.
     */
    aura::Sizes& GetEdgeSize()
    {
        return m_edge_size;
    }

    /**
     * @brief Configures extra size for padding.
     *
     * @param extra_size Sizes for extra padding.
     */
    DT_VOID SetExtraSize(const aura::Sizes &extra_size)
    {
        m_extra_size = extra_size;
    }

    /**
     * @brief Sets the input and output tiles for padding.
     *
     * @param srcs The input tiles for padding.
     * @param dsts The output tiles for padding.
     * 
     */
    DT_VOID SetExecuteTiles(vector<TileWrapper> &srcs, vector<TileWrapper> &dsts)
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            m_src_tiles.push_back(&srcs[i]);
        }

        for (int i = 0; i < dsts.size(); i++)
        {
            m_dst_tiles.push_back(&dsts[i]);
        }
    }

    /**
     * @brief Synchronously calls the operation with the given arguments.
     * The arguments are passed to the operation using the variadic template parameter pack.
     * @param args Variadic template parameter pack containing the arguments to pass to the operation.
     * @return Status code indicating success or failure.
     */
    template <typename Tp, typename ...ArgsType>
    Status Run(ArgsType &&...args)
    {
        if (!IsValid())
        {
            AURA_XTENSA_LOG("Invalid node");
            return Status::ERROR;
        }

        Tp *op = static_cast<Tp*>(m_op);
        if (DT_NULL == op)
        {
            AURA_XTENSA_LOG("null ptr");
            return Status::ERROR;
        }

        if (Extract() != Status::OK)
        {
            AURA_XTENSA_LOG("Extract failed");
            return Status::ERROR;
        }

        if (op->SetArgs(std::forward<ArgsType>(args)...) != Status::OK)
        {
            AURA_XTENSA_LOG("SetArgs failed");
            return Status::ERROR;
        }

        if (Pad() != Status::OK)
        {
            AURA_XTENSA_LOG("Pad failed");
            return Status::ERROR;
        }

        if (op->Initialize() != Status::OK)
        {
            AURA_XTENSA_LOG("Initialize failed");
            return Status::ERROR;
        }

        if (op->Run() != Status::OK)
        {
            AURA_XTENSA_LOG("Run failed");
            return Status::ERROR;
        }

        return Status::OK;
    }

    /**
     * @brief Restore the tile info for all the tiles of the operation object.
     *
     * @return Status code indicating success or failure.
     */
    Status Restore()
    {
        if (!IsValid())
        {
            AURA_XTENSA_LOG("Invalid node");
            return Status::ERROR;
        }

        for (auto& tile : m_src_tiles)
        {
            if (tile->Restore() != Status::OK)
            {
                AURA_XTENSA_LOG("Restore failed");
                return Status::ERROR;
            }
        }

        for (auto& tile : m_dst_tiles)
        {
            if (tile->Restore() != Status::OK)
            {
                AURA_XTENSA_LOG("Restore failed");
                return Status::ERROR;
            }
        }

        m_src_tiles.clear();
        m_dst_tiles.clear();

        return Status::OK;
    }

    /**
     * @brief Deinitializes the operation object.
     *
     * @return Status code indicating success or failure.
     */
    template <typename Tp>
    Status DeInitialize()
    {
        if (!IsValid())
        {
            AURA_XTENSA_LOG("Invalid node");
            return Status::ERROR;
        }

        Tp *op = static_cast<Tp*>(m_op);
        if (DT_NULL == op)
        {
            AURA_XTENSA_LOG("null ptr");
            return Status::ERROR;
        }

        if (op->DeInitialize() != Status::OK)
        {
            AURA_XTENSA_LOG("DeInitialize failed");
            return Status::ERROR;
        }

        return Status::OK;
    }

private:
    /**
     * @brief Extract the tile info for all the tiles of the operation object.
     *
     * @return Status code indicating success or failure.
     */
    Status Extract()
    {
        for (auto& tile : m_src_tiles)
        {
            if (tile->Extract() != Status::OK)
            {
                AURA_XTENSA_LOG("Extract failed");
                return Status::ERROR;
            }
        }

        for (auto& tile : m_dst_tiles)
        {
            if (tile->Extract() != Status::OK)
            {
                AURA_XTENSA_LOG("Extract failed");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    /**
     * @brief Applies pad to each tile.
     *
     * @return Status code indicating success or failure.
     */
    Status Pad()
    {
        for (const auto& tile : m_src_tiles)
        {
            if (tile->Reset(m_extra_size, m_edge_size) != Status::OK)
            {
                AURA_XTENSA_LOG("Reset failed");
                return Status::ERROR;
            }

            if (!m_enable_pad)
            {
                continue;
            }

            if (tile->Pad(m_border_type, m_border_value) != Status::OK)
            {
                AURA_XTENSA_LOG("Pad failed");
                return Status::ERROR;
            }
        }

        for (auto& tile : m_dst_tiles)
        {
            if (tile->Reset(m_extra_size, m_edge_size) != Status::OK)
            {
                AURA_XTENSA_LOG("Reset failed");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    /**
     * @brief Check if the operation object is valid.
     * @return True if valid, false otherwise.
     */
    DT_BOOL IsValid() const
    {
        return m_op != DT_NULL;
    }

protected:
    VdspOp *m_op;               /*!< Pointer to the Vdsp object. */
    DT_BOOL m_enable_pad;       /*!< Flag indicating if padding is enabled. */

    BorderType  m_border_type;  /*!< Type of border padding. */
    Scalar      m_border_value; /*!< Value for border padding. */
    aura::Sizes m_extra_size;   /*!< The boundary gap between the boundary of mem and the boundary of actual valid data.*/
    aura::Sizes m_edge_size;    /*!< Padding edge size. */

    vector<TileWrapper*> m_src_tiles;  /*!< Vector of in tile pointers. */
    vector<TileWrapper*> m_dst_tiles;  /*!< Vector of out tile pointers. */
};

/**
 * @brief A container class for managing VdspNode objects.
 * This class provides functionality to create and manage operation for various operations.
 */
class VdspNodeGroup
{
public:
    /**
     * @brief Default constructor for creating a VdspNodeGroup object.
     */
    VdspNodeGroup(TileManager tm) : m_tm(tm), m_is_valid(DT_TRUE)
    {}

    /**
     * @brief Destructor for cleaning up resources associated with the XtensaGraph object.
     */
    ~VdspNodeGroup()
    {
        m_tm       = DT_NULL;
        m_is_valid = DT_FALSE;

        m_nodes.clear();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(VdspNodeGroup);

    /**
     * @brief Creates an Node for a given tile manager and padding flag.
     *
     * @param name The name of the operation.
     * @param Tp The type of the operation.
     */
    template <typename Tp>
    DT_VOID MakeNode(const string &name)
    {
        DT_VOID *buffer = DT_NULL;
        VdspNode *node  = DT_NULL;
        Tp *op          = DT_NULL;

        if (m_nodes.find(name) != m_nodes.end())
        {
            AURA_XTENSA_LOG("invalid name %s", name.c_str());
            goto EXIT;
        }

        buffer = AllocateBuffer(m_tm, sizeof(Tp), 128);
        if (DT_NULL == buffer)
        {
            AURA_XTENSA_LOG("AllocateBuffer error");
            goto EXIT;
        }

        op = new(buffer) Tp(m_tm, ExecuteMode::TILE);
        if (DT_NULL == op)
        {
            AURA_XTENSA_LOG("op is null ptr");
            goto EXIT;
        }

        buffer = AllocateBuffer(m_tm, sizeof(VdspNode), 128);
        if (DT_NULL == buffer)
        {
            AURA_XTENSA_LOG("AllocateBuffer error");
            goto EXIT;
        }

        node = new(buffer) VdspNode(op);
        if (DT_NULL == node)
        {
            AURA_XTENSA_LOG("node is null ptr");
            goto EXIT;
        }

        m_nodes[name] = node;

        return;

    EXIT:
        m_is_valid = DT_FALSE;
    }

    /**
     * @brief Checks if the XtensaGraph object is valid.
     * This function returns true if the object is valid, false otherwise.
     * @return A boolean value indicating the validity of the XtensaGraph object.
     */
    DT_BOOL IsValid() const
    {
        return m_is_valid;
    }

    /**
     * @brief Accesses a VdspNode object by its name.
     * 
     * @param name The name of the VdspNode object to access.
     * @return A reference to the VdspNode object with the given name.
     */
    VdspNode& operator[](const string &name)
    {
        if (!m_is_valid)
        {
            AURA_XTENSA_LOG("invalid cascade");
            return m_dummy_node;
        }

        if (m_nodes.find(name) == m_nodes.end())
        {
            AURA_XTENSA_LOG("invalid node name");
            return m_dummy_node;
        }
        else
        {
            if (m_nodes[name] != DT_NULL)
            {
                return *(m_nodes[name]);
            }
            else
            {
                AURA_XTENSA_LOG("null ptr");
                return m_dummy_node;
            }
        }
    }

    /**
     * @brief Returns the extra size required for the operation.
     * This function returns a Sizes object containing the extra size required for the operation.
     * @return A Sizes object containing the extra size required for the operation.
     */
    aura::Sizes GetExtraSize()
    {
        if (!IsValid())
        {
            return aura::Sizes(0, 0);
        }

        aura::Sizes extra_size = aura::Sizes(0, 0);

        for (auto it = m_nodes.rbegin(); it != m_nodes.rend(); ++it)
        {
            VdspNode* node = it->second;

            node->SetExtraSize(extra_size);
            extra_size += node->GetEdgeSize();
        }

        m_extra_size += extra_size;

        return m_extra_size;
    }

    /**
     * @brief Restore the tile info for all the tiles of operation object.
     *
     * @return Status code indicating success or failure.
     */
    Status Restore()
    {
        if (!IsValid())
        {
            AURA_XTENSA_LOG("invalid graph");
            return Status::ERROR;
        }

        for (auto it = m_nodes.begin(); it != m_nodes.end(); it++)
        {
            VdspNode* node = it->second;
            if (node->Restore() != Status::OK)
            {
                AURA_XTENSA_LOG("Restore failed!");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

private:
    TileManager    m_tm;
    DT_BOOL        m_is_valid;
    aura::Sizes    m_extra_size;
    map<VdspNode*> m_nodes;
    VdspNode       m_dummy_node;
};

/**
 * @brief Base cascade implementation class.
 *
 * This class provides a base cascade implementation for cascade operators.
 */
class VdspOpCascade  : public VdspOpFrame
{
public:
    /**
     * @brief Constructor for creating a VdspOpCascade  object.
     */
    VdspOpCascade(TileManager tm, DT_S32 tile_num) : VdspOpFrame(tm, tile_num), m_nodes(tm)
    {}

    /**
     * @brief Initialize the operation object.
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status Initialize()
    {
        m_extra_size = m_nodes.GetExtraSize();

        return Status::OK;
    }

    /**
     * @brief DeInitialize the operation object.
     * @return Status::ERROR if successful; otherwise, an appropriate error status.
     */
    Status DeInitialize()
    {
        if (VdspOpFrame::DeInitialize() != Status::OK)
        {
            AURA_XTENSA_LOG("VdspOpFrame DeInitialize failed");
            return Status::ERROR;
        }

        m_extra_size = aura::Sizes(0, 0);

        return Status::OK;
    }

protected:
    VdspNodeGroup m_nodes;
    aura::Sizes   m_extra_size;
};

/**
 * @brief Base class for operator.
 *
 * This class provides a common interface and functionality for operator.
 */
class VdspOp
{
public:
    /**
     * @brief Constructor.
     *
     * @param xv_tm The pointer to the TileManager object.
     */
    VdspOp(TileManager tm, ExecuteMode mode) : m_tm(tm), m_mode(mode), m_idx(0), m_impl(DT_NULL)
    {
        m_idx = BufferCheckPointSave(tm);
        if (m_idx < 0)
        {
            AURA_XTENSA_LOG("BufferCheckPointSave failed!\n");
        }
    }

    /**
     * @brief DeInitialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status DeInitialize()
    {
        if (m_idx < 0)
        {
            AURA_XTENSA_LOG("idx failed!\n");
            return Status::ERROR;
        }

        if (BufferCheckPointRestore(m_tm, m_idx) != Status::OK)
        {
            AURA_XTENSA_LOG("BufferCheckPointRestore failed!\n");
            return Status::ERROR;
        }

        return Status::OK;
    }

protected:
    TileManager m_tm;          /*!< Pointer to TileManager object. */
    ExecuteMode m_mode;        /*!< Enumerated class of xtensa execution mode. */
    DT_S32      m_idx;         /*!< Pointer to start index of memory queues in local memory. */
    DT_VOID     *m_impl;       /*!< Pointer to implementation of operator. */
};

/**
 * @brief Execute an operator.
 *
 * This function simplifies the process of setting arguments, initializing, running, and deinitializing an operator.
 *
 * @param Tp The type of the operator.
 * @param ArgsType The types of the arguments.
 * @param op The operator to be executed.
 * @param args The arguments for the operator.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename ...ArgsType>
Status OpCall(Tp &op, ArgsType &&...args)
{
    Status ret = Status::ERROR;

    if ((ret = op.SetArgs(std::forward<ArgsType>(args)...)) != Status::OK)
    {
        AURA_XTENSA_LOG("SetArgs failed");
        goto EXIT;
    }

    if ((ret = op.Initialize()) != Status::OK)
    {
        AURA_XTENSA_LOG("Initialize failed");
        goto EXIT;
    }

    if ((ret = op.Run()) != Status::OK)
    {
        AURA_XTENSA_LOG("Run failed");
    }

EXIT:
    op.DeInitialize();

    return ret;
};

} // namespace xtensa
} // namespace aura

#endif //AURA_OPS_CORE_XTENSA_CORE_HPP__