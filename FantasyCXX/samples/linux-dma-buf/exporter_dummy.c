#include <linux/dma-buf.h>
#include <linux/module.h>

static struct sg_table* exporter_map_dma_buf(
    struct dma_buf_attachment* attachment, enum dma_data_direction direction) {
  return NULL;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment* attachment,
                                   struct sg_table* table,
                                   enum dma_data_direction direction) {}

static void exporter_release(struct dma_buf* dma_buffer) {}

static void* exporter_kmap_atomic(struct dma_buf* dma_buffer,
                                  unsigned long page_number) {
  return NULL;
}

static void* exporter_kmap(struct dma_buf* dma_buffer,
                           unsigned long page_number) {
  return NULL;
}

static int exporter_mmap(struct dma_buf* dma_buffer,
                         struct vm_area_struct* area) {
  return -ENODEV;
}

// 第一步：创建DMA buffer 的ops
static const struct dma_buf_ops exp_dmabuf_ops = {
    .map_dma_buf = exporter_map_dma_buf,
    .unmap_dma_buf = exporter_unmap_dma_buf,
    .release = exporter_release,
    .map_atomic = exporter_kmap_atomic,
    .map = exporter_kmap,
    .mmap = exporter_mmap,
};

static int __init exporter_init(void) {
  DEFINE_DMA_BUF_EXPORT_INFO(export_info);
  struct dma_buf* dma_buffer;

  export_info.ops = &exp_dmabuf_ops;
  export_info.size = PAGE_SIZE;
  export_info.flags = O_CLOEXEC;
  export_info.priv = "null";

  dma_buffer = dma_buf_export(&export_info);

  return IS_ERR(dma_buffer) ? PTR_ERR(dma_buffer) : 0;
}

module_init(exporter_init);
