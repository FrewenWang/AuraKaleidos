from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """创建 TensorBoard 日志写入器"""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()
