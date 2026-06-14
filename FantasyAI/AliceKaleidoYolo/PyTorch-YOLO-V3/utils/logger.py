import os
import logging
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        os.makedirs(log_dir, exist_ok=True)

        if TF_AVAILABLE:
            # 使用 TensorFlow 的 SummaryWriter（支持 TensorBoard）
            self.writer = tf.summary.create_file_writer(log_dir)
            self.use_tf = True
        else:
            # 回退到纯 Python 日志
            self.use_tf = False
            log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            # 同时输出到控制台
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

    def scalar_summary(self, tag, value, step):
        if self.use_tf:
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)
                self.writer.flush()
        else:
            self.logger.info(f"[{step}] {tag}: {value}")

    def list_of_scalars_summary(self, tag_value_pairs, step):
        if self.use_tf:
            with self.writer.as_default():
                for tag, value in tag_value_pairs:
                    tf.summary.scalar(tag, value, step=step)
                self.writer.flush()
        else:
            for tag, value in tag_value_pairs:
                self.logger.info(f"[{step}] {tag}: {value}")
