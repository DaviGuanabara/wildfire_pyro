import os
from typing import Optional
from torch.utils.tensorboard import SummaryWriter



class TensorBoardLogger:
    """
    Lightweight abstraction over TensorBoard's SummaryWriter.
    Responsible for recording evaluation metrics during training.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.writer = SummaryWriter(
            log_dir=log_dir) if log_dir else NoOpWriter()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Safely log a scalar value to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
        return self

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()

    def is_active(self) -> bool:
        return self.writer is not None
    
    
class NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return self

    def flush(self):
        return self
