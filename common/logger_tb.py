"""TensorBoard logger wrapper for unified logging interface."""

import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TBLogger:
    """Wrapper around torch.utils.tensorboard.SummaryWriter with automatic directory creation."""
    
    def __init__(self, log_dir: str, name: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for logs (e.g., 'runs')
            name: Experiment name (subdirectory under log_dir)
        """
        self.log_path = os.path.join(log_dir, name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
    
    def scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Metric name (e.g., 'loss/policy')
            value: Scalar value to log
            step: Global step/iteration number
        """
        self.writer.add_scalar(tag, value, step)
    
    def image(self, tag: str, img: np.ndarray, step: int, dataformats: str = 'HWC'):
        """
        Log an image.
        
        Args:
            tag: Image name (e.g., 'samples/frame')
            img: Image array (numpy)
            step: Global step/iteration number
            dataformats: Format string ('HWC', 'CHW', etc.)
        """
        self.writer.add_image(tag, img, step, dataformats=dataformats)
    
    def text(self, tag: str, text: str, step: int):
        """
        Log text data.
        
        Args:
            tag: Text identifier (e.g., 'prompts/sample')
            text: Text string to log
            step: Global step/iteration number
        """
        self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close the TensorBoard writer and flush remaining data."""
        self.writer.close()
