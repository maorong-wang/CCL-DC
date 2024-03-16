"""Tensorboard utils functions
"""
from datetime import datetime
import torch
import os


def get_writer(writer_name, base_path="runs/"):
    """Initialize tensorboard summary writer
    Args:
        writer_name (str): Name of the experiment to writer in tensorboard
    """
    if len(writer_name) < 1:
            now = datetime.now()
            exp_name = f"{base_path}noname/{now.year}_{now.month}_{now.day}_{now.hour}"
            writer = torch.utils.SummaryWriter(exp_name)
    else:
        writer = torch.utils.SummaryWriter(os.path.join(base_path, writer_name))
    
    return writer