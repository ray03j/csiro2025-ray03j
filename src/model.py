import torch
import torch.nn as nn
import torch.nn.functional as F
from .timmEncoder import TimmEncoder

def get_model_from_cfg(cfg):
    if cfg.model.arch == "timm_encoder":
        model = TimmEncoder(cfg)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model.arch}")
    return model