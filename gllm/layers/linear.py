from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        