import torch
import torch.nn as nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(chunks=2, dim=-1)
        return F.silu(x) * y
        