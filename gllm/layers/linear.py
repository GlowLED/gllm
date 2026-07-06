from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        cast(Any, self.weight).weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            cast(Any, self.bias).weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor) -> None:
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
    

class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(input_size, output_size, bias)
    
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor) -> None:
        param.data.copy_(loaded_weights)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
    
class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)
        
        
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor) -> None:
        param_data = param.data
        full_data_output_size = loaded_weights.size(0)
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0)
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
        
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int) -> None:
        param_data = param.data
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        param_data = param_data.narrow(0, offset, shard_size)
        
        loaded_weights_start_idx = self.tp_rank * shard_size
        shared_weights = loaded_weights.narrow(0, loaded_weights_start_idx, shard_size)
        param_data.copy_(shared_weights)
        
        
class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias)
    
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str) -> None:
        param_data = param.data
        assert load_weight_id in ('q', 'k', 'v'), "load_weight_id must be one of 'q', 'k', 'v'"
        
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")
        
        param_data = param_data.narrow(0, offset, shard_size)
        loaded_weights_start_idx = self.tp_rank * shard_size
        shared_weights = loaded_weights.narrow(0, loaded_weights_start_idx, shard_size)
        
        param_data.copy_(shared_weights)
    
