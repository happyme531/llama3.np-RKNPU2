from typing import Optional
from dataclasses import dataclass


# @dataclass
# class ModelArgs:
#     # @formatter:off
#     # Model params for ./stories15M.model.npz
#     dim: int                    = 288       # D
#     n_layers: int               = 6
#     n_heads: int                = 6         # QHN, HN, HD = 48
#     n_kv_heads: Optional[int]   = None      # KVHN = 6
#     vocab_size: int             = 32000     # VS
#     max_seq_len: int            = 256       # M
#     max_new_tokens: int         = 50
#     norm_eps: float             = 1e-6
#     max_batch_size: int         = 1
#     # @formatter:on


# @dataclass
class ModelArgs:
    # @formatter:off
    # Model params for llama3.2-1b
    dim: int                    = 2048       # D
    n_layers: int               = 16
    n_heads: int                = 32         # QHN, HN, HD = ?
    n_kv_heads: Optional[int]   = 8      # KVHN = 8
    vocab_size: int             = 128256     # VS
    max_seq_len: int            = 8192       # M
    rope_theta: float           = 500000.0
    max_new_tokens: int         = 100
    norm_eps: float             = 1e-5
    max_batch_size: int         = 1
    # @formatter:on
