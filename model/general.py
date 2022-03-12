from dataclasses import dataclass, field
from .tokenizer import Tokenizer
import torch
from typing import List, Tuple
from torch import nn

@dataclass
class Config:
    vocab_size: int
    d_model: int
    d_inner: int
    n_head: int
    n_layer: int
    mem_len: int
    clamp_len: int = 4096
    ignore_idx: int = -100
    dropout: float = 0.1
    use_cp: bool = True
    d_subembed: int = 256
    class_ranges: List[tuple] = None

@dataclass
class Checkpoint:
    epoch: int
    config: Config
    model_state_dict: dict
    optim_state_dict: dict
    loss: float
    tokenizer: Tokenizer

@dataclass
class Output:
    losses: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    pred_scores: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Tuple[torch.FloatTensor] = None
    attentions: Tuple[torch.FloatTensor] = None

class Embedding(nn.Module):
    def decompose(self, embed) -> List[torch.FloatTensor]:
        raise NotImplementedError()

class REMIEmbedding(Embedding):
    def __init__(self, vocab_size, d_embed):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.trans_backward = nn.Linear(d_embed, vocab_size)

    def forward(self, input_ids):
        return self.embed(input_ids)

    def decompose(self, embed) -> List[torch.FloatTensor]:
        return [self.trans_backward(embed)]


class CPEmbedding(Embedding):
    def __init__(self, vocab_size, d_embed, d_subembed, class_ranges, dropout=0.1):
        super().__init__()
        #assert d_subembed * len(class_ranges) <= d_embed
        self.d_embed = d_embed
        self.d_subembed = d_subembed
        self.class_ranges = class_ranges
        self.class_num = len(class_ranges)

        self.subembed = nn.Embedding(vocab_size, d_subembed)
        self.decoders = nn.ModuleList()
        for rng in class_ranges:
            start, end = rng
            self.decoders.append(nn.Linear(d_subembed, end-start))
        self.trans_forward = nn.Linear(self.class_num*d_subembed, d_embed)
        self.trans_backward = nn.Linear(d_embed, self.class_num*d_subembed)

        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        input_ids: (B, L, C)
            B: batch size
            L: sequence length
            C: number of classes
        """
        B, L, C = input_ids.shape
        out = self.subembed(input_ids)
        out = out.view(B, L, C*self.d_subembed)
        out = self.trans_forward(self.drop(out))
        return out

    def decompose(self, embed: torch.FloatTensor) -> List[torch.FloatTensor]:
        """
        embed: (B, L, D)
            B: batch size
            L: sequence length
            D: dimention
        """
        B, L, D = embed.shape
        subembed = self.drop(self.trans_backward(embed))
        subembed = subembed.view(B, L, self.class_num, self.d_subembed)
        out = []
        for i, decoder in enumerate(self.decoders):
            out.append(decoder(subembed[:,:,i,:]))
        return out

