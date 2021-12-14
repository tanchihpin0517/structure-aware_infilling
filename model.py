import torch
from torch import nn

class Config:
    def __init__(
        self,
        num_pitch,
        d_embed=256
    ):
        self.num_pitch = num_pitch
        self.d_embed = d_embed
        self.padding_idx = padding_idx

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float().masked_fill(attn_mask[None, :, :, None], -65000).type_as(attn_score)
                    )
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = nn.functional.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None, output_attentions=False):

        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.d_embed= config.embed

        self.pitch_emb = nn.Embedding(config.num_pitch, config.d_embed)

    def forward(self):
        ...
