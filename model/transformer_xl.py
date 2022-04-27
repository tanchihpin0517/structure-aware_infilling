import torch
from torch import nn
from .general import Config, Output, REMIEmbedding, CPEmbedding
from dataclasses import dataclass
import math
from utils import log as ulog
import numpy as np
import utils

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


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
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)

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

    def _rel_shift(self, x, cut_len=None):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)
        if cut_len is not None:
            x = x[:, 0:cut_len, :, :]

        return x

    def forward(self, query, content, r, attn_mask=None, mems=None, head_mask=None, output_attentions=False, add_and_norm=True):
        qlen, rlen, bsz = query.size(0), r.size(0), query.size(1)

        if mems is not None:
            cat = torch.cat([mems, content], 0)

            if add_and_norm and self.pre_lnorm:
                query = self.layer_norm(query)
                cat = self.layer_norm(cat)

            w_head_q = self.q_net(query)
            w_head_k = self.k_net(cat)
            w_head_v = self.v_net(cat)
            r_head_k = self.r_net(r)
        else:
            if add_and_norm and self.pre_lnorm:
                query = self.layer_norm(query)
                content = self.layer_norm(content)
            w_head_q = self.q_net(query)
            w_head_k = self.k_net(content)
            w_head_v = self.v_net(content)
            r_head_k = self.r_net(r)

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
        BD = self._rel_shift(BD, cut_len=klen)

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

        if add_and_norm:
            if self.pre_lnorm:
                # residual connection
                outputs = [query + attn_out]
            else:
                # residual connection + layer normalization
                outputs = [self.layer_norm(query + attn_out)]
        else:
            outputs = [attn_out]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.cross_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, dec_inp, dec_r, dec_attn_mask=None, enc_inp=None, enc_r=None, enc_attn_mask=None, enc_out_sel=None, enc_out_mask=None, mems=None, head_mask=None, output_attentions=False):

        # self attention
        attn_outputs = self.dec_attn(
            dec_inp,
            dec_inp,
            dec_r,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        if enc_inp is not None:
            assert enc_r is not None
            assert enc_out_sel is not None

            enc_attn_outs = []
            for i in range(enc_inp.shape[0]):
                # cross attention
                outs = self.cross_attn(
                    attn_outputs[0],
                    enc_inp[i],
                    enc_r,
                    attn_mask=enc_attn_mask[i] if enc_attn_mask is not None else None,
                    mems=None,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    add_and_norm=False,
                )

                enc_attn_outs.append(outs[0])

            enc_attn_outs = torch.stack(enc_attn_outs)
            #enc_out_sel = enc_out_sel.clone().detach()
            #if enc_out_mask is not None:
            #    enc_out_sel[enc_out_mask == 1] = 0 # each sel with -1 shound be masked
            enc_out_sel = enc_out_sel[None, :, :, None].expand(enc_attn_outs.shape[0], -1, -1, enc_attn_outs.shape[-1])
            enc_attn_outs = torch.gather(enc_attn_outs, 0, enc_out_sel)[0]

            if enc_out_mask is not None:
                enc_out_mask = enc_out_mask[:, :, None]
                enc_attn_outs = enc_attn_outs.masked_fill(enc_out_mask > 0, 0)

            attn_outputs[0] = self.layer_norm(attn_outputs[0] + enc_attn_outs)

        # feed forward
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs

class RelPartialLearnableEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.enc_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, enc_inp, r, enc_attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        enc_attn_outs = []
        for i in range(enc_inp.shape[0]):
            attn_outputs = self.enc_attn(
                enc_inp[i],
                enc_inp[i],
                r,
                attn_mask=enc_attn_mask[i] if enc_attn_mask is not None else None,
            )
            ff_output = self.pos_ff(attn_outputs[0])
            enc_attn_outs.append(ff_output)

        enc_attn_outs = torch.stack(enc_attn_outs)

        return [enc_attn_outs]

@dataclass
class TransformerXLConfig(Config):
    d_head: int = 0
    xavier: float = False
    init_std: float = 0.02
    token_type_num: int = 3

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        self.d_head = self.d_model // self.n_head

@dataclass
class TransformerXLOutput(Output):
    pass

class TransformerXL(nn.Module):
    def __init__(self, config):
        super(TransformerXL, self).__init__()

        self.config = config
        self.n_token = config.vocab_size
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.clamp_len = config.clamp_len
        self.use_cp = config.use_cp
        self.infilling = config.infilling
        self.token_type_num = config.token_type_num

        if config.use_cp:
            self.word_emb = CPEmbedding(config.vocab_size, config.d_model, config.d_subembed, config.class_ranges)
        else:
            self.word_emb = REMIEmbedding(config.vocab_size, config.d_model)

        self.seg_embed = nn.Embedding(config.token_type_num, config.d_model)
        self.seg_id_A = 0
        self.seg_id_B = 1
        self.seg_id_C = 2

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        self.enc_layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    config.n_head,
                    config.d_model,
                    config.d_head,
                    config.d_inner,
                    config.dropout,
                    dropatt=0.0,
                    pre_lnorm=False,
                    r_w_bias=self.r_w_bias,
                    r_r_bias=self.r_r_bias,
                    layer_norm_epsilon=1e-5,
                )
            )
            self.enc_layers.append(
                RelPartialLearnableEncoderLayer(
                    config.n_head,
                    config.d_model,
                    config.d_head,
                    config.d_inner,
                    config.dropout,
                    dropatt=0.0,
                    pre_lnorm=False,
                    r_w_bias=self.r_w_bias,
                    r_r_bias=self.r_r_bias,
                    layer_norm_epsilon=1e-5,
                )
            )
        #self.trans_layer = nn.Linear(config.d_model, config.vocab_size)

        self.drop = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim = -1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_idx, reduction="sum")
        self.ignore_idx = config.ignore_idx

        self.apply(self._init_weights)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids,
        struct_ids,
        struct_masks,
        struct_seqs,
        struct_seq_masks,
        mems=None,
        labels=None,
        head_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        input_ids = input_ids.transpose(0, 1).contiguous()
        struct_ids = struct_ids.transpose(0, 1).contiguous()
        struct_masks = struct_masks.transpose(0, 1).contiguous() if struct_masks is not None else None
        if self.use_cp:
            struct_seqs = struct_seqs.permute(1, 2, 0, 3).contiguous()
        else:
            struct_seqs = struct_seqs.permute(1, 2, 0).contiguous()
        struct_seq_masks = struct_seq_masks.permute(1, 2, 0).contiguous()

        qlen, bsz = input_ids.shape[:2]
        slen = struct_seqs.shape[1]

        #if mems is None:
        #    mems = self._init_mems(bsz)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        word_emb = self.word_emb(input_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.transpose(0, 1).contiguous()
            seg_emb = self.seg_embed(token_type_ids)
        else:
            seg_emb = None

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
        mask_len = klen - self.mem_len
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len - 1))[:, :, None]  # -1

        hids = []
        attentions = [] if output_attentions else None
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        # struct
        struct_emb = self.word_emb(struct_seqs)

        enc_struct_seq_masks = struct_seq_masks[:, None, :, :].expand(-1, slen, -1, -1)
        dec_struct_seq_masks = struct_seq_masks[:, None, :, :].expand(-1, qlen, -1, -1)

        struct_pos_seq = torch.arange(slen-1, -slen, -1.0, device=struct_emb.device, dtype=struct_emb.dtype)
        struct_pos_emb = self.pos_emb(struct_pos_seq)

        if struct_masks is not None:
            struct_ids[struct_masks == 1] = 0 # each sel with -1 shound be masked

        core_out = self.drop(word_emb + seg_emb) if seg_emb is not None else self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        struct_out = self.drop(struct_emb)
        struct_pos_emb = self.drop(struct_pos_emb)

        for i, layer in enumerate(self.enc_layers):
            layer_outputs = layer(
                enc_inp=struct_emb,
                r=struct_pos_emb,
                enc_attn_mask=enc_struct_seq_masks,
                output_attentions=output_attentions,
            )
            struct_out = layer_outputs[0]

        for i, layer in enumerate(self.layers):
            hids.append(core_out)
            mems_i = None if mems is None else mems[i]
            layer_outputs = layer(
                core_out,
                pos_emb,
                dec_attn_mask=dec_attn_mask,
                enc_inp=struct_out,
                enc_r=struct_pos_emb,
                enc_attn_mask=dec_struct_seq_masks,
                enc_out_sel=struct_ids,
                enc_out_mask=struct_masks,
                mems=mems_i,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            core_out = layer_outputs[0]
            if output_attentions:
                attentions.append(layer_outputs[1])

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        if output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(core_out)
            hids = tuple(t.transpose(0, 1).contiguous() for t in hids)
        else:
            hids = None
        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = core_out.transpose(0, 1).contiguous()
        scores = self.word_emb.decompose(core_out) # (C, B, L, D)
        pred_scores = [self.softmax(score) for score in scores]

        # torch does softmax in CrossEntropyLoss
        if labels is not None:
            assert len(labels) == len(scores)
            """
            we set reduction to "sum" to avoid nan while all elements of labels are ignore_index
            here we transform the "sum" losses to "mean" losses
            """
            losses = []
            for i in range(len(labels)):
                tgt_num = labels[i][labels[i] != self.ignore_idx].numel()
                d = tgt_num if tgt_num != 0 else 1
                losses.append(self.criterion(scores[i][:, :labels[i].size(1), :].transpose(1,2), labels[i]) / d)
        else:
            losses = None

        return TransformerXLOutput(
            losses = losses,
            last_hidden_states = core_out,
            pred_scores = pred_scores,
            mems = new_mems,
            hidden_states = hids,
            attentions = attentions,
        )

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)

    def _init_weight(self, weight):
        if self.config.xavier:
            nn.init.xavier_normal(weight)
        else:
            nn.init.normal_(weight, 0.0, std=self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            #return None
            return [h.detach() for h in hids]

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def prepare_training_data(
        self,
        song_ids: list,
        struct_ids: list,
        struct_indices: list,
        struct_masks: list,
        bar_ids: list,
        tokenizer,
        bar_range_num = 6,
        max_seq_len = None,
        only_middle=True,
    ):
        """
        song_ids should not be padded
        for *_ratio, use ratio because the lengths of each song are not the same

        song:        BOS (segment A) EOP (segment C) EOP (segment B) EOS (padding)
        """
        tn_song_ids = []
        tn_struct_ids = []
        tn_struct_indices = []
        tn_struct_masks = []
        tn_middle_indices = []
        tn_seg_ids = []
        tn_ignore_labels = []
        tn_expand_idx = []

        if self.infilling:
            for i, song in enumerate(song_ids):
                assert len(song_ids[i]) == len(struct_ids[i]) == len(struct_indices[i])

                # iterate the whole song along with struct
                s_end = 0
                while s_end < len(struct_indices[i]):
                    s_start = s_end
                    while s_end < len(struct_indices[i]) and struct_indices[i][s_end] == struct_indices[i][s_start]:
                        s_end += 1
                    sid = struct_ids[i][s_start]

                    if sid == tokenizer.NONE_ID:
                        continue # skip content without structure
                    if s_start == 0 or s_end == len(song_ids[i]):
                        continue # avoid data without past content or future content

                    p_start = s_start - 1
                    while p_start > 0 and bar_ids[i][s_start] - bar_ids[i][p_start] < bar_range_num:
                        p_start -= 1

                    f_end = s_end + 1
                    while f_end < len(bar_ids[i]) and bar_ids[i][f_end] - bar_ids[i][s_end] < bar_range_num:
                        f_end += 1

                    """
                    p_start(past_start) ------> s_start, s_end ------> f_end(future_end)
                    """
                    no_eos = (song_ids[i][f_end-1] != tokenizer.eos_id())
                    tn_song_ids.append(song_ids[i][p_start:f_end] + ([tokenizer.eos_id()] if no_eos else []))
                    tn_struct_ids.append(struct_ids[i][p_start:f_end] + ([tokenizer.NONE_ID] if no_eos else []))
                    tn_struct_indices.append(struct_indices[i][p_start:f_end] + ([struct_indices[i][f_end-1]+1] if no_eos else []))
                    tn_struct_masks.append(struct_masks[i][p_start:f_end] + ([1] if no_eos else []))
                    tn_middle_indices.append((s_start-p_start, s_end-p_start))
                    tn_expand_idx.append(i)

            (tn_song_ids,
             tn_struct_ids,
             tn_struct_indices,
             tn_struct_masks,
             tn_seg_ids,
             tn_ignore_labels) = self._permute(
                 tn_song_ids,
                 tn_struct_ids,
                 tn_struct_indices,
                 tn_struct_masks,
                 tn_middle_indices,
                 tokenizer,
                 only_middle=only_middle,
                 ignore_middle_first=True,
             )
        else:
            for i, song in enumerate(song_ids):
                seg_ids.append([self.past_id()] * len(song))
                ignore_labels.append([0] * len(song))

        for i in range(len(tn_song_ids)):
            l = len(tn_song_ids[i])
            assert len(tn_struct_ids[i]) == l
            assert len(tn_struct_indices[i]) == l
            assert len(tn_struct_masks[i]) == l
            assert len(tn_seg_ids[i]) == l
            assert len(tn_ignore_labels[i]) == l

        return (
            tn_song_ids,
            tn_struct_ids,
            tn_struct_indices,
            tn_struct_masks,
            tn_seg_ids,
            tn_ignore_labels,
            tn_expand_idx,
        )

    def _permute(self, song_ids, struct_ids, struct_indices, struct_masks, middle_indices, tokenizer, only_middle=False, ignore_middle_first=False):
        """
        song_ids should not be padded
        for *_ratio, use ratio because the lengths of each song are not the same

        song:        BOS (segment A) EOP (segment C) EOP (segment B) EOS (padding)
        """
        assert isinstance(song_ids, list)
        seg_ids = []
        ignore_labels = []

        if self.infilling:
            for i, song in enumerate(song_ids):
                B_start = middle_indices[i][0]
                B_end = middle_indices[i][1]
                assert B_start != B_end

                tmp = []
                s_tmp = []
                si_tmp = []
                m_tmp = []
                seg = []
                ignore = []

                tmp.extend(song_ids[i][:B_start] + [tokenizer.eop_id()]) # A
                s_tmp.extend(struct_ids[i][:B_start] + [tokenizer.NONE_ID])
                if struct_indices is not None:
                    si_tmp.extend(struct_indices[i][:B_start] + [tokenizer.NONE_ID])
                if struct_masks is not None:
                    m_tmp.extend(struct_masks[i][:B_start] + [1])
                seg.extend([self.past_id()] * (len(tmp)-len(seg)))
                if only_middle:
                    ignore.extend([1] * (len(tmp)-len(ignore)))
                else:
                    ignore.extend([0] * (len(tmp)-len(ignore)))
                    ignore[-1] = 1 # EOP

                tmp.extend(song_ids[i][B_end: -1] + [tokenizer.eop_id()]) # C without EOS
                s_tmp.extend(struct_ids[i][B_end: -1] + [tokenizer.NONE_ID])
                if struct_indices is not None:
                    si_tmp.extend(struct_indices[i][B_end: -1] + [tokenizer.NONE_ID])
                if struct_masks is not None:
                    m_tmp.extend(struct_masks[i][B_end: -1] + [1])
                seg.extend([self.future_id()] * (len(tmp)-len(seg)))
                ignore.extend([1] * (len(tmp)-len(ignore)))

                tmp.extend(song_ids[i][B_start: B_end] + [song[-1]]) # B + EOS
                s_tmp.extend(struct_ids[i][B_start: B_end] + [tokenizer.NONE_ID])
                if struct_indices is not None:
                    si_tmp.extend(struct_indices[i][B_start: B_end] + [tokenizer.NONE_ID])
                if struct_masks is not None:
                    m_tmp.extend(struct_masks[i][B_start: B_end] + [1])
                seg.extend([self.middle_id()] * (len(tmp)-len(seg)))
                if ignore_middle_first:
                    ignore.extend([1] + ([0] * (len(tmp)-len(ignore)-1)))
                else:
                    ignore.extend([0] * (len(tmp)-len(ignore)))

                song_ids[i] = tmp
                struct_ids[i] = s_tmp
                if struct_indices is not None:
                    struct_indices[i] = s_tmp
                if struct_masks is not None:
                    struct_masks[i] = m_tmp
                seg_ids.append(seg)
                ignore_labels.append(ignore)
        else:
            for i, song in enumerate(song_ids):
                seg_ids.append([self.past_id()] * len(song))
                ignore_labels.append([0] * len(song))

        return song_ids, struct_ids, struct_indices, struct_masks, seg_ids, ignore_labels

    def past_id(self):
        return self.seg_id_A

    def future_id(self):
        return self.seg_id_C

    def middle_id(self):
        return self.seg_id_B
