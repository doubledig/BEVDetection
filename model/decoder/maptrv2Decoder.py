import copy

import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer, BaseTransformerLayer
from mmengine.model import BaseModule, bias_init_with_prob
from torch import nn

from model.utils.functional import inverse_sigmoid


class MapTrv2Decoder(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_vec=50,
                 num_vec_train=350,
                 num_vec_len=20,
                 bev_h=200,
                 bev_w=100,
                 num_cls=3,
                 transformerlayers=None,
                 num_layers=6,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_vec = num_vec
        self.num_vec_train = num_vec_train
        self.num_vec_len = num_vec_len
        self.num_cls = num_cls

        self.ins_embedding = nn.Embedding(self.num_vec_train, self.embed_dims * 2)
        self.pts_embedding = nn.Embedding(self.num_vec_len, self.embed_dims * 2)

        self.reference_points = nn.Linear(self.embed_dims, 2)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        for layer in transformerlayers:
            self.layers.append(build_transformer_layer(layer))
            self.reg_branches.append(nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, 2)
            ))
            self.cls_branches.append(nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.num_cls)
            ))

        self.init_weights()

    def init_weights(self):
        if not self._is_init:
            nn.init.xavier_uniform_(self.reference_points.weight)
            nn.init.constant_(self.reference_points.bias, 0.0)
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            self._is_init = True

    def forward(self,
                bev_feat: torch.Tensor,
                return_all: bool = False):
        b, _, emb_dim = bev_feat.shape
        device = bev_feat.device

        if self.training:
            num_vec = self.num_vec_train
        else:
            num_vec = self.num_vec

        ins_embeds = self.ins_embedding.weight[:num_vec].unsqueeze(1)
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        object_embed = (ins_embeds + pts_embeds).flatten(0, 1)
        query, query_pos = torch.split(object_embed, self.embed_dims, dim=1)
        query = query.unsqueeze(0).expand(b, -1, -1)
        query_pos = query_pos.unsqueeze(0).expand(b, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        self_attn_mask = torch.ones([num_vec, num_vec], dtype=torch.bool, device=device)
        self_attn_mask[:50, :50] = False
        self_attn_mask[50:, 50:] = False

        inter_scores = []
        inter_points = []

        for i in range(self.num_layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            query = self.layers[i](
                query,
                key=None,
                value=bev_feat,
                query_pos=query_pos,
                reference_points=reference_points_input,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_vec_len=self.num_vec_len,
                spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=device),
                level_start_index=torch.tensor([0], device=device),
            )

            tmp = self.reg_branches[i](query)
            reference_points = tmp + inverse_sigmoid(reference_points)
            reference_points = reference_points.sigmoid()

            if return_all:
                inter_score = self.cls_branches[i](query.view(b, num_vec, -1, emb_dim).mean(2))
                inter_scores.append(inter_score)
                inter_points.append(reference_points.view(b, num_vec, -1, 2))
            reference_points = reference_points.detach()
        if return_all:
            return inter_scores, inter_points
        inter_score = self.cls_branches[-1](query.reshape(b, num_vec, -1, emb_dim).mean(2))
        return inter_score, reference_points.view(b, num_vec, -1, 2)


class MapTrv2DecoderLayer(BaseTransformerLayer):
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                self_attn_mask=None,
                num_vec=50,
                num_vec_len=20,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                n_batch, n_pts, n_dim = query.shape
                if attn_index == 0:
                    query = query.view(n_batch, num_vec, num_vec_len, n_dim).flatten(1, 2)
                    query_pos = query_pos.view(n_batch, num_vec, num_vec_len, n_dim).flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=self_attn_mask,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs
                    )
                    query = query.view(n_batch, num_vec, num_vec_len, n_dim).flatten(0, 1)
                    query_pos = query_pos.view(n_batch, num_vec, num_vec_len, n_dim).flatten(0, 1)
                    attn_index += 1
                    identity = query
                else:
                    query = query.view(n_batch,
                                       num_vec,
                                       num_vec_len,
                                       n_dim).permute(0, 2, 1, 3).contiguous().flatten(1, 2)
                    query_pos = query_pos.view(n_batch,
                                               num_vec,
                                               num_vec_len,
                                               n_dim).permute(0, 2, 1, 3).contiguous().flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs
                    )
                    query = query.view(n_batch,
                                       num_vec,
                                       num_vec_len,
                                       n_dim).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
                    query_pos = query_pos.view(n_batch,
                                               num_vec,
                                               num_vec_len,
                                               n_dim).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
                    attn_index += 1
                    identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query