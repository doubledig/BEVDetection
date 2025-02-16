import copy

import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmengine.model import BaseModule, bias_init_with_prob
from torch import nn

from model.utils.module import inverse_sigmoid


class MapTrDecoder(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_vec=50,
                 num_vec_len=20,
                 bev_h=200,
                 bev_w=100,
                 num_cls=3,
                 transformerlayers=None,
                 num_layers=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_vec = num_vec
        self.num_vec_len = num_vec_len
        self.num_cls = num_cls

        self.ins_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
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
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(self,
                bev_feat: torch.Tensor,
                return_all: bool = False):
        b, _, emb_dim = bev_feat.shape
        device = bev_feat.device

        ins_embeds = self.ins_embedding.weight.unsqueeze(1)
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        object_embed = (ins_embeds + pts_embeds).flatten(0, 1)
        query, query_pos = torch.split(object_embed, self.embed_dims, dim=1)
        query = query.unsqueeze(0).expand(b, -1, -1)
        query_pos = query_pos.unsqueeze(0).expand(b, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

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
                spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=device),
                level_start_index=torch.tensor([0], device=device),
            )

            tmp = self.reg_branches[i](query)
            reference_points = tmp + inverse_sigmoid(reference_points)
            reference_points = reference_points.sigmoid()

            if return_all:
                inter_score = self.cls_branches[i](query.view(b, self.num_vec, -1, emb_dim).mean(2))
                inter_scores.append(inter_score)
                inter_points.append(reference_points.view(b, self.num_vec, -1, 2))
            reference_points = reference_points.detach()
        if return_all:
            return inter_scores, inter_points
        inter_score = self.cls_branches[-1](query.reshape(b, self.num_vec, -1, emb_dim).mean(2))
        return inter_score, reference_points.view(b, self.num_vec, -1, 2)
