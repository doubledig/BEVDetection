import copy
from typing import List, Dict

import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer, build_norm_layer
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmengine.model import BaseModule, bias_init_with_prob

from model.utils.module import SinePositionalEncoding3D
from model.utils.functional import inverse_sigmoid, pos2pose3d


class PETRDecoder(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 with_position=True,
                 with_multiview=True,
                 load_depth=True,
                 depth_num=64,
                 depth_start=1,
                 position_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
                 num_layers=6,
                 transformerlayers=None,
                 use_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.embed_dims = in_channels
        self.num_query = num_query
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.load_depth = load_depth
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.position_range = position_range
        self.use_cp = use_cp  # 节省显存标志
        self.bbox_3d_len = 9  # x, y, z, w, l, h, rot, vx, vy
        self.bbox_3d_weight = (1, 1, 1, 1, 1, 1, 1, 0.2, 0.2)

        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
        self.num_layers = num_layers

        self._init_layers(transformerlayers)
        self.init_weights()

    def _init_layers(self, layers: List[Dict]):
        self.layers = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        for layer in layers:
            self.layers.append(build_transformer_layer(layer))
            self.reg_branches.append(nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.bbox_3d_len)
            ))
            self.cls_branches.append(nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.num_classes)
            ))

        self.positional_encoding = SinePositionalEncoding3D(
            self.embed_dims // 2,
            normalize=True,
        )

        self.input_proj = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1)

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.depth_num * 3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.post_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self):
        if not self._is_init:
            nn.init.uniform_(self.reference_points.weight, 0, 1)
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            for m in self.layers.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    nn.init.constant_(m.bias, 0)
            self.is_init = True

    def forward(self, pv_feats: torch.Tensor, img_metas, return_all: bool = False):
        b, num_cam = pv_feats.shape[:2]
        device = pv_feats.device

        i_h, i_w = img_metas[0].img_padshape
        masks = pv_feats.new_ones((b, num_cam, i_h, i_w))
        train2cam = []
        intrinsics = []
        for i in range(b):
            if self.with_position:
                train2cam.append(img_metas[i].ego2cam @ img_metas[i].train2ego)
                intrinsics.append(img_metas[i].intrinsic)
            for j in range(num_cam):
                i_h, i_w = img_metas[i].img_shape[j]
                masks[i, j, :i_h, :i_w] = 0
        if self.with_position:
            train2cam = torch.stack(train2cam).to(torch.float32).to(device)  # b, n, 4, 4
            intrinsics = torch.stack(intrinsics).to(torch.float32).to(device)  # b, n, 3, 3

        pv_feats = self.input_proj(pv_feats.flatten(0, 1))
        c, h, w = pv_feats.shape[-3:]
        pv_feats = pv_feats.view(b, num_cam, c, h, w)
        masks = nn.functional.interpolate(masks, size=(h, w)).to(torch.bool)

        if self.with_multiview:
            pos_embed = self.positional_encoding(masks)
            pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(b, num_cam, c, h, w)
        else:
            pos_embed = []
            for j in range(num_cam):
                xy_embed = self.positional_encoding(masks[:, j, :, :])
                pos_embed.append(xy_embed.unsqueeze(1))
            pos_embed = torch.cat(pos_embed, dim=1)
            pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(b, num_cam, c, h, w)

        if self.with_position:
            coords = self.position_embedding(h, w, i_h, i_w).to(device)  # D, h, w, 4
            coords[..., :2] = coords[..., :2] * torch.where(coords[..., 2:3] > 1e-5, coords[..., 2:3], 1e-5)
            coords = coords.view(1, 1, self.depth_num, h, w, 4, 1).repeat(b, num_cam,  1, 1, 1, 1, 1)
            intrinsics = intrinsics.view(b, num_cam, 1, 1, 1, 3, 3)
            train2cam = train2cam.view(b, num_cam, 1, 1, 1, 4, 4)
            coords[..., :3, :] = torch.linalg.solve(intrinsics, coords[..., :3, :])
            coords = torch.linalg.solve(train2cam, coords)
            coords = coords[..., :3, 0]
            coords[..., 2:3] = (coords[..., 2:3] - self.position_range[0]) / (
                        self.position_range[3] - self.position_range[0])
            coords[..., 1:2] = (coords[..., 1:2] - self.position_range[1]) / (
                        self.position_range[4] - self.position_range[1])
            coords[..., 0:1] = (coords[..., 0:1] - self.position_range[2]) / (
                        self.position_range[5] - self.position_range[2])
            coords = coords.permute(0, 1, 2, 5, 3, 4).contiguous().flatten(2, 3)
            coords = inverse_sigmoid(coords)
            coords = self.position_encoder(coords.flatten(0, 1)).view(b, num_cam, c, h, w)
            pos_embed = pos_embed + coords

        reference_points = self.reference_points.weight
        query_embed = self.query_embedding(pos2pose3d(reference_points))
        reference_points = reference_points.unsqueeze(0).expand(b, -1, -1)

        pv_feats = pv_feats.permute(0, 1, 3, 4, 2).contiguous().view(b, -1, c)
        pos_embed = pos_embed.permute(0, 1, 3, 4, 2).contiguous().view(b, -1, c)
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)
        masks = masks.view(b, -1)
        query = torch.zeros_like(query_embed)

        inter_classes = []
        inter_coords = []
        for i in range(self.num_layers):
            if self.use_cp and self.training:
                query = checkpoint(
                    self.layers[i].forward,
                    use_reentrant=False,
                    query=query,
                    key=pv_feats,
                    value=pv_feats,
                    query_pos=query_embed,
                    key_pos=pos_embed,
                    key_padding_mask=masks
                )
            else:
                query = self.layers[i](
                    query=query,
                    key=pv_feats,
                    value=pv_feats,
                    query_pos=query_embed,
                    key_pos=pos_embed,
                    key_padding_mask=masks
                )
            query = self.post_norm(query)
            if return_all:
                reference = inverse_sigmoid(reference_points)
                inter_class = self.cls_branches[i](query)
                inter_coord = self.reg_branches[i](query)

                inter_coord[..., 0:3] = torch.sigmoid(inter_coord[..., 0:3] + reference)

                inter_classes.append(inter_class)
                inter_coords.append(inter_coord)
        if return_all:
            return inter_classes, inter_coords
        reference = inverse_sigmoid(reference_points)
        inter_class = self.cls_branches[-1](query)
        inter_coord = self.reg_branches[-1](query)
        inter_coord[..., 0:3] = torch.sigmoid(inter_coord[..., 0:3] + reference)
        return inter_class, inter_coord

    def position_embedding(self, h, w, pad_h, pad_w):
        coords_h = torch.arange(h, dtype=torch.float32) * pad_h / h
        coords_w = torch.arange(w, dtype=torch.float32) * pad_w / w
        index = torch.arange(self.depth_num, dtype=torch.float32)
        bin_size = self.position_range[3] - self.depth_start
        if self.load_depth:
            coords_d = self.depth_start + bin_size * index * (index + 1) / (self.depth_num * (self.depth_num + 1))
        else:
            coords_d = self.depth_start + bin_size * index / self.depth_num
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w])).permute(1, 2, 3, 0)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)
        return coords
