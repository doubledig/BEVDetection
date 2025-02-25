import copy

import torch
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, build_transformer_layer
from mmengine.model import BaseModule
from torch import nn

from model.utils.module import LearnedPositionalEncoding


class BEVFormerTransform(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 bev_h=200,
                 bev_w=100,
                 real_box=(30, 60, 4),
                 num_cams=6,
                 num_points_in_pillar=4,
                 use_shift: bool = True,
                 use_can_bus: bool = True,
                 use_cam_embeds: bool = True,
                 transformerlayer: dict = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_h = real_box[1]
        self.real_w = real_box[0]
        self.real_z = real_box[2]
        self.num_cams = num_cams
        self.num_points_in_pillar = num_points_in_pillar
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.use_cam_embeds = use_cam_embeds

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.cam_embedding = nn.Embedding(self.num_cams, self.embed_dims)
        self.level_embedding = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=self.embed_dims,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w
        )
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.layer = build_transformer_layer(transformerlayer)

        self.ref_3d = self.get_ref_points('3d')
        self.ref_2d = self.get_ref_points('2d')

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.can_bus_mlp.weight)

    def get_ref_points(self, mode):
        if mode == '3d':
            zs = torch.linspace(0.5, self.real_z - 0.5, self.num_points_in_pillar) \
                     .view(-1, 1, 1).expand(-1, self.bev_h, self.bev_w) - self.real_z / 2
            ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h).view(1, -1, 1) \
                     .expand(self.num_points_in_pillar, -1, self.bev_w) / self.bev_h * self.real_h - self.real_h / 2
            xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w).view(1, 1, -1) \
                     .expand(self.num_points_in_pillar, self.bev_h, -1) / self.bev_w * self.real_w - self.real_w / 2
            ref = torch.stack((xs, ys, zs), dim=-1)
            ref = ref.flatten(1, 2).unsqueeze(0)
        else:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, self.bev_h - 0.5, self.bev_w) / self.bev_h,
                torch.linspace(
                    0.5, self.bev_w - 0.5, self.bev_w) / self.bev_w,
                indexing='ij'
            )
            ref = torch.stack((ref_x, ref_y), dim=-1).repeat(1, -1, 1, 2)
        return ref

    def forward(self, pv_feats: torch.Tensor, img_metas, prev_bev=None):
        b, num_cam, c, h, w = pv_feats.shape
        device = pv_feats.device

        bev_queries = self.bev_embedding.weight.unsqueeze(0).expand(b, -1, -1)
        bev_mask = torch.zeros((b, self.bev_h, self.bev_w), device=device)
        bev_pos = self.positional_encoding(bev_mask).flatten(2)

        can_bus = []
        train2cam = []
        intrinsics = []
        for img_meta in img_metas:
            can_bus.append(img_meta.can_bus)
            train2cam.append(img_meta.ego2cam @ img_meta.train2ego)
            intrinsics.append(img_meta.intrinsics)
        can_bus = torch.stack(can_bus).to(torch.float32).to(device)
        train2cam = torch.stack(train2cam).to(torch.float32).to(device)  # b, n, 4, 4
        intrinsics = torch.stack(intrinsics).to(torch.float32).to(device)  # b, n, 3, 3

        if self.use_shift:
            delta_x = can_bus[:, 0]
            delta_y = can_bus[:, 1]
            ego_angle = can_bus[:, -2]
            translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
            translation_angle = torch.arctan2(delta_y, delta_x)
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * torch.cos(bev_angle) / self.real_h
            shift_x = translation_length * torch.sin(bev_angle) / self.real_w
            shift = torch.stack([shift_x, shift_y], dim=1)
        else:
            shift = torch.zeros((b, 2), device=device)

        if self.use_can_bus:
            can_bus = self.can_bus_mlp(can_bus)
            bev_queries = bev_queries + can_bus.view(b, 1, -1)

        spatial_shapes = torch.tensor([[h, w]], device=device, dtype=torch.int)
        bev_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=device, dtype=torch.int)
        level_start_index = torch.tensor([0], device=device, dtype=torch.int)
        pv_feats = pv_feats.flatten(3).permute(0, 1, 3, 2)  # b, num_cam, hw, c
        if self.use_cam_embeds:
            pv_feats = pv_feats + self.cam_embedding.weight.view(1, num_cam, 1, c)
        pv_feats = pv_feats + self.level_embedding.weight.view(1, 1, 1, c)

        ref_3d = self.ref_3d.expand(b, -1, -1, -1).to(device)  # b, 4, hw, 3
        ref_2d = self.ref_2d.expand(b, -1, -1, -1).to(device)  # b, hw, 1, 2

        ref_cam = ref_3d.view(b, self.num_points_in_pillar, 1, -1, 3, 1)
        train2cam = train2cam.view(b, 1, num_cam, 1, 4, 4)
        intrinsics = intrinsics.view(b, 1, num_cam, 1, 3, 3)
        ref_cam = train2cam[..., :3, :3] @ ref_cam + train2cam[..., :3, :3]
        ref_cam = intrinsics @ ref_cam
        ref_cam = ref_cam.squeeze(-1)  # b, 4, num_cam, hw, 3

        bev_mask = ref_cam[..., 2] > 1e-5
        ref_cam = ref_cam[..., :2] / ref_cam[..., 2:3]  # b, 4, num_cam, hw, 2
        ref_cam[..., 0] = ref_cam[..., 0] / img_metas[0].img_padshape[1]
        ref_cam[..., 1] = ref_cam[..., 1] / img_metas[0].img_padshape[0]
        bev_mask = (bev_mask & (ref_cam[..., 0] > 0)
                    & (ref_cam[..., 1] > 0)
                    & (ref_cam[..., 0] < 1)
                    & (ref_cam[..., 1] < 1))
        bev_mask = torch.nan_to_num(bev_mask, False)
        ref_cam = ref_cam.permute(0, 2, 3, 1, 4)  # b, num_cam, hw, 4, 3
        bev_mask = bev_mask.permute(0, 2, 3, 1)  # b, num_cam, hw, 4,

        if prev_bev is None:
            ref_2d = torch.cat([ref_2d, ref_2d], dim=0)
            prev_bev = torch.cat([bev_queries, bev_queries], dim=0)
        else:
            ref_2d = torch.cat([ref_2d + shift.view(b, 1, 1, 2), ref_2d], dim=0)
            prev_bev = torch.cat([prev_bev, bev_queries], dim=0)

        output = self.layer(
            bev_queries,
            key=pv_feats,
            value=pv_feats,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            bev_pos=bev_pos,
            ref_2d=ref_2d,
            bev_shapes=bev_shapes,
            ref_cam=ref_cam,
            spatial_shapes=spatial_shapes,
            bev_mask=bev_mask,
        )

        return output


class BEVFormerLayer(BaseTransformerLayer):
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                spatial_shapes=None,
                prev_bev=None,
                bev_pos=None,
                ref_2d=None,
                ref_cam=None,
                bev_shapes=None,
                level_start_index=None,
                bev_mask=None,
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
                query = self.attentions[attn_index](
                    query,
                    value=prev_bev,
                    identity=identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=bev_shapes,
                    level_start_index=level_start_index,
                    **kwargs
                )
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
                    identity=identity if self.pre_norm else None,
                    reference_points=ref_cam,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_mask=bev_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
