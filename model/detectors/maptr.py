import copy
from typing import Optional

import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from scipy.optimize import linear_sum_assignment

from model.losses.loss import FocalCost, PtsL1Cost
from model.utils.misc import multi_apply
from model.utils.module import GridMask


class MapTR(BaseModel):
    def __init__(self,
                 num_obj=50,
                 num_cls=3,
                 real_box=(30, 60, 4),
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 encoder=None,
                 decoder=None,
                 cls_loss=None,
                 pts_loss=None,
                 dir_loss=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask()
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        self.cls_loss = MODELS.build(cls_loss)
        self.cls_loss_t = FocalCost(weight=cls_loss['loss_weight'])
        self.pts_loss = MODELS.build(pts_loss)
        self.pts_loss_t = PtsL1Cost(weight=pts_loss['loss_weight'])
        self.dir_loss = MODELS.build(dir_loss)

        self.num_obj = num_obj
        self.num_cls = num_cls
        self.real_h = real_box[1]
        self.real_w = real_box[0]

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.forward_train(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_test(inputs, data_samples)
        return None

    def forward_test(self, inputs: torch.Tensor, data_samples=None):
        img_feats = self.extract_feat(inputs)
        bev_feat = self.encoder(img_feats, data_samples)
        inter_scores, inter_points = self.decoder(bev_feat, return_all=False)
        inter_scores = inter_scores.flatten(0, 1).sigmoid()
        inter_points = inter_points.flatten(0, 1)

        scores, index = inter_scores.view(-1).topk(50)
        labels = index % self.num_cls
        pts_idx = index // self.num_cls
        inter_points = inter_points[pts_idx]

        inter_points[:, :, 0:1] = (inter_points[:, :, 0:1] - 0.5) * self.real_w
        inter_points[:, :, 1:2] = (inter_points[:, :, 1:2] - 0.5) * self.real_h

        return [scores.cpu(), labels.cpu(), inter_points.cpu()]

    def extract_feat(self, img):
        # backbone fpn
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)
        _, c, h, w = img_feats[0].shape
        img_feats = img_feats[0].reshape(b, n, c, h, w)
        return img_feats

    def forward_train(self, inputs: torch.Tensor, data_samples=None):
        img_feats = self.extract_feat(inputs)
        # 时序未实现
        # pv -> bev
        bev_feat = self.encoder(img_feats, data_samples)
        # 解码
        inter_scores, inter_points = self.decoder(bev_feat, return_all=True)

        # loss
        loss = self.loss(inter_scores, inter_points, data_samples)

    def loss(self,
             inter_scores,
             inter_points,
             img_metas):
        loss = dict()
        gt_label = []
        gt_pts = []
        for img_meta in img_metas:
            gt_label.append(img_meta.gt_label)
            gt_pts.append(img_meta.gt_pts)

        n = range(self.decoder.num_layers)
        gt_label = [copy.deepcopy(gt_label) for _ in n]
        gt_pts = [copy.deepcopy(gt_pts) for _ in n]

        cls_loss, pts_loss, dir_loss = multi_apply(self.loss_single,
                                                   inter_scores, inter_points,
                                                   gt_label, gt_pts)

        for i in n:
            loss[f'd{i}.cls_loss'] = cls_loss[i]
            loss[f'd{i}.pts_loss'] = pts_loss[i]
            loss[f'd{i}.dir_loss'] = dir_loss[i]

        return loss

    def loss_single(self, p_cls, p_pts, gt_cls, gt_pts):
        label, pts_targets, pts_weights, num_gts = multi_apply(self.get_target,
                                                               p_cls, p_pts,
                                                               gt_cls, gt_pts)
        num_gts = sum(num_gts)
        avg_factor = max(1, num_gts)
        label = torch.cat(label, 0)
        pts_targets = torch.cat(pts_targets, 0)
        pts_weights = torch.cat(pts_weights, 0)

        p_cls = p_cls.flatten(0, 1)
        cls_loss = self.cls_loss(p_cls, label, avg_factor=avg_factor)

        p_pts = p_pts.flatten(0, 1)
        pts_loss = self.pts_loss(
            p_pts,
            pts_targets,
            pts_weights,
            avg_factor=avg_factor
        )

        dir_weights = pts_weights[:, :-1, 0]
        d_pts = p_pts[:, 1:, :] - p_pts[:, :-1, :]
        d_targe = pts_targets[:, 1:, :] - pts_targets[:, :-1, :]
        dir_loss = self.dir_loss(d_pts,
                                 d_targe,
                                 dir_weights,
                                 avg_factor=avg_factor)

        return torch.nan_to_num(cls_loss), torch.nan_to_num(pts_loss), torch.nan_to_num(dir_loss)

    def get_target(self, p_cls, p_pts, gt_cls, gt_pts):
        num_gts = gt_cls.shape[0]
        num_pred = p_cls.shape[0]
        gt_pts = gt_pts.to(p_pts.device)
        gt_cls = gt_cls.to(p_cls.device)

        assigned_label = torch.full((num_pred,), self.num_cls, dtype=torch.long, device=p_cls.device)
        pts_targets = torch.zeros_like(p_pts)
        pts_weights = torch.zeros_like(p_pts)

        if num_gts == 0:
            return assigned_label, pts_targets, pts_weights, num_gts
        else:
            cls_cost = self.cls_loss_t(p_cls.detach(), gt_cls)

            gt_pts[..., 0:1] = gt_pts[..., 0:1] / self.real_h + 0.5
            gt_pts[..., 1:2] = gt_pts[..., 1:2] / self.real_w + 0.5
            pts_cost = self.pts_loss_t(p_pts.detach(), gt_pts).reshape(num_pred, num_gts, -1)
            pts_cost, order_index = torch.min(pts_cost, 2)

            cost = cls_cost + pts_cost
            # 对 cost 矩阵进行处理，替换 NaN 为一个非常大的数字
            cost = torch.nan_to_num(cost, 1e5, 1e5, -1e5)
            # 匈牙利匹配，转到cpu上
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(p_cls.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(p_cls.device)

            assigned_label[matched_row_inds] = gt_cls[matched_col_inds]

            pts_targets[matched_row_inds] = gt_pts[matched_col_inds, order_index[matched_row_inds, matched_col_inds]]
            pts_weights[matched_row_inds] = 1.0

            return assigned_label, pts_targets, pts_weights, num_gts
