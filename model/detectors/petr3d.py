import copy
from typing import Optional

import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from scipy.optimize import linear_sum_assignment

from model.utils.misc import multi_apply
from model.utils.module import GridMask


class Petr3D(BaseModel):
    def __init__(self,
                 num_cls=10,
                 num_obj=300,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 decoder=None,
                 detect_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 cls_loss=None,
                 cls_cost=None,
                 box_loss=None,
                 box_cost=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_cls = num_cls
        self.num_obj = num_obj
        self.use_grid_mask = use_grid_mask
        self.detect_range = detect_range
        if self.use_grid_mask:
            self.grid_mask = GridMask()
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.decoder = MODELS.build(decoder)
        self.cls_loss = MODELS.build(cls_loss)
        self.box_loss = MODELS.build(box_loss)
        self.cls_cost = MODELS.build(cls_cost)
        self.box_cost = MODELS.build(box_cost)

    def init_weights(self):
        self._is_init = True

    def extract_feat(self, img):
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.shape
            img_feats_reshaped.append(img_feat.view(b, n, c, h, w))
        return img_feats_reshaped

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.forward_train(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_test(inputs, data_samples)
        return None

    def forward_test(self, inputs, data_samples=None):
        # 实现默认batch size为1，多batch size暂不支持
        img_feats = self.extract_feat(inputs)
        inter_class, inter_coord = self.decoder(img_feats[0], data_samples, return_all=False)
        inter_class = inter_class.flatten(0, 1).sigmoid()  # 900*10

        scores, index = inter_class.view(-1).topk(self.num_obj)
        labels = index % self.num_cls
        box_ind = index // self.num_cls
        inter_coord = inter_coord.flatten(0, 1)[box_ind]

        # 归一化
        inter_coord[..., 0:1] = inter_coord[..., 0:1] * (self.detect_range[3] - self.detect_range[0]) + \
                                self.detect_range[0]
        inter_coord[..., 1:2] = inter_coord[..., 1:2] * (self.detect_range[4] - self.detect_range[1]) + \
                                self.detect_range[1]
        inter_coord[..., 2:3] = inter_coord[..., 2:3] * (self.detect_range[5] - self.detect_range[2]) + \
                                self.detect_range[2]
        inter_coord[..., 3:6] = torch.exp(inter_coord[..., 3:6])
        inter_coord[..., 6:7] = inter_coord[..., 6:7] * torch.pi / 2

        predictions_dict = {
            'bboxes': inter_coord,
            'scores': scores,
            'labels': labels
        }
        return [predictions_dict]

    def forward_train(self, inputs: torch.Tensor, data_samples=None):
        img_feats = self.extract_feat(inputs)
        img_feats = img_feats[0]  # 不使用多层次特征
        # petr 结构中不将特征转换到bev域下
        # decoder
        inter_classes, inter_coords = self.decoder(img_feats, data_samples, return_all=True)
        # loss
        loss = self.loss(inter_classes,
                         inter_coords,
                         data_samples)
        return loss

    def loss(self, inter_classes, inter_coords, data_samples=None):
        # 真值
        gt_label = []
        gt_bbox = []
        gt_velocity = []
        for img_meta in data_samples:
            gt_label.append(img_meta.gt_label)
            gts = img_meta.gt_bbox_3d
            # 归一化处理
            gts[..., 0:1] = (gts[..., 0:1] - self.detect_range[0]) / (self.detect_range[3] - self.detect_range[0])
            gts[..., 1:2] = (gts[..., 1:2] - self.detect_range[1]) / (self.detect_range[4] - self.detect_range[1])
            gts[..., 2:3] = (gts[..., 2:3] - self.detect_range[2]) / (self.detect_range[5] - self.detect_range[2])
            gts[..., 3:6] = torch.log(gts[..., 3:6])
            gts[..., 6:7] = gts[..., 6:7] * 2 / torch.pi
            # 速度没有归一化，如果需要归一化，使用什么方法？
            gt_bbox.append(gts)
            gt_velocity.append(img_meta.gt_velocity)

        n = range(self.decoder.num_layers)
        gt_label = [copy.deepcopy(gt_label) for _ in n]
        gt_bbox = [copy.deepcopy(gt_bbox) for _ in n]
        gt_velocity = [copy.deepcopy(gt_velocity) for _ in n]

        cls_loss, bbox_loss = multi_apply(self.loss_single,
                                          inter_classes, inter_coords,
                                          gt_label, gt_bbox, gt_velocity)
        loss = dict()
        for i in n:
            loss[f'd{i}.cls_loss'] = cls_loss[i]
            loss[f'd{i}.bbox_loss'] = bbox_loss[i]

        return loss

    def loss_single(self, p_cls, p_box, gt_cls, gt_bbox, gt_velocity):
        label, pt_box, t_box, box_weights, num_gts = multi_apply(self.get_target,
                                                                 p_cls, p_box,
                                                                 gt_cls, gt_bbox, gt_velocity)
        num_gts = sum(num_gts)
        avg_factor = max(1, num_gts)
        label = torch.cat(label, dim=0)
        pt_box = torch.cat(pt_box, dim=0)
        t_box = torch.cat(t_box, dim=0)
        box_weights = torch.cat(box_weights, dim=0)

        p_cls = p_cls.flatten(0, 1)
        cls_loss = self.cls_loss(p_cls, label, avg_factor=avg_factor)

        if num_gts > 0:
            box_loss = self.box_loss(pt_box, t_box, box_weights, avg_factor=avg_factor)
        else:
            box_loss = torch.tensor(0., device=p_box.device)

        return torch.nan_to_num(cls_loss), torch.nan_to_num(box_loss)

    def get_target(self, p_cls, p_box, gt_cls, gt_bbox, gt_velocity):
        num_gts = gt_cls.shape[0]
        num_pred = p_cls.shape[0]
        gt_cls = gt_cls.to(p_cls.device)
        gt_bbox = gt_bbox.to(p_cls.device)
        gt_velocity = gt_velocity.to(p_cls.device)

        assigned_label = torch.full((num_pred,), self.num_cls, dtype=torch.long, device=p_cls.device)
        box_weights = torch.ones_like(gt_bbox)

        if num_gts == 0:
            return assigned_label, torch.zeros_like(gt_bbox), torch.zeros_like(gt_bbox), box_weights, num_gts
        else:
            # class cost
            cls_cost = self.cls_cost(p_cls.detach(), gt_cls)
            # bbox cost
            bbox_cost = self.box_cost(p_box[:, :7].detach(), gt_bbox[:, :7])

            cost = cls_cost + bbox_cost
            cost = torch.nan_to_num(cost, 1e5, 1e5, -1e5)

            matched_row_ind, matched_col_ind = linear_sum_assignment(cost.detach().cpu())
            matched_row_ind = torch.from_numpy(matched_row_ind).to(p_cls.device)
            matched_col_ind = torch.from_numpy(matched_col_ind).to(p_cls.device)

            assigned_label[matched_row_ind] = gt_cls[matched_col_ind]

            p_box = p_box[matched_row_ind]
            gt_bbox = gt_bbox[matched_col_ind]
            # 剔除
            box_weights[gt_velocity[matched_col_ind], 7:9] = 0

            return assigned_label, p_box, gt_bbox, box_weights, num_gts
