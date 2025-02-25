import copy

import torch
from torch import nn
from mmengine import MODELS

from model.detectors.maptr import MapTR
from model.utils.misc import multi_apply


class MapTRv2(MapTR):
    def __init__(self,
                 embed_dims=256,
                 bev_h=200,
                 bev_w=100,
                 use_lss=True,
                 use_depth=True,
                 depth_loss_weight=3.0,
                 use_pv=True,
                 pv_loss=None,
                 use_bev=True,
                 bev_loss=None,
                 many_k=6,
                 many_weight=1.0,
                 **kwargs):
        MapTR.__init__(self, **kwargs)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_lss = use_lss
        self.use_depth = use_depth
        self.depth_loss_weight = depth_loss_weight

        self.use_pv = use_pv
        self.pv_loss = MODELS.build(pv_loss)
        self.use_bev = use_bev
        self.bev_loss = MODELS.build(bev_loss)
        self.many_k = many_k
        self.many_weight = many_weight

        if self.use_pv:
            self.pv_seg = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0)
            )
        if self.use_bev:
            self.bev_seg = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0)
            )

    def forward_test(self, inputs: torch.Tensor, data_samples=None):
        img_feats = self.extract_feat(inputs)
        if self.use_lss:
            bev_feat, _ = self.encoder(img_feats, data_samples)
        else:
            bev_feat = self.encoder(img_feats, data_samples)
        inter_scores, inter_points = self.decoder(bev_feat, return_all=False)
        inter_scores = inter_scores.flatten(0, 1).sigmoid()
        inter_points = inter_points.flatten(0, 1)

        scores, index = inter_scores.view(-1).topk(self.num_obj)
        labels = index % self.num_cls
        pts_idx = index // self.num_cls
        inter_points = inter_points[pts_idx]

        inter_points[:, :, 0:1] = (inter_points[:, :, 0:1] - 0.5) * self.real_w
        inter_points[:, :, 1:2] = (inter_points[:, :, 1:2] - 0.5) * self.real_h

        predictions_dict = {
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'pts': inter_points.cpu(),
        }
        return [predictions_dict]

    def forward_train(self, inputs: torch.Tensor, data_samples=None):
        img_feats = self.extract_feat(inputs)
        img_feats = img_feats[0]  # 不使用多层次特征
        # pv -> bev
        if self.use_lss:
            bev_feat, depth = self.encoder(img_feats, data_samples)
        else:
            bev_feat = self.encoder(img_feats, data_samples)
            depth = None
        # 解码
        inter_scores, inter_points = self.decoder(bev_feat, return_all=True)
        # loss
        loss = self.loss(inter_scores,
                         inter_points,
                         data_samples,
                         depth=depth,
                         img_feats=img_feats,
                         bev_feat=bev_feat)
        return loss

    def loss(self,
             inter_scores,
             inter_points,
             img_metas,
             depth=None,
             img_feats=None,
             bev_feat=None):
        loss = dict()
        gt_label = []
        gt_pts = []
        gt_depth = []
        gt_bev = []
        gt_pv = []
        for img_meta in img_metas:
            gt_label.append(img_meta.gt_label)
            gt_pts.append(img_meta.gt_pts)
            gt_depth.append(img_meta.depth)
            gt_bev.append(img_meta.gt_bev)
            gt_pv.append(img_meta.gt_pv)
        b, num_cam, _, _, _ = img_feats.shape
        if self.use_pv:
            gt_pv = torch.stack(gt_pv).to(torch.float).to(img_feats.device)
            seg_pv = self.pv_seg(img_feats.flatten(0, 1))
            seg_pv = seg_pv.reshape(b, num_cam, -1, gt_pv.shape[-2], gt_pv.shape[-1])
            pv_loss = self.pv_loss(seg_pv, gt_pv)
            loss['pv_loss'] = torch.nan_to_num(pv_loss)
        if self.use_bev:
            seg_bev = bev_feat.reshape(b, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            seg_bev = self.bev_seg(seg_bev)
            gt_bev = torch.stack(gt_bev).to(torch.float32).to(seg_bev.device)
            bev_loss = self.bev_loss(seg_bev, gt_bev)
            loss['bev_loss'] = torch.nan_to_num(bev_loss)
        if self.use_lss and self.use_depth:
            gt_depth = torch.stack(gt_depth).to(depth.device)
            depth_loss = self.encoder.get_depth_loss(gt_depth, depth)
            loss['depth_loss'] = depth_loss * self.depth_loss_weight

        n = range(self.decoder.num_layers)
        one_gt_label = [copy.deepcopy(gt_label) for _ in n]
        one_gt_pts = [copy.deepcopy(gt_pts) for _ in n]
        one_scores = [scores[:, :self.num_obj] for scores in inter_scores]
        one_points = [points[:, :self.num_obj] for points in inter_points]
        cls_loss, pts_loss, dir_loss = multi_apply(self.loss_single,
                                                   one_scores, one_points,
                                                   one_gt_label, one_gt_pts)
        for i in n:
            loss[f'd{i}.cls_loss'] = cls_loss[i]
            loss[f'd{i}.pts_loss'] = pts_loss[i]
            loss[f'd{i}.dir_loss'] = dir_loss[i]
        m_gt_label = []
        m_gt_pts = []
        for i in n:
            m_gt_label.append(gt_label[i].repeat([self.many_k]))
            m_gt_pts.append(gt_pts[i].repeat([self.many_k, 1, 1, 1]))
        m_gt_label = [copy.deepcopy(m_gt_label) for _ in n]
        m_gt_pts = [copy.deepcopy(m_gt_pts) for _ in n]
        m_scores = [scores[:, self.num_obj:] for scores in inter_scores]
        m_points = [points[:, self.num_obj:] for points in inter_points]
        cls_loss, pts_loss, dir_loss = multi_apply(self.loss_single,
                                                   m_scores, m_points,
                                                   m_gt_label, m_gt_pts)
        for i in range(len(cls_loss)):
            loss[f'd{i}.cls_loss_m'] = cls_loss[i] * self.many_weight
            loss[f'd{i}.pts_loss_m'] = pts_loss[i] * self.many_weight
            loss[f'd{i}.dir_loss_m'] = dir_loss[i] * self.many_weight
        return loss