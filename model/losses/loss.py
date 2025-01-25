import torch
from mmcv.ops.focal_loss import SigmoidFocalLossFunction
from torch import nn


def weight_reduce_loss(loss,
                       weight=None,
                       reduction: str = 'mean',
                       avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = SigmoidFocalLossFunction.apply(pred.contiguous(), target.contiguous(), gamma,
                                          alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class SimpleLoss(nn.Module):
    def __init__(self, loss_weight, pos_weight=0):
        super().__init__()
        self.loss_weight = loss_weight
        self.pos_weight = nn.Parameter(torch.tensor([pos_weight]), requires_grad=False)

    def forward(self, pred, tgt):
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, tgt, pos_weight=self.pos_weight
        )
        return loss * self.loss_weight


class PtsL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):
        if weight.sum() == 0:
            return pred.sum() * 0
        dists = nn.functional.l1_loss(pred, target, reduction='none')
        dists = dists * weight
        eps = torch.finfo(torch.float32).eps
        dists = dists.sum() / (avg_factor + eps)
        return dists * self.loss_weight

class PtsKpLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=5.0, line_weight=6.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.line_weight = line_weight

    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None):
        if mask.sum() == 0:
            return pred.sum() * 0
        weight = weight.to(torch.bool)[:, 0, 0]
        pred = pred[weight]
        target = target[weight]
        mask = mask[weight]
        # 关键点
        dists = nn.functional.l1_loss(pred, target, reduction='none').sum(dim=-1)
        dists = dists * self.loss_weight
        l_line = target.roll(1, dims=-2)
        r_line = target.roll(-1, dims=-2)
        a = r_line[..., 1] - l_line[..., 1]
        b = l_line[..., 0] - r_line[..., 0]
        c = l_line[..., 1] * r_line[..., 0] - l_line[..., 0] * r_line[..., 1]
        aa = a * a
        bb = b * b
        ab = a * b
        target_t = torch.zeros_like(target)
        target_t[..., 0] = (bb * pred[..., 0].detach() - ab * pred[..., 1].detach() - a * c) / (aa + bb)
        target_t[..., 1] = (aa * pred[..., 1].detach() - ab * pred[..., 0].detach() - b * c) / (aa + bb)
        line_dists = nn.functional.l1_loss(pred, target_t, reduction='none').sum(dim=-1)
        dists[mask] = line_dists[mask] * self.line_weight # + dists[mask]
        eps = torch.finfo(torch.float32).eps
        dists = dists.sum() / (avg_factor + eps)
        return dists

class PtsDirCosLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):
        num_samples, num_dir, num_coords = pred.shape
        tgt_param = target.new_ones((num_samples * num_dir))
        loss = nn.functional.cosine_embedding_loss(
            pred.flatten(0, 1),
            target.flatten(0, 1),
            tgt_param,
            reduction='none'
        )
        loss = loss.view(num_samples, num_dir) * weight
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
        return loss * self.loss_weight


class PtsL1Cost:
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_pts = gt_bboxes.shape[-2]
        num_coords = gt_bboxes.shape[-1]
        bbox_pred = bbox_pred.unsqueeze(1)
        gt_bboxes = gt_bboxes.view(1, -1, num_pts, num_coords)
        dists = (bbox_pred - gt_bboxes).abs().sum(dim=(-1, -2))
        return dists * self.weight

class PtsKpCost:
    def __init__(self, loss_weight=5.0, line_weight=6.0):
        self.loss_weight = loss_weight
        self.line_weight = line_weight

    def __call__(self, bbox_pred, gt_bboxes, gt_masks):
        num_pts = gt_bboxes.shape[-2]
        num_coords = gt_bboxes.shape[-1]
        bbox_pred = bbox_pred.unsqueeze(1)
        gt_bboxes = gt_bboxes.view(1, -1, num_pts, num_coords)
        dists = (bbox_pred - gt_bboxes).abs().sum(dim=-1) * self.loss_weight
        gt_masks = gt_masks.view(-1, num_pts)
        # 计算l1距离
        l_line = gt_bboxes.roll(1, dims=-2)
        r_line = gt_bboxes.roll(-1, dims=-2)
        a = r_line[..., 1] - l_line[..., 1]
        b = l_line[..., 0] - r_line[..., 0]
        c = l_line[..., 1] * r_line[..., 0] - l_line[..., 0] * r_line[..., 1]
        aa = a * a
        bb = b * b
        ab = a * b
        line_dists = torch.abs(aa * bbox_pred[..., 0] + ab * bbox_pred[..., 1] + a * c) + \
                     torch.abs(bb * bbox_pred[..., 1] + ab * bbox_pred[..., 0] + b * c)
        line_dists = line_dists / (aa + bb + 1e-5)
        dists[:, gt_masks] = line_dists[:, gt_masks] * self.line_weight #+ dists[:, gt_masks]
        return dists.sum(dim=-1)

class FocalCost:
    def __init__(self, weight=2., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
        # neg_cost = -((1 - cls_pred)* 0.9+0.05).log() * (
        #         1 - self.alpha) * cls_pred.pow(self.gamma)
        # pos_cost = -(cls_pred * 0.9 + 0.05).log() * self.alpha * (
        #         1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if pred.dim() == target.dim():
                # this means that target is already in One-Hot form.
                calculate_loss_func = py_sigmoid_focal_loss
            elif torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = nn.functional.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls



