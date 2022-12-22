import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )

        iou_loss_type = losses_cfg.get('LOSS_IOU', None)
        if iou_loss_type == 'WeightedSmoothL1Loss':
            self.iou_loss_function = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('iou_code_weights', None)
            )

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def get_seg_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_seg = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_seg = point_loss_seg * loss_weights_dict['point_seg_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_seg': point_loss_seg.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_seg, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        if self.model_cfg.LOSS_CONFIG.LOSS_REG == 'WeightedSmoothL1Loss':
            pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
            point_box_labels = self.forward_ret_dict['point_box_labels']
            point_box_preds = self.forward_ret_dict['point_box_preds']

            reg_weights = pos_mask.float()
            pos_normalizer = pos_mask.sum().float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            point_loss_box_src = self.reg_loss_func(
                point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
            )
            point_loss_box = point_loss_box_src.sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({'point_loss_box': point_loss_box.item()})
        elif self.model_cfg.LOSS_CONFIG.LOSS_REG == 'DIOU_loss':
            pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
            reg_weights = (pos_mask.float()).view(-1)
            pos_normalizer = pos_mask.sum().float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            bboxes1 = self.forward_ret_dict['predict_boxes']
            bboxes2 = self.forward_ret_dict['gt_of_boxes']
            p = torch.isnan(bboxes1).int().sum()
            q = torch.isnan(bboxes2).int().sum()
            # 分别提取出box和GT的长、宽、高、旋转角
            l1, w1, h1, angle1 = bboxes1[:, 3], bboxes1[:, 4], bboxes1[:, 5], bboxes1[:, 6]
            l2, w2, h2, angle2 = bboxes2[:, 3], bboxes2[:, 4], bboxes2[:, 5], bboxes2[:, 6]
            # 分别提取出box和GT的三个坐标点
            center_x1, center_y1, center_z1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2]
            center_x2, center_y2, center_z2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2]

            c_f1, c_b1, c_l1, c_r1, c_f2, c_b2, c_l2, c_r2 = bboxes1.new_zeros(bboxes1.shape[0]), bboxes1.new_zeros(
                bboxes1.shape[0]), bboxes1.new_zeros(bboxes1.shape[0]), bboxes1.new_zeros(
                bboxes1.shape[0]), bboxes1.new_zeros(bboxes1.shape[0]), bboxes1.new_zeros(
                bboxes1.shape[0]), bboxes1.new_zeros(bboxes1.shape[0]), bboxes1.new_zeros(bboxes1.shape[0])
            # box四种旋转角度下的mask
            angle1_mask1, angle1_mask2, angle1_mask3, angle1_mask4 = (angle1 >= 0) & (angle1 < (np.pi)/2), (angle1 >= (np.pi)/2) & (angle1 <= (np.pi)), \
                                                                     (angle1 >= -(np.pi)/2) & (angle1 < 0), (angle1 >= -(np.pi)) & (angle1 < -(np.pi)/2)
            # 将四种mask下的box参数x、y、l、w提取出来
            cur_x1_mask1, cur_y1_mask1, cur_l1_mask1, cur_w1_mask1 = center_x1[angle1_mask1], center_y1[angle1_mask1], l1[angle1_mask1], w1[angle1_mask1]
            cur_x1_mask2, cur_y1_mask2, cur_l1_mask2, cur_w1_mask2 = center_x1[angle1_mask2], center_y1[angle1_mask2], l1[angle1_mask2], w1[angle1_mask2]
            cur_x1_mask3, cur_y1_mask3, cur_l1_mask3, cur_w1_mask3 = center_x1[angle1_mask3], center_y1[angle1_mask3], l1[angle1_mask3], w1[angle1_mask3]
            cur_x1_mask4, cur_y1_mask4, cur_l1_mask4, cur_w1_mask4 = center_x1[angle1_mask4], center_y1[angle1_mask4], l1[angle1_mask4], w1[angle1_mask4]
            # 计算每种mask下外接矩形参数（上下左右边界）
            cur_cf1_mask1 = cur_x1_mask1 + 0.5 * cur_l1_mask1 * torch.cos(angle1[angle1_mask1]) + 0.5 * cur_w1_mask1 * torch.sin(angle1[angle1_mask1])
            cur_cb1_mask1 = cur_x1_mask1 - 0.5 * cur_l1_mask1 * torch.cos(angle1[angle1_mask1]) - 0.5 * cur_w1_mask1 * torch.sin(angle1[angle1_mask1])
            cur_cl1_mask1 = cur_y1_mask1 + 0.5 * cur_l1_mask1 * torch.sin(angle1[angle1_mask1]) + 0.5 * cur_w1_mask1 * torch.cos(angle1[angle1_mask1])
            cur_cr1_mask1 = cur_y1_mask1 - 0.5 * cur_l1_mask1 * torch.sin(angle1[angle1_mask1]) - 0.5 * cur_w1_mask1 * torch.cos(angle1[angle1_mask1])

            cur_cf1_mask2 = cur_x1_mask2 - 0.5 * cur_l1_mask2 * torch.cos(angle1[angle1_mask2]) + 0.5 * cur_w1_mask2 * torch.sin(angle1[angle1_mask2])
            cur_cb1_mask2 = cur_x1_mask2 + 0.5 * cur_l1_mask2 * torch.cos(angle1[angle1_mask2]) - 0.5 * cur_w1_mask2 * torch.sin(angle1[angle1_mask2])
            cur_cl1_mask2 = cur_y1_mask2 + 0.5 * cur_l1_mask2 * torch.sin(angle1[angle1_mask2]) - 0.5 * cur_w1_mask2 * torch.cos(angle1[angle1_mask2])
            cur_cr1_mask2 = cur_y1_mask2 - 0.5 * cur_l1_mask2 * torch.sin(angle1[angle1_mask2]) + 0.5 * cur_w1_mask2 * torch.cos(angle1[angle1_mask2])

            cur_cf1_mask3 = cur_x1_mask3 + 0.5 * cur_l1_mask3 * torch.cos(angle1[angle1_mask3]) - 0.5 * cur_w1_mask3 * torch.sin(angle1[angle1_mask3])
            cur_cb1_mask3 = cur_x1_mask3 - 0.5 * cur_l1_mask3 * torch.cos(angle1[angle1_mask3]) + 0.5 * cur_w1_mask3 * torch.sin(angle1[angle1_mask3])
            cur_cl1_mask3 = cur_y1_mask3 - 0.5 * cur_l1_mask3 * torch.sin(angle1[angle1_mask3]) + 0.5 * cur_w1_mask3 * torch.cos(angle1[angle1_mask3])
            cur_cr1_mask3 = cur_y1_mask3 + 0.5 * cur_l1_mask3 * torch.sin(angle1[angle1_mask3]) - 0.5 * cur_w1_mask3 * torch.cos(angle1[angle1_mask3])

            cur_cf1_mask4 = cur_x1_mask4 - 0.5 * cur_l1_mask4 * torch.cos(angle1[angle1_mask4]) - 0.5 * cur_w1_mask4 * torch.sin(angle1[angle1_mask4])
            cur_cb1_mask4 = cur_x1_mask4 + 0.5 * cur_l1_mask4 * torch.cos(angle1[angle1_mask4]) + 0.5 * cur_w1_mask4 * torch.sin(angle1[angle1_mask4])
            cur_cl1_mask4 = cur_y1_mask4 - 0.5 * cur_l1_mask4 * torch.sin(angle1[angle1_mask4]) - 0.5 * cur_w1_mask4 * torch.cos(angle1[angle1_mask4])
            cur_cr1_mask4 = cur_y1_mask4 + 0.5 * cur_l1_mask4 * torch.sin(angle1[angle1_mask4]) + 0.5 * cur_w1_mask4 * torch.cos(angle1[angle1_mask4])
            # 参数整合
            c_f1[angle1_mask1], c_f1[angle1_mask2], c_f1[angle1_mask3], c_f1[angle1_mask4] = cur_cf1_mask1, cur_cf1_mask2, cur_cf1_mask3, cur_cf1_mask4
            c_b1[angle1_mask1], c_b1[angle1_mask2], c_b1[angle1_mask3], c_b1[angle1_mask4] = cur_cb1_mask1, cur_cb1_mask2, cur_cb1_mask3, cur_cb1_mask4
            c_l1[angle1_mask1], c_l1[angle1_mask2], c_l1[angle1_mask3], c_l1[angle1_mask4] = cur_cl1_mask1, cur_cl1_mask2, cur_cl1_mask3, cur_cl1_mask4
            c_r1[angle1_mask1], c_r1[angle1_mask2], c_r1[angle1_mask3], c_r1[angle1_mask4] = cur_cr1_mask1, cur_cr1_mask2, cur_cr1_mask3, cur_cr1_mask4
            # GT四种旋转角度下的mask
            angle2_mask1, angle2_mask2, angle2_mask3, angle2_mask4 = (angle2 >= 0) & (angle2 < (np.pi) / 2), (angle2 >= (np.pi) / 2) & (angle2 <= (np.pi)), \
                                                                     (angle2 >= -(np.pi) / 2) & (angle2 < 0), (angle2 >= -(np.pi)) & (angle2 < -(np.pi) / 2)
            # 将四种mask下的GT参数x、y、l、w提取出来
            cur_x2_mask1, cur_y2_mask1, cur_l2_mask1, cur_w2_mask1 = center_x2[angle2_mask1], center_y2[angle2_mask1], l2[angle2_mask1], w2[angle2_mask1]
            cur_x2_mask2, cur_y2_mask2, cur_l2_mask2, cur_w2_mask2 = center_x2[angle2_mask2], center_y2[angle2_mask2], l2[angle2_mask2], w2[angle2_mask2]
            cur_x2_mask3, cur_y2_mask3, cur_l2_mask3, cur_w2_mask3 = center_x2[angle2_mask3], center_y2[angle2_mask3], l2[angle2_mask3], w2[angle2_mask3]
            cur_x2_mask4, cur_y2_mask4, cur_l2_mask4, cur_w2_mask4 = center_x2[angle2_mask4], center_y2[angle2_mask4], l2[angle2_mask4], w2[angle2_mask4]
            # 计算每种mask下外接矩形参数（上下左右边界）
            cur_cf2_mask1 = cur_x2_mask1 + 0.5 * cur_l2_mask1 * torch.cos(angle2[angle2_mask1]) + 0.5 * cur_w2_mask1 * torch.sin(angle2[angle2_mask1])
            cur_cb2_mask1 = cur_x2_mask1 - 0.5 * cur_l2_mask1 * torch.cos(angle2[angle2_mask1]) - 0.5 * cur_w2_mask1 * torch.sin(angle2[angle2_mask1])
            cur_cl2_mask1 = cur_y2_mask1 + 0.5 * cur_l2_mask1 * torch.sin(angle2[angle2_mask1]) + 0.5 * cur_w2_mask1 * torch.cos(angle2[angle2_mask1])
            cur_cr2_mask1 = cur_y2_mask1 - 0.5 * cur_l2_mask1 * torch.sin(angle2[angle2_mask1]) - 0.5 * cur_w2_mask1 * torch.cos(angle2[angle2_mask1])

            cur_cf2_mask2 = cur_x2_mask2 - 0.5 * cur_l2_mask2 * torch.cos(angle2[angle2_mask2]) + 0.5 * cur_w2_mask2 * torch.sin(angle2[angle2_mask2])
            cur_cb2_mask2 = cur_x2_mask2 + 0.5 * cur_l2_mask2 * torch.cos(angle2[angle2_mask2]) - 0.5 * cur_w2_mask2 * torch.sin(angle2[angle2_mask2])
            cur_cl2_mask2 = cur_y2_mask2 + 0.5 * cur_l2_mask2 * torch.sin(angle2[angle2_mask2]) - 0.5 * cur_w2_mask2 * torch.cos(angle2[angle2_mask2])
            cur_cr2_mask2 = cur_y2_mask2 - 0.5 * cur_l2_mask2 * torch.sin(angle2[angle2_mask2]) + 0.5 * cur_w2_mask2 * torch.cos(angle2[angle2_mask2])

            cur_cf2_mask3 = cur_x2_mask3 + 0.5 * cur_l2_mask3 * torch.cos(angle2[angle2_mask3]) - 0.5 * cur_w2_mask3 * torch.sin(angle2[angle2_mask3])
            cur_cb2_mask3 = cur_x2_mask3 - 0.5 * cur_l2_mask3 * torch.cos(angle2[angle2_mask3]) + 0.5 * cur_w2_mask3 * torch.sin(angle2[angle2_mask3])
            cur_cl2_mask3 = cur_y2_mask3 - 0.5 * cur_l2_mask3 * torch.sin(angle2[angle2_mask3]) + 0.5 * cur_w2_mask3 * torch.cos(angle2[angle2_mask3])
            cur_cr2_mask3 = cur_y2_mask3 + 0.5 * cur_l2_mask3 * torch.sin(angle2[angle2_mask3]) - 0.5 * cur_w2_mask3 * torch.cos(angle2[angle2_mask3])

            cur_cf2_mask4 = cur_x2_mask4 - 0.5 * cur_l2_mask4 * torch.cos(angle2[angle2_mask4]) - 0.5 * cur_w2_mask4 * torch.sin(angle2[angle2_mask4])
            cur_cb2_mask4 = cur_x2_mask4 + 0.5 * cur_l2_mask4 * torch.cos(angle2[angle2_mask4]) + 0.5 * cur_w2_mask4 * torch.sin(angle2[angle2_mask4])
            cur_cl2_mask4 = cur_y2_mask4 - 0.5 * cur_l2_mask4 * torch.sin(angle2[angle2_mask4]) - 0.5 * cur_w2_mask4 * torch.cos(angle2[angle2_mask4])
            cur_cr2_mask4 = cur_y2_mask4 + 0.5 * cur_l2_mask4 * torch.sin(angle2[angle2_mask4]) + 0.5 * cur_w2_mask4 * torch.cos(angle2[angle2_mask4])
            # 参数整合
            c_f2[angle2_mask1], c_f2[angle2_mask2], c_f2[angle2_mask3], c_f2[angle2_mask4] = cur_cf2_mask1, cur_cf2_mask2, cur_cf2_mask3, cur_cf2_mask4
            c_b2[angle2_mask1], c_b2[angle2_mask2], c_b2[angle2_mask3], c_b2[angle2_mask4] = cur_cb2_mask1, cur_cb2_mask2, cur_cb2_mask3, cur_cb2_mask4
            c_l2[angle2_mask1], c_l2[angle2_mask2], c_l2[angle2_mask3], c_l2[angle2_mask4] = cur_cl2_mask1, cur_cl2_mask2, cur_cl2_mask3, cur_cl2_mask4
            c_r2[angle2_mask1], c_r2[angle2_mask2], c_r2[angle2_mask3], c_r2[angle2_mask4] = cur_cr2_mask1, cur_cr2_mask2, cur_cr2_mask3, cur_cr2_mask4
            # 每个box与对应的GT之间的外接矩形参数
            c_f, c_b, c_l, c_r = torch.max(c_f1, c_f2), torch.min(c_b1, c_b2), torch.max(c_l1, c_l2), torch.min(c_r1,c_r2)
            c_u = torch.max(center_z1 + h1/2, center_z2 + h2/2)
            c_d = torch.min(center_z1 - h1/2, center_z2 - h2/2)
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2 + (center_z2 - center_z1) ** 2
            c_diag = torch.clamp((c_f - c_b), min=0) ** 2 + torch.clamp((c_l - c_r), min=0) ** 2 + torch.clamp((c_u - c_d), min=0) ** 2
            u = (inter_diag)/(c_diag)
            iou = (self.forward_ret_dict['box_ious']).view(-1)
            dious = iou - u - 1.25 * (1 - torch.abs(torch.cos(angle1 - angle2)))
            point_loss_box = 1 - dious
            point_loss_box = (point_loss_box * reg_weights).sum()

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        iou_weights = pos_mask.float()
        point_cls_scores_loss = self.forward_ret_dict['point_cls_scores_loss'].view(-1)
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        box_cls_labels = torch.where(torch.isnan(box_cls_labels), point_cls_scores_loss, box_cls_labels)
        box_cls_labels = box_cls_labels.detach()
        if loss_cfgs.LOSS_CLS == 'BinaryCrossEntropy':
            point_loss_cls = F.binary_cross_entropy(point_cls_scores_loss, box_cls_labels.float(), reduction='none')
            point_loss_cls = (point_loss_cls * iou_weights).sum() / torch.clamp(pos_mask.sum(), min=1.0)
        elif loss_cfgs.LOSS_CLS == 'smooth-l1':
            box_cls_labels = torch.where(torch.isnan(box_cls_labels), point_cls_scores_loss, box_cls_labels)
            point_loss_cls = F.smooth_l1_loss(point_cls_scores_loss[None, ...], box_cls_labels[None, ...].float())
            point_loss_cls = (point_loss_cls * iou_weights).sum() / torch.clamp(pos_mask.sum(), min=1.0)
        point_loss_cls = point_loss_cls * loss_cfgs.LOSS_WEIGHTS['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_cls': point_loss_cls.item()})
        return point_loss_cls, tb_dict

    def get_iou_layer_loss(self, tb_dict=None):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        iou_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        iou_weights /= torch.clamp(pos_normalizer, min=1.0)
        point_iou_preds = self.forward_ret_dict['point_iou_preds'].view(-1, 1)
        point_iou_labels = self.forward_ret_dict['box_ious'].view(-1, 1)
        a = torch.isnan(point_iou_preds).int().sum()
        b = torch.isnan(point_iou_labels).int().sum()
        if loss_cfgs.LOSS_IOU == 'WeightedSmoothL1Loss':
            point_loss_iou = self.iou_loss_function(point_iou_preds[None, ...], point_iou_labels[None, ...], weights=iou_weights[None, ...])
            point_loss_iou = point_loss_iou.sum()
        point_loss_iou = point_loss_iou * loss_cfgs.LOSS_WEIGHTS['point_iou_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_iou': point_loss_iou.item()})
        return point_loss_iou, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
