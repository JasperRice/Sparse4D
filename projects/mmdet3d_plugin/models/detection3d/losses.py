import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.box3d import *


@LOSSES.register_module()
class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ):
        """构建损失函数

        Args:
            loss_box (dict): L1Loss, loss_weight=0.25
            loss_centerness (dict, optional): CrossEntropyLoss, use_sigmoid=True.
                - 标准的 sigmoid + Binary Cross-entropy
                    pred_sigmoid = torch.sigmoid(pred)
                    loss = -[target * log(pred_sigmoid) + (1-target) * log(1-pred_sigmoid)]
                    loss = loss.mean()  # 或 loss.sum() / avg_factor
                - 没有额外的 focal 权重
                - 适合回归性质 (结果在0~1) 的任务
                Defaults to None.
            loss_yawness (dict, optional): GaussianFocalLoss. Defaults to None.
                如果 y == 1 (正样本): loss = -α * (1 - p)^γ * log(p), 与 FocalLoss 相同
                如果 y == 0 (负样本): loss = -α * p^γ * log(1 - p), 使用 p^γ 而不是 (1 - p)^γ
                - p 是预测概率 (经过 sigmoid)
                - y 是目标 (0 或 1)
                - α 是平衡因子 (通常默认为 2.0)
                - γ 是 focal 参数 (通常默认为 4.0)
                这中设计使得负样本的损失在预测值接近 0 时更小, 符合高速分布的特性
            cls_allow_reverse (list, optional): 类别反转允许列表. Defaults to None.
        """
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_box = build(loss_box, LOSSES)
        self.loss_cns = build(loss_centerness, LOSSES)
        self.loss_yns = build(loss_yawness, LOSSES)
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # ! 1. 处理方向反转 (如 barrier 类别)
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            # ! 使用 cosine similarity, 如果小于0, 则代表夹角大于90°
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            # ! 筛选允许方向反转的类别
            if_reverse = (
                torch.isin(cls_target, cls_target.new_tensor(self.cls_allow_reverse))
                & if_reverse
            )
            # ! 将夹角大于90°并且允许方向反转的 3D 框反转
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        # ! 2. 计算边界框损失
        box_loss = self.loss_box(box, box_target, weight=weight, avg_factor=avg_factor)
        output[f"loss_box{suffix}"] = box_loss

        # ! 3. 计算中心度损失 (质量估计)
        if quality is not None:
            # ! 位置质量
            # 预测的位置质量 (cns) 应该和实际的位置质量 (cns_target) 一致
            cns = quality[..., CNS]
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss

            # ! 角度质量 (二分类问题)
            # 预测的角度质量 (yns) 应该和实际的角度质量 (yns_target) 一致
            # yns_target 为 0 或 1
            yns = quality[..., YNS].sigmoid()
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"loss_yns{suffix}"] = yns_loss
        return output
