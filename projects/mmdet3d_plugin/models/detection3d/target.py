import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from scipy.optimize import linear_sum_assignment

from projects.mmdet3d_plugin.core.box3d import *

from ..base_target import BaseTargetWithDenoising

__all__ = ["CustomizedSparseBox3DTarget", "SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(num_dn_groups, num_temp_dn_groups)
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,  # 类别预测 [batch_size, num_predictions, num_classes]
        box_pred,  # 3D框预测 [batch_size, num_predictions, 11]
        cls_target,  # 类别目标 (list of tensors)
        box_target,  # 3D框目标 (list of tensors)
    ):
        """基于代价的匈牙利匹配器, 用于将模型的预测结果与真实标注进行最优匹配.
        这是稀疏3D目标检测的关键步骤, 决定了哪些预测应该负责监督哪些真实目标.

        核心工作流程:
        ------------
        1. 分类代价计算 (_cls_cost):
           - 使用Focal Loss变体计算分类代价
           - 公式: cost = (pos_cost - neg_cost) * cls_weight
           - 使用alpha和gamma参数控制难易样本的权重

        2. 回归目标编码 (encode_reg_target):
           - 位置(X, Y, Z): 保持不变
           - 尺寸(W, L, H): 取对数log()
           - 朝向角: 分解为sin(yaw)和cos(yaw) (避免周期性不连续）
           - 速度(VX, VY, VZ): 保持不变

        3. 实例级回归权重计算:
           - 标记非NaN的维度 (有效维度)
           - 支持类别相关的回归权重

        4. 回归代价计算 (_box_cost):
           - 计算预测框与目标框之间的L1距离并加权
           - cost = sum(|pred - target| * instance_weights * reg_weights) * box_weight

        5. 匈牙利算法匹配:
           - 合并分类和回归代价
           - 使用scipy.optimize.linear_sum_assignment求解最优二分图匹配
           - 处理无效值 (-inf或NaN替换为1e8)

        6. 构建输出目标:
           - output_cls_target: 匹配成功的设为目标类别, 未匹配的设为num_cls (背景)
           - output_box_target: 匹配成功的设为目标框, 未匹配的设为0
           - output_reg_weights: 匹配成功的设对应权重, 未匹配的设为0

        关键设计特点:
        -----------
        - 稀疏性: 只有匹配成功的预测才计算回归损失
        - 端到端匹配: 同时考虑分类和回归代价, 避免传统NMS后处理
        - 灵活的权重机制: 支持类别相关的回归权重
        - 数值稳定性: 处理NaN和无穷值
        - 朝向角参数化: 使用sin/cos表示朝向, 避免角度周期性

        返回值:
        -------
        output_cls_target: torch.Tensor, 形状 [bs, num_pred]
            每个预测的类别目标, 未匹配的设为num_cls (背景类索引)
        output_box_target: torch.Tensor, 形状 [bs, num_pred, 11]
            每个预测的3D框目标 (编码后格式)
        output_reg_weights: torch.Tensor, 形状 [bs, num_pred, 11]
            每个预测的回归损失权重
        """
        bs, num_pred, num_cls = cls_pred.shape

        # ! 步骤1: 计算分类代价 (使用Focal Loss变体)
        cls_cost = self._cls_cost(cls_pred, cls_target)

        # ! 步骤2: 编码回归目标 (位置不变, 尺寸取log, 朝向角分解为sin/cos)
        box_target = self.encode_reg_target(box_target, box_pred.device)

        # ! 步骤3: 计算每个实例的回归权重
        # - 首先标记非NaN的维度 (有效维度）
        # - 如果设置了类别相关权重, 则根据不同类别调整权重
        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)

        # ! 步骤4: 计算回归代价 (加权L1距离)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        # ! 步骤5: 使用匈牙利算法进行最优匹配
        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                # 合并分类和回归代价
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                # 处理无效值：将-inf和NaN替换为1e8
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                # 求解最优二分图匹配
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])

        # ! 步骤6: 构建输出目标
        # 初始化：所有预测设为背景类
        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        # 初始化：所有预测框设为0
        output_box_target = box_pred.new_zeros(box_pred.shape)
        # 初始化：所有回归权重设为0
        output_reg_weights = box_pred.new_zeros(box_pred.shape)

        # 将匹配成功的预测设置为目标值
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx]

        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) for x in cls_target]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(dn_anchor, box_target, torch.ones_like(box_target))
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(mask, temp_meta, meta[:, :num_temp_dn_groups])
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(bs, num_dn_groups, num_temp_dn, -1)[
            :, temp_group_mask
        ]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(bs, num_dn_groups, num_temp_dn)[
                :, temp_group_mask
            ]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )


@BBOX_SAMPLERS.register_module()
class CustomizedSparseBox3DTarget(SparseBox3DTarget):
    def get_dn_reg_weights(
        self,
        dn_reg_target,
        dn_cls_target,
        dn_id_target,
        gt_instance_id,
        vel_valid_flags,
    ):
        """
        get_dn_reg_weights 的核心作用是: 为每个 DN box 生成回归损失权重,
            并用于处理速度维度的有效性问题

        核心问题: 在实际场景中, GT boxes 的速度信息不总是有效的, 例如:
        - 静止目标的速度应该为 0, 但标注可能不准确;
        - 某些目标无法获取速度信息, 需要根据 GT 的速度有效性标志来决定是否
          计算速度维度的损失

        输入:
            dn_reg_target:
                - 形状: [bs, num_dn, 11]
                - 说明: DN boxes 对应的回归目标 (x, y, z, l, w, h, sin, cos, vx, vy, vz)
            dn_cls_target
                - 形状: [bs, num_dn]
                - 说明: DN boxes 的类别目标
            dn_id_target,
                - 形状: [bs, num_dn]
                - 说明: DN boxes 匹配的 GT instance ID
            gt_instance_id,
                - 形状: List[torch.Tensor]
                - 说明: 每个 batch 的 GT instance IDs
            vel_valid_flags,
                - 形状: List[torch.Tensor]
                - 说明: 每个 batch 的 GT 速度有效性标志

        输出:
            dn_reg_weights:
                - 形状: [bs, num_dn, 11]
                - 说明: 每个 DN boxes 各维度的回归权重
        """
        # ! 阶段 1: 初始化权重为全 1
        dn_reg_weights = torch.ones_like(dn_reg_target)

        # ! 阶段 2: 类别级别的权重调整
        # 应用场景: 不同类别的目标可能有不同的定位精度要求, 例如:
        # 行人的尺寸变化大, 可能需要降低尺寸维度的权重

        # ! 阶段 3: 速度有效性处理
        for batch_index in range((len(gt_instance_id))):
            # 3.2 构建 GT 的权重矩阵
            weight = torch.ones(
                [gt_instance_id[batch_index].shape[0], dn_reg_target.shape[-1]],
                device=dn_reg_target.device,
            )
            weight[~vel_valid_flags[batch_index], -3:] = 0

            # 3.3 获取 ID 用于匹配
            # 匹配目标: 找到每个 DN box 对应的 GT, 从而继承该 GT 的速度有效性权重
            dn_ids = dn_id_target[batch_index]  # [M], M 是该 batch 的 DN box 数量
            gt_ids = gt_instance_id[batch_index]  # [N], N 是该 batch 的 GT 数量

            # 3.4 过滤无效 DN
            valid_dn_mask = dn_ids != -1  # 只保留匹配到真实 GT 的 DN box

            # 3.5 构建 ID 匹配矩阵
            # 矩阵含义:
            # comparison[i,j] = True 表示第 i 个 GT 与第 j 个 DN box 匹配 (instance ID 相同)
            comparison = gt_ids.unsqueeze(1) == dn_ids.unsqueeze(0)  # [N, M]
            comparison = comparison & valid_dn_mask.unsqueeze(0)  # [N, M]

            # 3.6 使用 cumsum 找到第一个匹配
            # 目的: 一个 DN box 可能匹配多个 GT (因为一个 GT 可能生成多个 DN),
            # 但每个 DN 只能继承一个 GT 的权重
            cumsum_comp = comparison.cumsum(dim=0)
            first_match = (cumsum_comp == 1) & comparison

            # 3.7 提取匹配索引并赋值权重
            # 赋值逻辑:
            # 根据 GT 索引获取对应的 weight, 并将 weight 赋给匹配的 DN box
            match_indices = first_match.nonzero(as_tuple=False)  # [K, 2], K 是匹配数量
            if match_indices.shape[0] > 0:
                gt_indices = match_indices[:, 0]  # 匹配的 GT 索引
                dn_indices = match_indices[:, 1]  # 匹配的 DN 索引
                dn_reg_weights[batch_index, dn_indices, :] *= weight[gt_indices, :]

        return dn_reg_weights
