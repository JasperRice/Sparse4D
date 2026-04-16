# Copyright (c) Horizon Robotics. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK,
                                      NORM_LAYERS, PLUGIN_LAYERS,
                                      POSITIONAL_ENCODING)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_CODERS, BBOX_SAMPLERS
from mmdet.models import HEADS, LOSSES

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        """
        forward 是 Sparse4DHead 的核心前向传播方法, 实现了 Sparse4D 检测器的完整推理流程:

        1. 输入处理: 多尺度特征图, 元数据
        2. 实例初始化: 从 InstanceBank 获取可学习 anchor 和时序缓存
        3. 去噪训练准备: 生成 DN anchors (仅训练模式)
        4. Decoder 层迭代: 多层特征聚合 (Temporal Cross-attention, DFA) 和 anchor 精细化
        5. 输出收集与缓存: 整理输出并缓存当前帧信息

        输入:
            feature_maps (List[Tensor]):
                - 形状: [bs, num_cams, C, H, W]
                - 说明: 多尺度相机特征图
            metas (dict):
                - 说明: 包含图像元信息 (投影矩阵, 时间戳等) 和标注信息的字典

        输出:
            dict: 包含所有预测结果的字典
                - classification
                    - 形状: List[[bs, num_anchor, num_cls]]
                    - 说明: 各层分类预测
                - prediction
                    - 形状: List[[bs, num_anchor, 11]]
                    - 说明: 各层 anchor 预测
                - quality
                    - 形状: List[[bs, num_anchor, 2]]
                    - 说明: 各层质量估计 (可选)
                - instance_feature
                    - 形状: [bs, num_anchor, embed_dims]
                    - 说明: 最终实例特征
                - dn_*
                    - 说明: 各种 DN 相关输出 (仅训练时)
        """
        # ! 阶段 1: 输入数据预处理
        # 1. 统一特征图格式为列表
        # 2. 获取 batch size
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ! 阶段 2: 获取实例信息
        # 检查缓存的 DN metas 是否有效
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        # 从 InstanceBank 获取实例信息
        (
            instance_feature,  # 当前帧的可学习实例特征, 初始化后每帧共享, [bs, num_anchor, embed_dims]
            anchor,  # 当前帧的 anchor 参数, [bs, num_anchor, 11]
            temp_instance_feature,  # 时序缓存的实例特征, [bs, num_anchor, embed_dims] 或 None
            temp_anchor,  # 时序缓存的 anchor (已运动补偿到当前帧), [bs, num_anchor, 11] 或 None
            time_interval,  # 当前帧与历史帧的时间间隔
        ) = self.instance_bank.get(batch_size, metas, dn_metas=self.sampler.dn_metas)

        # ! 阶段 3: 去噪训练准备
        # 3.1 初始化变量
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        # 3.2 生成 DN anchors
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            # 获取 GT instance IDs
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            # 生成 DN anchors, 详见 projects\mmdet3d_plugin\models\detection3d\target.py
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            # 计算 DN 的回归权重, 详见 projects\mmdet3d_plugin\models\detection3d\sampler.py
            if hasattr(self.sampler, "get_dn_reg_weights"):
                dn_reg_weights = self.sampler.get_dn_reg_weights(
                    dn_reg_target,
                    dn_cls_target,
                    dn_id_target,
                    gt_instance_id,
                    metas,
                )
            else:
                dn_reg_weights = torch.ones()
            # 3.3 维度对齐, 确保 DN anchor 和可学习 anchor 的维度一致 (某些情况下 DN 可能缺少速度维度)
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
            # ! Attenion Mask 结构
            #               free_anchors    DN_anchors
            # free_anchors      1               0
            #   DN_anchors      0          dn_attn_mask
            # 0: 不可见
            # 1: 可见
            # - free_anchors 只看自己
            # - DN 按 dn_attn_mask 规则

        # ! 阶段 4: Anchor 编码
        # ! Anchor Encoder 的作用
        # - 将 anchor 的几何信息注入到特征空间
        # - 类似于 Transformer 中的 positional encoding
        # - 使模型能够感知每个实例的空间位置和形状
        # 编码 anchor 特征
        anchor_embed = self.anchor_encoder(anchor)
        instance_feature += anchor_embed
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # ! 阶段 5: Decoder Layer 循环
        # 这是 forward 方法最核心的部分, 通过多层迭代逐步优化 anchor
        # 5.1 初始化输出收集器
        prediction = []
        classification = []
        quality = []
        # ! 5.2 Operation Order
        # 默认的 operation_order (第一个 decoder 省略 gnn 和 norm):
        # 第 1 个 decoder, 可以得到更好的初始 instance_feature:
        # ["deformable", "norm", "ffn", "norm", "refine"]
        # 第 2 ~ num_decoder 个 decoder:
        # ["temp_gnn", "gnn", "norm", "deformable", "norm", "ffn", "norm", "refine"]
        for i, op in enumerate(self.operation_order):
            # ! 5.3 各操作详解
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                # ! 时序 Cross-attention
                # 输出融合时序信息的 instance features
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # ? 是否只处理 free instances
                    temp_instance_feature,  # 历史帧特征
                    temp_instance_feature,  # key = value
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "gnn":
                # ! Self-attention
                # 实例之间的 self-attention, 学习实例间的空间关系
                # Attention mask 的作用:
                # 1. 自由 anchor 可以互相 attend
                # 2. DN anchor 只能在同一组内互相 attend
                # 3. 自由 anchor 和 DN anchor 之间隔离
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm":
                # ! LayerNorm 归一化层
                instance_feature = self.layers[i](instance_feature)
            elif op == "ffn":
                # ! Feed-forward network
                # 通常是 MLP + 残差连接
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                # ! Deformable 可变形特征聚合
                # 这是 Sparse4D 的核心操作, 详见 DeformableFeatureAggregation
                # 简要流程:
                # 1. 生成关键点 (基于 anchor 的 3D box)
                # 2. 计算采样权重 (基于 instance_feature 和 anchor_embed)
                # 3. 将 3D 关键点投影到 2D 图像
                # 4. 多视角多尺度特征采样
                # 5. 加权融合得到最终特征
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                # ! Anchor 精细化
                # 详见 SparseBox3DRefinementModule 模块
                # 每组 Layers 输出一组结果
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                # ! 5.4 时序更新
                if len(prediction) == self.num_single_frame_decoder:
                    # ! Instance Bank 更新
                    # 1. 计算每个 anchor 的置信度
                    # 2. 使用 topk 选择置信度最高的 N 个实例
                    # 3. 将 cached_feature 与选择的实例拼接
                    # 4. 根据 mask 决定是否使用缓存
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    # ! DN 更新
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                # ! 重新生成 anchor_embed
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # ! 阶段 6: 输出处理
        # ! 6.1 分离 DN 输出
        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [x[:, num_free_instance:] for x in classification]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # ! 6.2 构建输出字典
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # ! 6.3 缓存当前帧信息
        # cache current instances for temporal modeling
        self.instance_bank.cache(instance_feature, anchor, cls, metas, feature_maps)
        if not self.training:
            # ! 生成 instance ID 用于跟踪
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ! 阶段 1: 预测损失 (Prediction Losses)
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        # 遍历每个 decoder 层的输出
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            # ! SparseBox3DTarget.sample
            # 给每个 anchor 分配 GT
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            # ! 1.1 分类损失 (Focal Loss)
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            # ! 1.2 回归损失 (SparseBox3DLoss)
            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ! 阶段 2: 去噪损失 (Denoising Losses)
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")
            # ! 2.1 DN 分类损失
            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            # ! 2.2 DN 回归损失
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """prepare_for_dn_loss 是 Sparse4DHead 中用于准备去噪(Denoising)训练损失计算的辅助函数。
        它的主要作用是从模型输出中提取和预处理 DN 相关的目标数据, 为后续的损失计算做准备。

        Args:
            model_outs (dict): 模型前向传播的输出字典, 包含分类预测、回归预测、DN相关元数据等
            prefix (str, optional): 前缀字符串, 用于区分普通DN和时序DN数据. Defaults to "".

        Returns:
            dict: 返回一个包含6个元素的元组:
            - dn_valid_mask: DN有效掩码, 标识哪些DN样本是有效的 (非padding)
            - dn_cls_target: DN分类目标, 经过过滤后的类别标签
            - dn_reg_target: DN回归目标, 经过过滤和维度裁剪后的边界框目标
            - dn_pos_mask: 正样本掩码, 标识哪些是正样本 (类别 >= 0)
            - reg_weights: 回归权重, 根据配置为不同维度分配不同权重
            - num_dn_pos: 正样本数量 (经过多GPU同步后的值)
        """
        # ! 步骤1: 获取DN有效掩码并展平
        # dn_valid_mask: [bs, num_dn] -> [bs*num_dn]
        # 作用: 标识哪些DN样本是真实有效的（非padding填充的）
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        # ! 步骤2: 提取有效DN样本的分类目标
        # 先获取所有DN的分类目标，然后只保留有效样本
        # 结果: [num_valid_dn]
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]
        # ! 步骤3: 提取有效DN样本的回归目标并裁剪维度
        # 1) 获取所有DN的回归目标
        # 2) 只保留有效样本
        # 3) 只保留前len(self.reg_weights)个维度（某些维度如速度可能不需要回归）
        # 结果: [num_valid_dn, reg_dim]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        # ! 步骤4: 生成正样本掩码
        # dn_cls_target >= 0 表示正样本（背景类通常为-1，忽略类为-2，padding为-3）
        # 结果: [num_valid_dn]
        dn_pos_mask = dn_cls_target >= 0
        # ! 步骤5: 只保留正样本的回归目标
        # 结果: [num_pos_dn, reg_dim]
        dn_reg_target = dn_reg_target[dn_pos_mask]
        # ! 步骤6: 生成回归权重
        # 为每个正样本分配相同的权重向量（来自config的reg_weights配置）
        # 结果: [num_pos_dn, reg_dim]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        # ! 步骤7: 计算正样本数量（多GPU同步）
        # reduce_mean: 对所有GPU的正样本数量取平均，确保损失计算的稳定性
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,  # 至少为1，避免除以0
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
