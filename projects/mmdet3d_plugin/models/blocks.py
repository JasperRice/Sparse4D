# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      constant_init, xavier_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK,
                                      PLUGIN_LAYERS)
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils import build_from_cfg
from torch.cuda.amp.autocast_mode import autocast

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)  # ! 每组的维度
        self.embed_dims = embed_dims  # ! 总特征维度
        self.num_levels = num_levels  # ! 特征金字塔层级数
        self.num_groups = (
            num_groups  # ! 分组数, 类似 multi-head attention, 不同组学习不同的聚合模式
        )
        self.num_cams = num_cams  # ! 相机数量
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts  # ! 每个 anchor 的采样多少个关键点
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(temporal_fusion_module, PLUGIN_LAYERS)
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = Linear(embed_dims, num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """方法概述

        DeformableFeatureAggregation 是 Sparse4D 的核心特征聚合模块, 它的作用是:
        1. 生成 3D 关键点: 基于 anchor 的 3D 边界框生成采样点
        2. 投影到 2D: 将 3D 关键点投影到多个相机视角
        3. 多尺度特征采样: 从不同层级特征图采样特征
        4. 加权融合: 根据学习到的权重融合多视角, 多尺度的特征

        核心思想: 类似于 Deformable DETR 的可变形注意力, 但扩展到多相机 3D 场景

        Args:
            instance_feature (torch.Tensor): [bs, num_anchor, embed_dims], 实例特征向量
            anchor (torch.Tensor): [bs, num_anchor, 11], 3D anchor 参数
            anchor_embed (torch.Tensor): [bs, num_anchor, embed_dims], anchor 编码特征
            feature_maps (List[torch.Tensor]): [bs, num_anchor, C, H, W] x num_levels, 多尺度特征图
            metas (dict): 元数据 (投影矩阵等)

        Returns:
            torch.Tensor: [bs, num_anchor, embed_dims], 聚合后的特征
        """
        # ! 阶段 1: 生成关键点, 详见 SparseBox3DKeyPointsGenerator.forward
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        # ! 阶段 2: 计算采样权重
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            # ! 分支 A: 使用自定义 CUDA 算子
            # ! 阶段 A.1: 3D 到 2D 投影
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            # ! 阶段 A.2: 特征采样与融合
            # 4.1 特征重排
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            # 4.2 调用自定义 CUDA 算子
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            # ! 分支 B: 纯 PyTorch 实现
            # ! 阶段 B.1: 特征采样
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            # ! 阶段 B.2: 多视角融合
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
        # ! 阶段 3: 输出投影与残差连接
        # MLP + Dropout 得到输出
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        # ! 阶段 1: 特征融合, 结合实例特征和 anchor 位置编码
        feature = instance_feature + anchor_embed
        # ! 阶段 2: 相机编码
        # 让权重考虑相机内外参, 适应不同相机视角
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]
        # ! 阶段 3: 预测权重
        # 权重含义:
        # - 对于每个 anchor, 每个 group, 预测 num_cams x num_levels x num_pts 个权重
        # - softmax 确保权重和为 1
        # - 不同 group 可以关注不同的视角/尺度组合
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(
                dim=-2
            )  # ! 关键: 在 (num_cams x num_levels x num_pts) 维度做softmax
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        # ! 阶段 4: 训练时的 Dropout
        # 随机丢弃某些视角的权重, 增加鲁棒性
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # ! 1. 投影到 2D
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        # ! 2. 转换到 grid sample 格式 [-1, 1]
        points_2d = points_2d * 2 - 1
        # ! 3. Flatten 用于批量处理
        points_2d = points_2d.flatten(end_dim=1)
        # ! 4. 对每个层级的特征图进行双线性插值采样
        # grid_sample 说明:
        # - 输入: 特征图 [N, C, H, W], 采样坐标 [N, H_out, W_out, 2]
        # - 输出: 采样特征 [N, C, H_out, W_out]
        # - 坐标范围: [-1, 1] (-1 对应左/上边缘, 1 对应右/下边缘)
        # - 支持双线性插值
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d)
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]

        # ! 1. 将特征按 group 分割
        # [bs, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )

        # ! 2. 沿 cams 和 levels 维度求和
        # [bs, num_anchor, num_pts, num_groups, group_dims]
        features = features.sum(dim=2).sum(dim=2)

        # ! 3. Reshape 到最终形状
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
