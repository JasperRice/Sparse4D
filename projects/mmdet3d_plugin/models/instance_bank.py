import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.utils import build_from_cfg
from torch import nn

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(
        -1
    )
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        """
        时序建模: 利用历史缓存数据进行跨帧信息传递.
        坐标对齐: 通过 anchor_handler 将历史锚点变换到当前帧坐标系, 解决目标运动与视角变化.
        鲁棒性处理: 通过掩码过滤无效历史帧, 避免长期累积误差.
        """
        # 从实例库中取出基础特征 instance_feature 和锚点 anchor, 并扩展为当前 batch_size 的维度, 以供当前批次使用.
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        # 检查是否存在缓存的上一帧锚点 cached_anchor, 且其批次大小与当前请求一致.
        # 如果满足条件, 说明是连续帧处理, 可以进行时序建模.
        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            # 1. 计算当前帧与缓存帧的时间差 time_interval.
            # 2. 生成掩码mask, 标记哪些帧的时间间隔在允许范围内 (max_time_interval), 用于筛选有效历史信息.
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                # 如果存在 anchor_handler (负责坐标变换), 则计算从缓存帧坐标系到当前帧坐标系的变换矩阵 T_temp2cur.
                # 通过 anchor_projection 将缓存的锚点 cached_anchor 投影到当前帧坐标系, 实现跨帧的几何对齐.
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                # 如果提供了去噪锚点 dn_anchor, 同样对其进行坐标变换, 以保持与当前帧的空间一致性.
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )

            # 对时间间隔进行后处理:
            # 如果 time_interval 不为0且在有效掩码内, 则保留原值.
            # 否则, 设为默认值 default_time_interval (如0.5), 避免无效值干扰.
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            # 如果没有缓存或批次不匹配, 则重置实例库状态, 并将所有时间间隔设为默认值.
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,  # 当前实例特征 [B, N, C]
            anchor,  # 当前锚点 [B, N, D]
            self.cached_feature,  # 缓存的特征 (可能被更新)
            self.cached_anchor,  # 缓存的锚点 (已坐标对齐)
            time_interval,  # 时间间隔 [B]
        )

    def update(self, instance_feature, anchor, confidence):
        """
        更新实例库中的缓存特征和锚点, 实现跨帧的时序信息传递和实例管理.

        该方法在 Sparse4DHead 的 forward 过程中被调用, 用于在多帧处理时更新实例库的状态.
        主要功能包括:
        1. 时序特征融合: 将当前帧的高置信度实例与历史缓存的实例进行融合
        2. 实例选择: 通过置信度排序选择最可靠的实例进行缓存
        3. DN (Denoising) 实例处理: 分离并保留去噪训练相关的实例
        4. 实例ID管理: 更新实例ID, 处理丢失的实例

        Args:
            instance_feature (torch.Tensor): 当前帧的实例特征, 形状为 [B, N, C]
                - B: batch size
                - N: 实例总数 (包括可学习实例和DN实例)
                - C: 特征维度
            anchor (torch.Tensor): 当前帧的锚点参数, 形状为 [B, N, D]
                - D: 锚点维度 (通常为11, 包含位置、尺寸、旋转、速度等)
            confidence (torch.Tensor): 实例的分类置信度, 形状为 [B, N, num_classes]
                - 每个实例对每个类别的置信度分数

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 更新后的实例特征和锚点
                - instance_feature (torch.Tensor): 更新后的实例特征, 形状为 [B, N, C]
                - anchor (torch.Tensor): 更新后的锚点参数, 形状为 [B, N, D]

        Note:
            - 该方法是 Sparse4D 检测器实现时序建模的关键组件
            - 通过置信度驱动的实例选择, 确保缓存的是最可靠的实例信息
            - 通过掩码机制, 避免使用时间间隔过长的历史信息, 提高跟踪的鲁棒性
            - DN实例的特殊处理保证了去噪训练的正确性
        """
        if self.cached_feature is None:
            return instance_feature, anchor

        # ! 步骤 1. DN实例分离:
        # - 如果实例总数超过预设的锚点数量 (self.num_anchor), 说明包含DN实例
        # - 将DN实例从主实例中分离出来, 保存到 dn_instance_feature 和 dn_anchor
        # - 主实例数量被限制为 self.num_anchor
        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        # ! 步骤 2. 置信度计算与实例选择:
        # - 计算每个实例的最高类别置信度: confidence.max(dim=-1).values
        # - 确定需要保留的实例数量 N = self.num_anchor - self.num_temp_instances
        #     (总锚点数减去临时实例数), 实际 self.num_temp_instances 为 600
        # - 使用 topk 函数选择置信度最高的 N 个实例的特征和锚点
        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )

        # ! 步骤 3. 时序特征融合:
        # - 将历史缓存的特征 self.cached_feature 与当前选择的高置信度特征拼接
        # - 将历史缓存的锚点 self.cached_anchor 与当前选择的高置信度锚点拼接
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)

        # ! 步骤 4. 条件更新:
        # - 根据掩码 self.mask 决定是否使用融合后的特征和锚点
        # - 如果掩码为 True (表示有效的时间间隔), 使用融合结果
        # - 否则, 保持当前帧的特征和锚点不变
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)

        # ! 步骤 5. 实例ID管理:
        # - 如果存在实例ID (self.instance_id), 根据掩码更新ID
        # - 对于无效的实例 (掩码为False), 将其ID设置为-1 (表示丢失)
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        # ! 步骤 6. DN实例恢复:
        # - 如果存在DN实例 (num_dn > 0), 将其重新附加到更新后的特征和锚点末尾
        # - 确保DN实例在后续处理中不被遗忘
        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        # ! 步骤 7. 返回结果:
        # - 返回更新后的实例特征和锚点, 用于后续的处理步骤
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][0]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
