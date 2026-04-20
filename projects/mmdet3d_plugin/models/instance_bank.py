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
        """
        缓存当前帧的实例信息, 用于下一帧的时序建模.

        该方法在 Sparse4DHead 的 forward 过程末尾被调用 (第494行), 用于将当前帧的高置信度
        实例缓存起来, 供下一帧使用. 这是 Sparse4D 实现跨帧时序信息传递的核心机制.

        主要功能包括:
        1. 实例缓存: 选择当前帧中置信度最高的实例进行缓存
        2. 置信度衰减: 对历史置信度进行衰减, 避免旧信息过度影响
        3. 元数据保存: 保存当前帧的元数据, 用于计算时间间隔和坐标变换
        4. 临时置信度管理: 维护 temp_confidence 用于实例ID跟踪

        Args:
            instance_feature (torch.Tensor): 当前帧的实例特征, 形状为 [B, N, C]
                - B: batch size
                - N: 实例总数
                - C: 特征维度 (embed_dims)
            anchor (torch.Tensor): 当前帧的锚点参数, 形状为 [B, N, D]
                - D: 锚点维度 (通常为11, 包含位置、尺寸、旋转、速度等)
            confidence (torch.Tensor): 实例的分类置信度, 形状为 [B, N, num_classes]
                - 来自 decoder 层的分类预测结果
            metas (dict, optional): 当前帧的元数据字典, 包含以下关键字段:
                - timestamp: 当前帧的时间戳
                - img_metas: 图像元信息列表, 包含 T_global 和 T_global_inv 等变换矩阵
                - 其他相机内参、外参等信息
            feature_maps (list, optional): 多尺度特征图 (当前未使用, 保留用于扩展)

        Returns:
            None: 该方法直接修改实例库的内部状态, 不返回任何值

        Note:
            - 该方法是 Sparse4D 时序建模的"写入"端, 与 get() 方法的"读取"端配合使用
            - 通过置信度驱动的实例选择, 确保缓存的是最可靠的实例信息
            - 置信度衰减机制防止旧实例的置信度过高, 提高跟踪的鲁棒性
            - 梯度分离避免了跨帧梯度传播, 使训练更加稳定
            - 缓存的实例数量 (num_temp_instances) 是一个重要的超参数, 需要权衡效果和速度
        """
        # ! 1. 检查是否需要缓存:
        # - 如果 self.num_temp_instances <= 0, 表示不启用时序建模, 直接返回
        # - 这是通过配置文件控制的, 例如 num_temp_instances=600 表示缓存600个实例
        if self.num_temp_instances <= 0:
            return

        # ! 2. 梯度分离:
        # - 对 instance_feature、anchor、confidence 进行 .detach() 操作
        # - 原因: 缓存的信息仅用于推理, 不应参与梯度反向传播
        # - 这可以避免跨帧的梯度传播, 减少显存占用
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        # ! 3. 保存元数据:
        # - 将当前帧的 metas 保存到 self.metas
        # - 用于下一帧调用 get() 时计算时间间隔和进行坐标变换
        self.metas = metas

        # ! 4. 置信度处理:
        # - 取每个实例的最高类别置信度: confidence.max(dim=-1).values
        # - 通过 sigmoid 将 logits 转换为概率值 (0-1范围)
        # - sigmoid 后的值可以理解为"该实例包含目标的概率"
        confidence = confidence.max(dim=-1).values.sigmoid()

        # ! 5. 置信度衰减与更新:
        # - 如果存在历史置信度 (self.confidence is not None):
        #     - 对历史置信度应用衰减: self.confidence * self.confidence_decay
        #     - self.confidence_decay 默认为 0.6, 表示每帧衰减40%
        #     - 取衰减后的历史置信度与当前置信度的最大值
        #     - 这确保了缓存实例的置信度不会因时间推移而过高
        # - 将更新后的置信度保存到 self.temp_confidence
        # - temp_confidence 用于实例ID的跟踪和更新
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        # ! 6. 实例选择与缓存:
        # - 使用 topk 函数选择置信度最高的 self.num_temp_instances 个实例
        # - 例如, 如果 num_temp_instances=600, 则从900个实例中选择600个
        # - 将选中的实例特征保存到 self.cached_feature
        # - 将选中的实例锚点保存到 self.cached_anchor
        # - 将选中的实例置信度保存到 self.confidence
        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        """
        为当前帧的检测实例分配唯一的跟踪ID, 用于多目标跟踪.

        该方法在 Sparse4DHead 的 forward 过程末尾被调用 (第497-500行, 仅在推理模式下),
        用于为检测到的目标分配唯一的实例ID, 实现跨帧的目标关联和跟踪.

        主要功能包括:
        1. ID继承: 将历史缓存的实例ID继承到当前帧的对应位置
        2. 新ID分配: 为新检测到的实例分配新的唯一ID
        3. 阈值过滤: 可选地过滤低置信度的实例, 不为其分配ID
        4. ID更新: 更新缓存的实例ID, 用于下一帧的ID继承

        Args:
            confidence (torch.Tensor): 实例的分类置信度, 形状为 [B, N, num_classes]
                - B: batch size
                - N: 实例总数
                - num_classes: 类别数量
                - 用于确定哪些实例需要分配ID以及ID的继承顺序
            anchor (torch.Tensor, optional): 当前帧的锚点参数, 形状为 [B, N, D]
                - 当前未使用, 保留用于扩展或兼容性
            threshold (float, optional): 置信度阈值
                - 如果提供, 只有置信度 >= threshold 的实例才会被分配ID
                - 用于过滤低置信度的检测, 避免为背景或噪声分配ID
                - 默认值 None 表示不为置信度设置阈值

        Returns:
            torch.Tensor: 实例ID张量, 形状为 [B, N], 数据类型为 torch.int64
                - 每个元素表示对应实例的唯一跟踪ID
                - ID >= 0: 有效的实例ID, 用于跨帧关联同一目标
                - ID = -1: 表示该实例未被分配有效ID (可能是背景、低置信度或新检测但未达阈值)
                - ID在整个视频序列中保持唯一性, 通过 self.prev_id 计数器实现

        Note:
            - 该方法是 Sparse4D 实现多目标跟踪的核心组件
            - 通过全局计数器 self.prev_id 确保ID在整个视频序列中的唯一性
            - ID继承机制使得同一目标在跨帧时保持相同的ID
            - 阈值过滤机制可以避免为背景或低置信度检测分配ID, 提高跟踪质量
            - update_instance_id 确保只有高置信度的实例ID被缓存, 提高跟踪的可靠性
            - 该方法仅在推理模式下被调用 (见 sparse4d_head.py 第495-500行)
        """
        # ! 1. 置信度处理:
        # - 取每个实例的最高类别置信度: confidence.max(dim=-1).values
        # - 通过 sigmoid 将 logits 转换为概率值 (0-1范围)
        # - sigmoid 后的值表示"该实例包含目标的概率"
        confidence = confidence.max(dim=-1).values.sigmoid()

        # ! 2. 初始化ID张量:
        # - 创建一个与 confidence 形状相同的张量 instance_id, 初始值全为 -1
        # - -1 表示"未分配ID", 是实例ID的默认无效值
        # - 数据类型为 torch.int64 (long), 适合用作索引和ID
        instance_id = confidence.new_full(confidence.shape, -1).long()

        # ! 3. ID继承:
        # - 检查是否存在历史缓存的实例ID (self.instance_id is not None)
        # - 检查 batch size 是否匹配 (self.instance_id.shape[0] == instance_id.shape[0])
        # - 如果条件满足, 将历史ID复制到当前帧的对应位置:
        #     instance_id[:, : self.instance_id.shape[1]] = self.instance_id
        # - 这意味着前 self.instance_id.shape[1] 个实例继承了历史ID
        # - 这些实例通常是时序建模中缓存的高置信度实例
        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        # ! 4. 确定需要新ID的实例:
        # - 创建掩码 mask = instance_id < 0, 标识所有尚未分配ID的实例
        # - 如果提供了 threshold 参数, 进一步过滤:
        #     mask = mask & (confidence >= threshold)
        # - 这确保只有置信度足够高的新实例才会被分配ID
        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)

        # ! 5. 分配新ID:
        # - 计算需要新ID的实例数量: num_new_instance = mask.sum()
        # - 生成新的ID序列: new_ids = torch.arange(num_new_instance) + self.prev_id
        #     - self.prev_id 是一个全局计数器, 确保ID在整个序列中唯一
        #     - 例如, 如果 prev_id=100, 新检测3个实例, 则ID为 [100, 101, 102]
        # - 将新ID分配给需要ID的实例: instance_id[torch.where(mask)] = new_ids
        # - 更新全局计数器: self.prev_id += num_new_instance
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance

        # ! 6. 更新缓存的实例ID:
        # - 如果启用了时序建模 (self.num_temp_instances > 0):
        #     - 调用 update_instance_id 方法更新 self.instance_id
        #     - 根据置信度选择高置信度实例的ID进行缓存
        #     - 这些缓存的ID将在下一帧被继承
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        """
        更新缓存的实例ID, 选择高置信度实例的ID进行保存, 用于下一帧的ID继承.

        该方法是 get_instance_id 的辅助方法, 在 get_instance_id 中被调用 (第447-448行).
        其主要作用是根据置信度从当前帧的所有实例中选择最可靠的 num_temp_instances 个实例,
        将它们的ID保存到 self.instance_id 中, 以便在下一帧进行ID继承.

        主要功能包括:
        1. 置信度选择: 根据置信度确定哪些实例的ID应该被缓存
        2. ID筛选: 使用 topk 选择高置信度实例的ID
        3. ID缓存: 将筛选后的ID保存到 self.instance_id, 用于下一帧继承

        Args:
            instance_id (torch.Tensor, optional): 当前帧的实例ID张量, 形状为 [B, N]
                - B: batch size
                - N: 实例总数
                - 包含当前帧所有实例的跟踪ID (>=0 表示有效ID, -1 表示无效ID)
                - 如果为 None, 则从 self.temp_confidence 中获取置信度进行选择
            confidence (torch.Tensor, optional): 当前帧的实例置信度, 形状为 [B, N] 或 [B, N, num_classes]
                - 用于确定哪些实例的ID应该被缓存
                - 如果为 None, 则使用 self.temp_confidence 作为置信度

        Returns:
            None: 该方法直接修改 self.instance_id, 不返回任何值

        Note:
            - 该方法是 Sparse4D 多目标跟踪的关键组件, 确保ID继承的可靠性
            - 通过置信度驱动的ID选择, 确保只有高置信度实例的ID被缓存
            - 这避免了低置信度或噪声实例的ID被继承到下一帧, 提高跟踪质量
            - self.instance_id 的形状是 [B, num_anchor], 与实例特征和锚点的数量一致
            - 在 get_instance_id 中, 只有前 self.instance_id.shape[1] 个实例会继承历史ID
            - 该方法仅在启用了时序建模 (self.num_temp_instances > 0) 时才会被调用
        """
        # ! 1. 确定用于排序的置信度:
        # - 如果 self.temp_confidence 存在 (在 cache 方法中被设置):
        #     - 使用 self.temp_confidence 作为排序依据
        #     - self.temp_confidence 是在 cache 方法中计算并保存的, 已经过 sigmoid 处理
        # - 如果 self.temp_confidence 不存在:
        #     - 使用传入的 confidence 参数
        #     - 如果 confidence 是3D张量 [B, N, num_classes], 取最高类别置信度
        #     - 如果 confidence 是2D张量 [B, N], 直接使用
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence

        # ! 2. 选择高置信度实例的ID:
        # - 使用 topk 函数根据置信度选择最高的 self.num_temp_instances 个实例
        # - 例如, 如果 num_temp_instances=600, 则从900个实例中选择600个
        # - topk 返回 (confidence, [selected_instance_id, ...])
        # - 我们只关心 selected_instance_id, 即高置信度实例的ID
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][0]

        # ! 3. 维度调整:
        # - topk 返回的 instance_id 形状为 [B, num_temp_instances, 1]
        # - 使用 squeeze(dim=-1) 将其压缩为 [B, num_temp_instances]
        instance_id = instance_id.squeeze(dim=-1)

        # ! 4. 填充到完整尺寸:
        # - 使用 F.pad 将 instance_id 从 [B, num_temp_instances] 填充到 [B, num_anchor]
        # - 填充的值为 -1, 表示这些位置没有有效的ID
        # - 最终 self.instance_id 的形状为 [B, num_anchor]
        # - 前 num_temp_instances 个位置是高置信度实例的ID
        # - 剩余位置是 -1 (无效ID)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
