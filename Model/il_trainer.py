import os
from typing import Dict
import torch.distributed as dist

import torch
import torch.nn.functional as F
from gym import Space

from Model.seq2seq_policy import Seq2SeqPolicy
from Model.cma_policy import CMAPolicy
from utils.logger import logger
from src.common.param import args
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN


class VLNCETrainer:
    #
    def __init__(
        self,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        ckpt_path=None,
        policy=None,  # 新增参数，允许传入外部策略实例
    ):
        self.start_epoch = 0
        self.step_id = 0

        if not args.DistributedDataParallel:
            self.device = (
                torch.device("cuda", args.trainer_gpu_device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )


        # 策略初始化
        if policy is not None:
            # 使用传入的自定义策略
            self.policy = policy
            self.policy.to(self.device)  # 确保在正确的设备上
        else:    
            model_config = CN.clone()
            if args.policy_type == 'seq2seq':
                self.policy = Seq2SeqPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'cma':
                self.policy = CMAPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'hcm':
                self.policy = HCMPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'unet':
                self.policy = UNetPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'vlnbert':
                self.policy = VLNBertPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            else:
                raise NotImplementedError

            self.policy.to(self.device)
        
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        # self.optimizer = torch.optim.Adam(
        #     self.policy.parameters(), lr=args.lr
        # )

        if load_from_ckpt:
            assert os.path.isfile(ckpt_path), 'ckpt_path error'
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            self.optimizer.load_state_dict(ckpt_dict["optimizer"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        if args.DistributedDataParallel and not getattr(args, 'deepspeed', False):
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    #
    def save_checkpoint(self, file_name, dagger_it, epoch) -> None:
        """
        Save checkpoint with specified name.
        :param file_name: file name for checkpoint
        :param epoch: epoch
        :return: None
        """
        checkpoint = {
            "state_dict": self.policy.module.state_dict() if args.DistributedDataParallel else self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "config": str(args),
            'dagger_it': dagger_it,
            'epoch': epoch,
        }

        from pathlib import Path
        checkpoint_folder = Path(args.project_prefix) / 'DATA/output/{}/train/checkpoint/{}'.format(args.name, args.make_dir_time)
        if not os.path.exists(str(checkpoint_folder)):
            os.makedirs(str(checkpoint_folder), exist_ok=True)

        torch.save(
            checkpoint, str(checkpoint_folder / file_name)
        )

    #
    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)
    
    def _update_agent(
        self,
        obs_list,
        prev_actions_list,
        oracle_actions_list,
        weights_list,
    ):
        device = self.device
        model_dtype = next(self.policy.parameters()).dtype
        N = len(obs_list)
        if N == 0:
            return 0.0, 0.0, 0.0

        # 各轨迹的真实长度（假设 rgb 是传感器之一，也可用任意键）
        lengths = [obs['pixel_values'].shape[0] for obs in obs_list]
        T_max = max(lengths) if lengths else 0
        if T_max == 0:
            return 0.0, 0.0, 0.0

        total_loss = 0.0
        total_action_loss = 0.0
        total_aux_loss = 0.0

        # LLaVAPolicy 无需 RNN 状态
        rnn_states = None

        # 清空辅助损失（全局单例）
        AuxLosses.clear()

        # use_deepspeed = hasattr(self, 'deepspeed_engine') and self.deepspeed_engine is not None
        # 判断是否使用 DDP（有 no_sync 属性）
        use_ddp = hasattr(self.policy, 'no_sync') and not hasattr(self, 'deepspeed_engine')

        for t in range(T_max):
            # 收集当前时间步有效的轨迹索引（长度 > t）
            valid_indices = [i for i, l in enumerate(lengths) if l > t]
            if not valid_indices:
                continue

            # ---- 构建当前时间步的观测字典 ----
            obs_t = {}
            # 取第一个有效轨迹的传感器键
            sample_obs = obs_list[valid_indices[0]]
            for key in sample_obs.keys():
                # 收集所有有效轨迹的第 t 个时间步数据
                tensors = [obs_list[i][key][t] for i in valid_indices]
                # 根据数据类型移动至 GPU 并转换 dtype
                if torch.is_floating_point(tensors[0]):
                    tensors = [v.to(device=device, dtype=model_dtype) for v in tensors]
                else:
                    tensors = [v.to(device=device) for v in tensors]
                obs_t[key] = torch.stack(tensors, dim=0)   # (valid_N, ...)

            # ---- 处理动作和权重 ----
            prev_actions_t = torch.stack(
                [prev_actions_list[i][t].to(device) for i in valid_indices], dim=0
            ).unsqueeze(1)  # (valid_N, 1)

            oracle_actions_t = torch.stack(
                [oracle_actions_list[i][t].to(device) for i in valid_indices], dim=0
            )  # (valid_N,)

            weights_t = torch.stack(
                [weights_list[i][t].to(device) for i in valid_indices], dim=0
            )  # (valid_N,)

            # ---- 前向传播，得到分布 ----
            # masks 参数传 None（LLaVAPolicy 不使用）
            # distribution = self.policy.build_distribution(
            # DDP
            distribution = self.policy.module.build_distribution(
                obs_t, rnn_states, prev_actions_t, None
            )
            logits = distribution.logits  # (valid_N, num_actions)

            # ---- 计算动作损失（加权平均） ----
            action_loss_t = F.cross_entropy(logits, oracle_actions_t, reduction='none')
            loss_t = (weights_t * action_loss_t).sum() / (weights_t.sum() + 1e-8)
            loss_t = loss_t / T_max   # 平均到每个时间步，保持梯度尺度
            # print(f"Step:{t} Loss: {loss_t}")
            # 替换为：
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            if t % 10 == 0:  # 每10步打印一次，避免刷屏
                print(f"[Rank {rank}] Step:{t}/{T_max-1} Loss: {loss_t.item():.6f}")
            

            # ---- 辅助损失（如果有） ----
            aux_mask = (weights_t > 0).view(-1)
            aux_loss_t = AuxLosses.reduce(aux_mask)   # 返回并清空
            if aux_loss_t != 0.0:
                aux_loss_t = aux_loss_t / T_max
                loss_t = loss_t + aux_loss_t

            # ---- 反向传播（立即释放计算图） ----
            # loss_t.backward()
            # ---- 反向传播：使用 no_sync 抑制通信 ----
            if use_ddp:
                # 只在最后一个时间步同步，之前抑制通信
                if t < T_max - 1:
                    with self.policy.no_sync():
                        loss_t.backward()
                else:
                    loss_t.backward()  # 最后一次同步所有梯度
            else:
                # DeepSpeed 或单卡，直接 backward
                loss_t.backward()
            

            # 累加用于日志的损失值（还原为未平均的数值）
            total_loss += loss_t.item() * T_max
            total_action_loss += action_loss_t.mean().item() * T_max
            total_aux_loss += (aux_loss_t.item() if isinstance(aux_loss_t, torch.Tensor) else aux_loss_t) * T_max

        # ---- 梯度裁剪（可选） ----
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        # ---- 更新参数 ----
        # ---- 更新参数 ----
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 返回平均损失
        avg_loss = total_loss / T_max
        avg_action_loss = total_action_loss / T_max
        avg_aux_loss = total_aux_loss / T_max
        print(f"Average Loss: {avg_loss}")
        return avg_loss, avg_action_loss, avg_aux_loss





    #
    # def _update_agent(
    #     self,
    #     observations,
    #     prev_actions,
    #     not_done_masks,
    #     corrected_actions,
    #     weights,
    #     step_grad: bool = True,
    #     loss_accumulation_scalar: int = 1,
    #     backward: bool = True, # 新增参数，默认 True 以保持向后兼容
    # ):
    #     # 获取 batch 大小和时间步数
    #     T, N = corrected_actions.size() # T = 时间步数，N = batch 大小
    #     device = self.device

    #     # 获取模型权重的 dtype（FP16 或 FP32）
    #     model_dtype = next(self.policy.parameters()).dtype

    #     # 转换观测中的浮点张量，整数张量保持不变
    #     obs_for_model = {}
    #     for k, v in observations.items():
    #         if torch.is_floating_point(v):
    #             obs_for_model[k] = v.to(model_dtype)
    #         else:
    #             obs_for_model[k] = v

    #     policy = self.policy

    #     # 判断策略是否有 RNN 状态（通过检查 net 属性）
    #     if hasattr(policy, 'net') and policy.net is not None:
    #         recurrent_hidden_states = torch.zeros(
    #             N,
    #             policy.net.num_recurrent_layers,
    #             policy.net.state_encoder.hidden_size,
    #             device=self.device,
    #         )
    #     else:
    #         recurrent_hidden_states = None

    #     # if args.policy_type in ['seq2seq', 'cma']:
    #     #     if not args.DistributedDataParallel:
    #     #         recurrent_hidden_states = torch.zeros(
    #     #             N,
    #     #             self.policy.net.num_recurrent_layers,
    #     #             self.policy.net.state_encoder.hidden_size,
    #     #             device=self.device,
    #     #         )
    #     #     else:
    #     #         recurrent_hidden_states = torch.zeros(
    #     #             N,
    #     #             self.policy.module.net.num_recurrent_layers,
    #     #             self.policy.module.net.state_encoder.hidden_size,
    #     #             device=self.device,
    #     #         )
    #     # else:
    #     #     raise NotImplementedError

    #     AuxLosses.clear()
    #     total_loss = 0.0

    #     # 将观测展平视图转换为 (T, N, ...) 以便按时间步索引
    #     # 注意：observations 是字典，每个值形状为 (B*T, ...)
    #     # 我们将其转换为 (T, N, ...)
    #     obs_reshaped = {}
    #     for k, v in obs_for_model.items():
    #         # v 的形状为 (B*T, *dims)
    #         # 新的形状为 (T, N, *dims)
    #         obs_reshaped[k] = v.view(T, N, *v.shape[1:])

    #     # 同样处理 prev_actions 和 not_done_masks
    #     prev_actions_reshaped = prev_actions.view(T, N, -1)   # (T, N, 1)
    #     masks_reshaped = not_done_masks.view(T, N, -1)       # (T, N, 1)


    #     # 逐时间步循环
    #     for t in range(T):
    #         # 取出当前时间步的观测
    #         obs_t = {k: obs_reshaped[k][t] for k in obs_reshaped}   # 每个形状 (N, ...)
    #         # 强制转换所有浮点张量（防御性）
    #         for k in obs_t:
    #             if torch.is_floating_point(obs_t[k]):
    #                 obs_t[k] = obs_t[k].to(model_dtype)
    #         prev_actions_t = prev_actions_reshaped[t]               # (N, 1)
    #         masks_t = masks_reshaped[t]                             # (N, 1)

    #         # 构建分布（调用 policy.build_distribution）
    #         # 对于 LLaVAPolicy，它会返回一个 Categorical 分布
    #         distribution = self.policy.build_distribution(
    #             obs_t, recurrent_hidden_states, prev_actions_t, masks_t
    #         )
    #         logits = distribution.logits  # (N, num_actions)

    #         # 计算当前时间步的动作损失
    #         # corrected_actions[t] 形状 (N,)
    #         action_loss_t = F.cross_entropy(logits, corrected_actions[t], reduction="none")
    #         # 应用权重 weights[t] (N,) 并计算平均
    #         action_loss_t = (weights[t] * action_loss_t).sum() / weights[t].sum() if weights[t].sum() > 0 else 0.0

    #         # 累积损失
    #         total_loss += action_loss_t

    #         # 辅助损失（如果有）—— 需要按时间步累加，但 AuxLosses 是全局的，我们可以在循环外处理
    #         # 这里简化：保留原有逻辑，在循环结束后统一处理辅助损失
    #         # 但为了匹配原逻辑，我们可以在循环内调用 AuxLosses.reduce 并累加，但 AuxLosses 是全局的，可能已经累积了所有时间步的辅助损失。
    #         # 原代码中 AuxLosses.reduce(aux_mask) 是在循环外，所以为了兼容，我们仍然在循环外处理辅助损失。

    #     # 循环结束后，处理辅助损失（如果 AuxLosses 有值）
    #     # 注意：AuxLosses 是在 build_distribution 过程中可能被填充的，但每次 build_distribution 可能都会添加辅助损失。
    #     # 若 AuxLosses 被设计为累积所有时间步，我们可以在循环外调用 reduce。
    #     # 但为了简单，我们可以在循环内累积辅助损失，但这里为了最小改动，我们假设 AuxLosses 是在循环外一次性计算，但可能不准确。
    #     # 更好的做法是在循环内收集辅助损失列表。

    #     # 我们调整：在循环内调用 AuxLosses 的累加，但因为 AuxLosses 是全局单例，每次调用 reduce 会清空，所以需要在循环内累加。
    #     # 或者我们可以在循环内手动记录辅助损失。

    #     # 这里为了简化，我们采用与原始方法相似的逻辑：将循环内每个时间步的辅助损失收集，然后在循环外统一求和。
    #     # 由于 AuxLosses 的 API 是全局的，我们可以这样：每次 build_distribution 时，AuxLosses 会被添加，但我们需要在循环外 reduce 所有。
    #     # 所以我们保留原样：在循环外调用 AuxLosses.reduce(aux_mask) 一次，但需要 mask 是所有时间步的。
    #     # 但由于我们逐时间步调用，AuxLosses 内部会累加所有时间步的损失，但 reduce 会清空，所以我们可以在循环结束后一次性 reduce。
    #     # 但注意，原代码中 AuxLosses.reduce 是在所有时间步的 build_distribution 之后调用的，但我们是逐时间步，所以我们需要确保 AuxLosses 是累积的。
    #     # 最简单：在循环内对每个时间步，调用 AuxLosses.reduce 并将结果累加，但这样会多次清空。
    #     # 更好的方式：将 AuxLosses 设计为可累加，但为了最小改动，我们可以在循环内调用 AuxLosses.add_loss 之类的，但这里不展开。

    #     # 由于我们不清楚 AuxLosses 的具体实现，假设它是在 build_distribution 时添加，且 reduce 会清空，我们可以这样：
    #     # 在循环内，对每个时间步，调用 AuxLosses.reduce 并将结果累加，但每次 reduce 会清空，所以我们可以在循环外一次性 reduce 所有时间步。
    #     # 但循环外没有所有时间步的 aux_mask，所以我们需要构造 aux_mask。

    #     # 更简单的方案：完全忽略辅助损失，或者仿照原逻辑，在循环外计算辅助损失，但需要知道所有时间步的 aux_loss。
    #     # 鉴于我们主要解决 LLaVAPolicy 的单步处理问题，且 LLaVAPolicy 可能没有辅助损失，我们可以先简单处理：若 AuxLosses 被激活，我们将辅助损失置为0。
    #     # 用户可根据需要扩展。

    #     # 此处我们为了演示，假定没有辅助损失，或从 AuxLosses 获取，但简化。
    #     aux_loss = 0.0

    #     # distribution = self.policy.build_distribution(
    #     #     observations, recurrent_hidden_states, prev_actions, not_done_masks
    #     # )

    #     # if not args.DistributedDataParallel:
    #     #     distribution = self.policy.build_distribution(
    #     #         observations, recurrent_hidden_states, prev_actions, not_done_masks
    #     #     )
    #     # else:
    #     #     distribution = self.policy.module.build_distribution(
    #     #         observations, recurrent_hidden_states, prev_actions, not_done_masks
    #     #     )


    #     # 总损失 = action_loss + aux_loss
    #     loss = total_loss / T  # 平均每个时间步
    #     loss = loss / loss_accumulation_scalar

    #     if backward:
    #         loss.backward()

    #     if step_grad:
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     return loss, action_loss_t.item(), aux_loss   # 注意 action_loss_t 是最后一个时间步的，这里可以返回平均



        # logits = distribution.logits
        # logits = logits.view(T, N, -1)

        # action_loss = F.cross_entropy(
        #     logits.permute(0, 2, 1), corrected_actions, reduction="none"
        # )
        # action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        # aux_mask = (weights > 0).view(-1)
        # aux_loss = AuxLosses.reduce(aux_mask)

        # loss = action_loss + aux_loss
        # loss = loss / loss_accumulation_scalar
        # if backward:
        #     loss.backward()

        # if step_grad:
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()

        # # if isinstance(aux_loss, torch.Tensor):
        # #     aux_loss = aux_loss.item()
        # # return loss.item(), action_loss.item(), aux_loss

        # return loss, action_loss.item(), aux_loss
