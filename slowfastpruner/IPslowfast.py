# 剪枝技巧:各层conv3部分狠狠的剪枝 点数就上去了
"""
SlowFast Networks for Video Recognition
ICCV 2019, https://arxiv.org/abs/1812.03982
Code adapted from https://github.com/r1ch88/SlowFastNetworks
"""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d
import torch
import torch.nn as nn
import math
import tqdm
import torch
import torch.nn as nn
import math
import tqdm
import heapq
import torch.nn.functional as F  # 必须添加这一行
class InteractionPruner:
    def __init__(self, model, sparsity=0.5, tau_max=2, lam=0.1):
        self.model = model
        self.sparsity = sparsity
        self.tau_max = tau_max
        self.lam = lam
        self.activations = {}
        self.hooks = []
        self.device = next(model.parameters()).device
        self.ordered_layer_names = []
        self.layer_avg_scores = {}

    def register_hooks(self):
        self.hooks = []
        self.activations = {}
        self.ordered_layer_names = []
        
        for name, m in self.model.named_modules():
            # 添加 .conv3 识别
            if isinstance(m, nn.Conv3d) and ('.conv1' in name or '.conv2' in name or '.conv3' in name):
                h = m.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(h)
                self.ordered_layer_names.append(name)

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            z = output.detach() # [B, C, T, H, W]
            z_pool = torch.mean(z, dim=(3, 4)) # [B, C, T]
            
            # 修正点 2：论文友好型时间下采样对齐
            # 如果时间步过长，统一对齐到 8 帧，确保 Granger 分析的计算量可控
            if z_pool.shape[2] > 8:
                # pool1d 作用在最后一个维度，需保持 [B, C, T]
                z_pool = F.adaptive_avg_pool1d(z_pool, 8)
            
            z_final = z_pool.permute(0, 2, 1) # [B, T, C]
            
            # 限制 Batch 显存占用
            if z_final.shape[0] > 64:
                z_final = z_final[:64]
                
            # 修正点 3：EMA 累积逻辑（见下文）
            self._update_activations(name, z_final.to("cpu", dtype=torch.float16))
        return hook
    def _update_activations(self, name, z_new):
        """
        修正点 3：引入 EMA 累积，而非直接覆盖
        """
        if name not in self.activations:
            self.activations[name] = z_new
        else:
            # 确保形状一致（防止最后一个 batch 大小不同）
            if self.activations[name].shape[0] == z_new.shape[0]:
                self.activations[name] = 0.9 * self.activations[name] + 0.1 * z_new
            else:
                # 如果 batch 维度不匹配，则取最小 batch 进行更新或跳过
                min_b = min(self.activations[name].shape[0], z_new.shape[0])
                self.activations[name][:min_b] = 0.9 * self.activations[name][:min_b] + 0.1 * z_new[:min_b]
    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def run_calibration(self, loader, device):
        self.register_hooks()
        self.model.eval()
        print(">>> 正在采集层间交互特征...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(loader, desc="Calibration")):
                if i >= 10: break # 采样 10 个 batch 足够评估交互强度
                vids = batch[0] 
                self.model(vids.float().to(device, non_blocking=True))
        self.remove_hooks()
        

    def estimate_unit_cost(self, module):
        """
        计算剪掉一个输出通道所节省的代价
        """
        in_c = module.in_channels
        k = module.kernel_size 
        params_per_channel = in_c * k[0] * k[1] * k[2]
        return params_per_channel 
    
    @torch.no_grad()
    def prune(self, intra_weight=0.5, inter_weight=0.5, beta=0.9, min_keep_ratio=0.35):
        """
        最终进化版：
        1. Score Normalization: 消除 conv3 的绝对数值霸权。
        2. Cost Weighting: 修正 1x1 卷积的成本评估。
        3. Adaptive Keep Ratio: 允许 conv3 承担更多剪枝任务。
        """
        valid_layer_names = [n for n in self.ordered_layer_names if n in self.activations]
        device = next(self.model.parameters()).device

        # --- 1. 层内能量计算 (舒尔补扰动) ---
        layer_E_intra = {}
        layer_ranks = []
        for name in valid_layer_names:
            z = self.activations[name]
            S = self._compute_granger(z) 
            C_dim = S.shape[0]
            diag_S = torch.diag(S)
            E_intra = torch.zeros(C_dim, device=device)
            
            for k in range(C_dim):
                s_kk = diag_S[k]
                s_rk = S[:, k].clone(); s_rk[k] = 0
                s_kr = S[k, :].clone(); s_kr[k] = 0
                # 扰动能量
                E_intra[k] = torch.norm((s_rk.unsqueeze(1) @ s_kr.unsqueeze(0)) / (s_kk + 1e-7), p=1)
            
            # 存储原始均值用于统计打印
            self.layer_avg_scores[name] = E_intra.mean().item()
            layer_E_intra[name] = E_intra
            
            rank_val = torch.matrix_rank(S.float()).item()
            layer_ranks.append(math.log(rank_val + 1.0))

        # --- 2. 候选池初始化 (引入归一化与成本修正) ---
        num_layers = len(valid_layer_names)
        rank_threshold = torch.tensor(layer_ranks).mean().item()
        layer_info = {}
        pq = [] 

        for l in range(num_layers):
            l_name = valid_layer_names[l]
            E_l = layer_E_intra[l_name]
            m = dict(self.model.named_modules())[l_name]
            
            # A. 跨层路径得分
            path_sum = torch.zeros_like(E_l)
            for m_idx in range(l + 1, min(l + 5, num_layers)):
                dist = m_idx - l
                avg_log_rank = sum(layer_ranks[l:m_idx]) / dist
                weight = math.exp(avg_log_rank - rank_threshold) * (beta ** dist)
                path_sum += weight * E_l
            
            raw_scores = (intra_weight * E_l) + (inter_weight * (path_sum / (num_layers - l) if num_layers > l else 1))
            
            # B. 【重要修正】层内归一化：将每一层的得分映射到同一量级 [0, 1]
            #这样 conv3 的“弱通道”才会被排到堆顶，优先于 conv1 的“强通道”
            norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
            l_min_keep = min_keep_ratio 
            unit_cost = (self.estimate_unit_cost(m) ) + 1e-9   
            u_count = len(norm_scores)
            max_p = u_count - max(1, int(u_count * l_min_keep))
            layer_info[l_name] = {'total': u_count, 'pruned': 0, 'max_prunable': max_p}
            for i, s in enumerate(norm_scores):
                # 此时 base_unit_score 反映的是“相对本层的重要性 / 物理代价”
                base_unit_score = s.item() / unit_cost
                heapq.heappush(pq, [base_unit_score, l_name, i, base_unit_score, unit_cost, 0])

        # --- 3. 动态博弈剪枝核心 (基于 Heap 惰性更新) ---
        total_p = sum(p.numel() for p in self.model.parameters())
        target_red = total_p * self.sparsity
        current_red = 0
        registry = {name: set() for name in valid_layer_names}

        

        with tqdm.tqdm(total=int(target_red), desc="结构化对齐剪枝中") as pbar:
            while current_red < target_red and pq:
                prio, ln, idx, base_score, cost, last_cnt = heapq.heappop(pq)
                
                info = layer_info[ln]
                if info['pruned'] >= info['max_prunable']:
                    continue                
                # 惰性生存机制：随着剪枝增加，剩余通道的“价格”呈指数级上升
                if last_cnt != info['pruned']:
                    # 惩罚因子：已剪越多，剩下的越贵
                    survival_factor = math.exp(info['pruned'] / info['total'])
                    new_priority = base_score * survival_factor
                    heapq.heappush(pq, [new_priority, ln, idx, base_score, cost, info['pruned']])
                    continue
                
                # 确认执行剪枝
                registry[ln].add(idx)
                current_red += cost
                info['pruned'] += 1
                pbar.update(int(cost))

        # --- 4. 掩码生成 ---
        for name in valid_layer_names:
            m = dict(self.model.named_modules())[name]
            mask = torch.ones(m.out_channels)
            for idx in registry[name]:
                mask[idx] = 0.0
            m.register_buffer('interaction_mask', mask.to(device))

        print(f"\n>>> 剪枝完成！有效参数减少率: {current_red/total_p:.2%}")
        return {'sparsity': current_red/total_p, 'total_p': total_p, 'current_red': current_red}

    def _compute_granger(self, z):
        """
        手动实现相关系数计算，兼容旧版 PyTorch
        """
        N, T, C = z.shape
        # 转到 GPU 并增加一个小偏移量防止全零
        z = z.to(self.device).float() + 1e-6
        delta = 1e-7
        
        # 1. 基础预测逻辑
        taus = torch.arange(1, self.tau_max + 1, device=self.device).float()
        w_tau = torch.exp(-self.lam * taus)
        w_tau /= w_tau.sum()
        
        z_unfold = z.unfold(1, self.tau_max + 1, 1)
        current = z_unfold[:, :, :, -1] 
        past = z_unfold[:, :, :, :-1]   

        # 自预测残差
        pred_self = torch.einsum('ntcm,m->ntc', past, w_tau)
        err_self = torch.mean((current - pred_self)**2, dim=(0, 1)) # [C]

        # 2. 手动计算相关系数矩阵 (corrcoef)
        # z_flat 形状: [C, N*T]
        z_flat = z.permute(2, 0, 1).reshape(C, -1) 
        
        # 去均值
        z_centered = z_flat - z_flat.mean(dim=1, keepdim=True)
        # 计算协方差矩阵: (z * z^T) / (n - 1)
        cov = torch.matmul(z_centered, z_centered.t()) / (z_flat.shape[1] - 1)
        # 获取标准差: sqrt(diag(cov))
        std = torch.sqrt(torch.diag(cov) + delta)
        # 相关系数矩阵: cov / (std_i * std_j)
        S = (cov / std.unsqueeze(1)) / std.unsqueeze(0)
        S = S.abs() # 取绝对值表示交互强度

        # 3. 结合 Granger 误差增益
        # 误差小的通道重要性更高，取对数倒数并归一化
        gain_weight = torch.log(1.0 / (err_self + delta))
        gain_weight = (gain_weight - gain_weight.min()) / (gain_weight.max() - gain_weight.min() + delta)
        
        # 赋予权重
        S = S * gain_weight.unsqueeze(0)
        
        # 归一化到 [0.1, 1.0]
        S_min, S_max = S.min(), S.max()
        S = 0.9 * (S - S_min) / (S_max - S_min + delta) + 0.1
        return S

    def get_stats(self):
        """
        打印每层的剪枝比例及交互得分
        """
        print("\n" + "="*85)
        print(f"{'Layer Name':<40} | {'Score':<10} | {'Kept/Total':<15} | {'Ratio'}")
        print("-" * 85)
        
        # 获取所有层的平均分，用于计算相对重要性
        all_scores = list(self.layer_avg_scores.values())
        max_score = max(all_scores) if all_scores else 1.0

        for name, m in self.model.named_modules():
            if hasattr(m, 'interaction_mask'):
                kept = int(m.interaction_mask.sum())
                total = m.out_channels
                
                # 获取存好的平均得分
                avg_score = self.layer_avg_scores.get(name, 0.0)
                
                # 归一化得分柱状图（简单可视化）
                bar_len = int((avg_score / max_score) * 10)
                score_bar = "█" * bar_len
                
                print(f"{name:40} | {avg_score:10.4f} | {kept:>4}/{total:<4} | {kept/total:>6.1%} {score_bar}")
        print("="*85 + "\n")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, strides=1, downsample=None, head_conv=1,
                 norm_layer=BatchNorm3d, norm_kwargs=None, layer_name=''):
        super(Bottleneck, self).__init__()

        # Conv 1
        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 1, 1), 
                                   padding=(1, 0, 0), bias=False)
        else:
            raise ValueError("Unsupported head_conv!")
        
        self.bn1 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        
        # Conv 2
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=(1, 3, 3),
                               stride=(1, strides, strides), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        
        # Conv 3
        self.conv3 = nn.Conv3d(in_channels=planes, out_channels=planes * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(num_features=planes * self.expansion, **({} if norm_kwargs is None else norm_kwargs))
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _apply_mask(self, x, layer):
        """应用来自 InteractionPruner 的剪枝掩码"""
        if hasattr(layer, 'interaction_mask'):
            # 将 [C] 维度的掩码广播到 [B, C, T, H, W]
            mask = layer.interaction_mask.view(1, -1, 1, 1, 1)
            return x * mask
        return x

    def forward(self, x):
        identity = x

        # Conv 1
        out = self.conv1(x)
        out = self._apply_mask(out, self.conv1)
        out = self.bn1(out)
        out = self.relu(out)

        # Conv 2
        out = self.conv2(out)
        out = self._apply_mask(out, self.conv2)
        out = self.bn2(out)
        out = self.relu(out)

        # Conv 3 (新增剪枝支持)
        out = self.conv3(out)
        # 如果你也给 conv3 注册了掩码，这里会生效
        # 注意：这里的 mask 会应用在相加之前，因此不会改变张量形状
        out = self._apply_mask(out, self.conv3) 
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out

# --- 2. 完整的 SlowFast 模型类 ---
class SlowFast(nn.Module):
    def __init__(self,
                 num_classes,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3], # 默认 ResNet50 结构
                 num_block_temp_kernel_fast=None,
                 num_block_temp_kernel_slow=None,
                 dropout_ratio=0.5,
                 alpha=8,
                 beta_inv=8,
                 fusion_conv_channel_ratio=2,
                 fusion_kernel_size=5,
                 width_per_group=64,
                 slow_temporal_stride=16,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 **kwargs):
        super(SlowFast, self).__init__()
        
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_size = fusion_kernel_size
        self.width_per_group = width_per_group
        self.dim_inner = width_per_group
        self.out_dim_ratio = beta_inv // fusion_conv_channel_ratio
        self.slow_temporal_stride = slow_temporal_stride
        self.dropout_ratio = dropout_ratio

        # --- Fast Pathway ---
        fast_in_c = width_per_group // beta_inv
        self.fast_conv1 = nn.Conv3d(3, fast_in_c, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = norm_layer(num_features=fast_in_c, **({} if norm_kwargs is None else norm_kwargs))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.fast_res2 = self._make_layer_fast(fast_in_c, self.dim_inner // beta_inv, layers[0], head_conv=3, norm_layer=norm_layer)
        self.fast_res3 = self._make_layer_fast(self.dim_inner * 4 // beta_inv, self.dim_inner * 2 // beta_inv, layers[1], strides=2, head_conv=3, norm_layer=norm_layer)
        self.fast_res4 = self._make_layer_fast(self.dim_inner * 8 // beta_inv, self.dim_inner * 4 // beta_inv, layers[2], strides=2, head_conv=3, norm_layer=norm_layer)
        self.fast_res5 = self._make_layer_fast(self.dim_inner * 16 // beta_inv, self.dim_inner * 8 // beta_inv, layers[3], strides=2, head_conv=3, norm_layer=norm_layer)

        # --- Lateral Connections ---
        self.lateral_p1 = self._make_lateral_conv(fast_in_c, norm_layer)
        self.lateral_res2 = self._make_lateral_conv(self.dim_inner * 4 // beta_inv, norm_layer)
        self.lateral_res3 = self._make_lateral_conv(self.dim_inner * 8 // beta_inv, norm_layer)
        self.lateral_res4 = self._make_lateral_conv(self.dim_inner * 16 // beta_inv, norm_layer)

        # --- Slow Pathway ---
        self.slow_conv1 = nn.Conv3d(3, width_per_group, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = norm_layer(num_features=width_per_group, **({} if norm_kwargs is None else norm_kwargs))
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 这里的 in_channels 考虑了横向连接拼接后的通道数
        self.slow_res2 = self._make_layer_slow(width_per_group + (fast_in_c * fusion_conv_channel_ratio), self.dim_inner, layers[0], head_conv=1, norm_layer=norm_layer)
        self.slow_res3 = self._make_layer_slow(self.dim_inner * 4 + (self.dim_inner * 4 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 2, layers[1], strides=2, head_conv=1, norm_layer=norm_layer)
        self.slow_res4 = self._make_layer_slow(self.dim_inner * 8 + (self.dim_inner * 8 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 4, layers[2], strides=2, head_conv=3, norm_layer=norm_layer)
        self.slow_res5 = self._make_layer_slow(self.dim_inner * 16 + (self.dim_inner * 16 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 8, layers[3], strides=2, head_conv=3, norm_layer=norm_layer)

        # --- Classifier ---
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.dp = nn.Dropout(p=self.dropout_ratio)
        self.feat_dim = (self.dim_inner * 32 // beta_inv) + (self.dim_inner * 32)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def _make_lateral_conv(self, in_c, norm_layer):
        return nn.Sequential(
            nn.Conv3d(in_c, in_c * self.fusion_conv_channel_ratio, kernel_size=(self.fusion_kernel_size, 1, 1), 
                      stride=(self.alpha, 1, 1), padding=(self.fusion_kernel_size // 2, 0, 0), bias=False),
            norm_layer(num_features=in_c * self.fusion_conv_channel_ratio),
            nn.ReLU(inplace=True)
        )

    def _make_layer_fast(self, inplanes, planes, num_blocks, strides=1, head_conv=1, norm_layer=BatchNorm3d):
        downsample = None
        if strides != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=(1, strides, strides), bias=False),
                norm_layer(num_features=planes * Bottleneck.expansion)
            )
        layers = [Bottleneck(inplanes, planes, strides, downsample, head_conv, norm_layer)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(inplanes, planes, 1, None, head_conv, norm_layer))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, inplanes, planes, num_blocks, strides=1, head_conv=1, norm_layer=BatchNorm3d):
        downsample = None
        if strides != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=(1, strides, strides), bias=False),
                norm_layer(num_features=planes * Bottleneck.expansion)
            )
        layers = [Bottleneck(inplanes, planes, strides, downsample, head_conv, norm_layer)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(inplanes, planes, 1, None, head_conv, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x 形状: [B, 3, T, H, W]
        fast_input = x
        # Slow 路径对时间维度进行稀疏采样
        slow_input = x[:, :, ::self.slow_temporal_stride // 2, :, :]

        # --- Fast Path ---
        f = self.fast_conv1(fast_input)
        f = self.fast_bn1(f)
        f = self.fast_relu(f)
        f_pool = self.fast_maxpool(f)
        
        l1 = self.lateral_p1(f_pool)
        f_res2 = self.fast_res2(f_pool)
        l2 = self.lateral_res2(f_res2)
        f_res3 = self.fast_res3(f_res2)
        l3 = self.lateral_res3(f_res3)
        f_res4 = self.fast_res4(f_res3)
        l4 = self.lateral_res4(f_res4)
        f_res5 = self.fast_res5(f_res4)
        f_out = self.avg(f_res5).view(f_res5.size(0), -1)

        # --- Slow Path ---
        s = self.slow_conv1(slow_input)
        s = self.slow_bn1(s)
        s = self.slow_relu(s)
        s_pool = self.slow_maxpool(s)
        
        s_res2 = self.slow_res2(torch.cat([s_pool, l1], dim=1))
        s_res3 = self.slow_res3(torch.cat([s_res2, l2], dim=1))
        s_res4 = self.slow_res4(torch.cat([s_res3, l3], dim=1))
        s_res5 = self.slow_res5(torch.cat([s_res4, l4], dim=1))
        s_out = self.avg(s_res5).view(s_res5.size(0), -1)

        # --- Fusion ---
        out = torch.cat([s_out, f_out], dim=1)
        out = self.dp(out)
        out = self.fc(out)
        return out

    def get_detailed_pruning_report(self):
        """
        修正版：以全模型总参数量为分母，统计真实的物理参数减少比例。
        """
        # 1. 计算整个模型的原始总参数量 (包含所有层)
        total_model_params = sum(p.numel() for p in self.parameters())
        
        total_pruned_params = 0
        layer_count = 0

        for name, module in self.named_modules():
            # 只有卷积层（Conv3d）会被 InteractionPruner 处理
            if hasattr(module, 'interaction_mask'):
                mask = module.interaction_mask
                # 被剪掉的通道数 = 总通道 - 保留通道
                pruned_channels = mask.numel() - int(mask.sum().item())
                
                # 计算该层单个输出通道对应的参数量 (in_c * kt * kh * kw)
                weight_shape = module.weight.shape
                # weight.shape: [out_c, in_c, kt, kh, kw]
                params_per_channel = torch.prod(torch.tensor(weight_shape[1:])).item()
                
                # 累计被剪掉的参数
                total_pruned_params += pruned_channels * params_per_channel
                layer_count += 1

        if total_model_params == 0:
            return {'sparsity': 0.0, 'pruned_layers': 0}

        # 核心：剪掉的参数 / 全模型总参数
        actual_sparsity = total_pruned_params / total_model_params

        report = {
            'sparsity': actual_sparsity,          # 相对于全模型的剪枝率
            'total_model_params': total_model_params,
            'total_pruned_params': total_pruned_params,
            'pruned_layers': layer_count
        }

        print(f"\n" + "="*40)
        print(f">>> 全模型参数量: {total_model_params:,}")
        print(f">>> 已剪掉参数量: {total_pruned_params:,}")
        print(f">>> 实际总剪枝率: {actual_sparsity:.2%}")
        print(f">>> 涉及剪枝层数: {layer_count}")
        print("="*40 + "\n")
        
        return report
    # def get_parameter_distribution(self):
    #     """
    #     统计可剪枝部分与总参数的比例
    #     """
    #     total_params = sum(p.numel() for p in self.parameters())
    #     prunable_params = 0
    #     fixed_params = 0
        
    #     # 细分统计
    #     categories = {
    #         'prunable_convs': 0,  # conv1, 2, 3
    #         'stem_lateral': 0,    # fast_conv1, slow_conv1, lateral layers
    #         'classifier': 0,      # fc
    #         'others': 0           # BN, Downsample 等
    #     }

    #     for name, m in self.named_modules():
    #         # 获取该层自身的参数量（不包含子模块）
    #         layer_params = sum(p.numel() for p in m.parameters(recurse=False))
    #         if layer_params == 0: continue

    #         if isinstance(m, nn.Conv3d) and any(x in name for x in ['.conv1', '.conv2', '.conv3']):
    #             categories['prunable_convs'] += layer_params
    #             prunable_params += layer_params
    #         elif 'conv1' in name or 'lateral' in name:
    #             categories['stem_lateral'] += layer_params
    #             fixed_params += layer_params
    #         elif 'fc' in name:
    #             categories['classifier'] += layer_params
    #             fixed_params += layer_params
    #         else:
    #             categories['others'] += layer_params
    #             fixed_params += layer_params

    #     print("\n" + "="*50)
    #     print(f"{'Category':<20} | {'Params':<15} | {'Ratio'}")
    #     print("-" * 50)
    #     for cat, val in categories.items():
    #         print(f"{cat:<20} | {val:>15,} | {val/total_params:>7.2%}")
    #     print("-" * 50)
    #     print(f"{'TOTAL':<20} | {total_params:>15,} | 100.00%")
    #     print(f"{'PRUNABLE TOTAL':<20} | {prunable_params:>15,} | {prunable_params/total_params:>7.2%}")
    #     print("="*50 + "\n")

    #     return categories



def slowfast_16x8_resnet101_kinetics400(num_classes):
    model = SlowFast(num_classes=num_classes,
                     layers=[3, 4, 23, 3],
                     pretrained=False,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     bn_eval=False,
                     partial_bn=False,
                     bn_frozen=False)
    return model





