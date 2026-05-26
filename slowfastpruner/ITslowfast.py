# 剪枝技巧:各层conv3部分狠狠的剪枝 点数就上去了

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


class ITPruner:
    def __init__(self, model, sparsity=0.5, device='cuda:0'):
        self.model = model
        self.sparsity = sparsity
        self.device = torch.device(device)
        self.activations = {}
        self.hooks = []
        self.ordered_layer_names = []
        self.layer_avg_scores = {} # 用于 get_stats 打印数据

    def register_hooks(self):
        self.hooks = []
        self.activations = {}
        self.ordered_layer_names = []
        
        for name, m in self.model.named_modules():
            # 保持原有的卷积层识别逻辑
            if isinstance(m, nn.Conv3d) and ('.conv1' in name or '.conv2' in name or '.conv3' in name):
                h = m.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(h)
                self.ordered_layer_names.append(name)

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            z = output.detach()
            # 针对 3D 视频特征优化：[B, C, T, H, W] -> [B, C*T]
            # 空间池化以降低 HSIC 计算的内存压力
            z_pool = torch.mean(z, dim=(3, 4)) 
            z_final = z_pool.reshape(z_pool.shape[0], -1)
            
            # 显存保护：HSIC 随 Batch 平方增长，建议限制在 128-256
            if z_final.shape[0] > 128:
                z_final = z_final[:128]
            
            self.activations[name] = z_final.to(self.device, dtype=torch.float32)
        return hook

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def run_calibration(self, loader, device):
        """接口保持不变"""
        self.register_hooks()
        self.model.eval()
        print(">>> 正在采集层间信息流特征 (nHSIC)...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(loader, desc="Calibration")):
                if i >= 10: break 
                vids = batch[0] 
                self.model(vids.float().to(device, non_blocking=True))
        self.remove_hooks()

    def _compute_nhsic(self, x, y):
        """核心算法：计算归一化 HSIC (Hilbert-Schmidt Independence Criterion)"""
        n = x.size(0)
        if n < 2: return torch.tensor(0.0, device=self.device)

        # 中心化矩阵 H = I - 1/n
        H = torch.eye(n, device=self.device) - (1.0 / n)
        
        # 计算线性核矩阵并中心化
        K = torch.mm(x, x.t())
        L = torch.mm(y, y.t())
        Kc = torch.mm(torch.mm(H, K), H)
        Lc = torch.mm(torch.mm(H, L), H)
        
        # 计算归一化统计量
        hsic_xy = torch.trace(torch.mm(Kc, Lc))
        hsic_xx = torch.trace(torch.mm(Kc, Kc))
        hsic_yy = torch.trace(torch.mm(Lc, Lc))
        
        return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)

    def estimate_unit_cost(self, module):
        """接口保持不变：计算通道参数代价"""
        in_c = module.in_channels
        k = module.kernel_size 
        return in_c * k[0] * k[1] * k[2]

    @torch.no_grad()
    def prune(self, min_keep_ratio=0.1):
        """
        基于信息冗余度的博弈剪枝
        """
        valid_layer_names = [n for n in self.ordered_layer_names if n in self.activations]
        L = len(valid_layer_names)
        
        # 1. 构建层间冗余矩阵 (Independence Matrix)
        print(f">>> 构建 {L} 层冗余依赖矩阵...")
        importance_matrix = torch.zeros((L, L), device=self.device)
        for i in range(L):
            for j in range(i, L):
                val = self._compute_nhsic(self.activations[valid_layer_names[i]], 
                                         self.activations[valid_layer_names[j]])
                importance_matrix[i, j] = importance_matrix[j, i] = val

        # 2. 计算层重要性得分
        # 冗余度定义：该层与其他层信息的平均相关性
        redundancy = torch.sum(importance_matrix, dim=1) - 1.0 
        scores = -redundancy # 相关性越低（独立性越高），得分越高
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # 3. 填充优先队列 (PQ)
        pq = []
        layer_info = {}
        for idx, name in enumerate(valid_layer_names):
            m = dict(self.model.named_modules())[name]
            self.layer_avg_scores[name] = scores[idx].item() # 存入用于统计
            
            # 计算该层每个通道的 L1 权重重要性
            w_norm = torch.norm(m.weight.data.view(m.out_channels, -1), p=1, dim=1)
            # 结合层重要性得分进行缩放
            norm_scores = (w_norm - w_norm.min()) / (w_norm.max() - w_norm.min() + 1e-8)
            combined_scores = norm_scores * (0.5 + scores[idx]) # 层权重偏置
            
            cost = self.estimate_unit_cost(m) + 1e-9
            max_p = m.out_channels - max(1, int(m.out_channels * min_keep_ratio))
            layer_info[name] = {'total': m.out_channels, 'pruned': 0, 'max_prunable': max_p}

            for i, s in enumerate(combined_scores):
                # 初始优先级：得分 / 代价
                prio = s.item() / cost
                heapq.heappush(pq, [prio, name, i, s.item(), cost, 0])

        # 4. 执行博弈剪枝 (PQ 贪婪搜索)
        total_p = sum(p.numel() for p in self.model.parameters())
        target_red = total_p * self.sparsity
        current_red = 0
        registry = {name: set() for name in valid_layer_names}

        with tqdm.tqdm(total=int(target_red), desc="IT-Pruning") as pbar:
            while current_red < target_red and pq:
                prio, ln, idx, base_score, cost, last_cnt = heapq.heappop(pq)
                info = layer_info[ln]
                
                if info['pruned'] >= info['max_prunable']: continue

                # 惰性生存机制惩罚：已剪越多，剩下的通道“价值”被抬得越高
                if last_cnt != info['pruned']:
                    survival_penalty = math.pow(1.5, (info['pruned'] / info['total']) * 10)
                    heapq.heappush(pq, [base_score * survival_penalty / cost, ln, idx, base_score, cost, info['pruned']])
                    continue

                registry[ln].add(idx)
                current_red += cost
                info['pruned'] += 1
                pbar.update(int(cost))

        # 5. 生成 interaction_mask
        for name in valid_layer_names:
            m = dict(self.model.named_modules())[name]
            mask = torch.ones(m.out_channels)
            for idx in registry[name]:
                mask[idx] = 0.0
            m.register_buffer('interaction_mask', mask.to(self.device))

        return {'actual_sparsity': current_red/total_p}

    def get_stats(self):
        """保持接口不变：打印每层状态"""
        print("\n" + "="*85)
        print(f"{'Layer Name':<40} | {'Score':<10} | {'Kept/Total':<15} | {'Ratio'}")
        print("-" * 85)
        all_scores = list(self.layer_avg_scores.values())
        max_s = max(all_scores) if all_scores else 1.0

        for name, m in self.model.named_modules():
            if hasattr(m, 'interaction_mask'):
                kept = int(m.interaction_mask.sum())
                total = m.out_channels
                avg_s = self.layer_avg_scores.get(name, 0.0)
                bar = "█" * int((avg_s / max_s) * 10)
                print(f"{name:40} | {avg_s:10.4f} | {kept:>4}/{total:<4} | {kept/total:>6.1%} {bar}")
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





