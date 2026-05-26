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



class WandaPruner:
    def __init__(self, model, sparsity=0.5):
        self.model = model
        self.sparsity = sparsity
        self.activations_norm = {}  # 存储输入激活值的 L2 范数
        self.hooks = []
        self.device = next(model.parameters()).device
        self.ordered_layer_names = []
        self.layer_avg_scores = {}

    def register_hooks(self):
        self.hooks = []
        self.activations_norm = {}
        self.ordered_layer_names = []
        
        for name, m in self.model.named_modules():
            # 识别 SlowFast 中的主要卷积层
            if isinstance(m, nn.Conv3d) and ('.conv1' in name or '.conv2' in name or '.conv3' in name):
                h = m.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(h)
                self.ordered_layer_names.append(name)

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            # input[0] 形状: [B, C_in, T, H, W]
            x = input[0].detach()
            
            # Wanda 需要计算输入特征的 L2 范数：\|X\|_2
            # 针对卷积，我们在 [B, T, H, W] 维度上计算，保留 C_in 维度
            # 累加范数以进行校准（Calibration）
            norm = torch.norm(x, p=2, dim=(0, 2, 3, 4)) 
            
            if name not in self.activations_norm:
                self.activations_norm[name] = norm.to("cpu")
            else:
                # 滚动平均或累加
                self.activations_norm[name] += norm.to("cpu")
        return hook

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def run_calibration(self, loader, device):
        self.register_hooks()
        self.model.eval()
        print(">>> 正在收集 Wanda 激活值统计量...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(loader, desc="Wanda Calibration")):
                if i >= 10: break 
                vids = batch[0] 
                self.model(vids.float().to(device, non_blocking=True))
        self.remove_hooks()

    @torch.no_grad()
    def prune(self, **kwargs):
        """
        Wanda 剪枝核心实现
        """
        device = next(self.model.parameters()).device
        valid_layer_names = [n for n in self.ordered_layer_names if n in self.activations_norm]
        
        total_params_pruned = 0
        total_params_all = 0

        print(f">>> 开始 Wanda 剪枝 (目标稀疏度: {self.sparsity})...")

        for name in valid_layer_names:
            m = dict(self.model.named_modules())[name]
            
            # 1. 获取权重 |W| -> [C_out, C_in, KT, KH, KW]
            w = m.weight.data.abs()
            
            # 2. 获取输入激活值范数 ||X|| -> [C_in]
            # 形状对齐：将 [C_in] 扩展到与权重兼容的形状 [1, C_in, 1, 1, 1]
            x_norm = self.activations_norm[name].to(device)
            x_norm = x_norm.view(1, -1, 1, 1, 1)
            
            # 3. 计算 Wanda Score: S = |W| * ||X||
            wanda_score = w * x_norm
            
            # 4. 确定阈值（每层局部剪枝）
            # 针对输出通道（out_channels）进行评估
            # 也可以全局评估，但 Wanda 原作推荐 Layer-wise
            flat_score = wanda_score.view(m.out_channels, -1)
            
            # 计算每个输出神经元对应的输入连接重要性
            # 或者简单地对整个层的 score 排序
            # 这里采用原文的 per-output-channel 比较或全局每层比较
            # 我们简化为每层统一排序：
            all_scores = wanda_score.view(-1)
            k = int(len(all_scores) * self.sparsity)
            if k > 0:
                threshold, _ = torch.kthvalue(all_scores, k)
                
                # 5. 生成并应用 Mask
                # Wanda 通常是权重级的剪枝（Unstructured），但可以模拟通道剪枝
                # 这里为了适配你的 Bottleneck _apply_mask，我们生成通道掩码
                # 方案：如果一个输出通道的平均 Score 低于阈值，则剪掉
                channel_scores = wanda_score.mean(dim=(1, 2, 3, 4))
                self.layer_avg_scores[name] = channel_scores.mean().item()
                
                c_k = int(m.out_channels * self.sparsity)
                if c_k > 0:
                    c_threshold, _ = torch.kthvalue(channel_scores, c_k)
                    mask = (channel_scores > c_threshold).float()
                else:
                    mask = torch.ones(m.out_channels).to(device)
            else:
                mask = torch.ones(m.out_channels).to(device)
                self.layer_avg_scores[name] = 0.0

            m.register_buffer('interaction_mask', mask.to(device))
            
            # 统计
            num_pruned = (mask == 0).sum().item() * (m.weight.numel() / m.out_channels)
            total_params_pruned += num_pruned
            total_params_all += m.weight.numel()

        actual_sparsity = total_params_pruned / total_params_all if total_params_all > 0 else 0
        return {'sparsity': actual_sparsity, 'total_p': total_params_all, 'current_red': total_params_pruned}

    def get_stats(self):
        """打印每层的剪枝统计"""
        print("\n" + "="*85)
        print(f"{'Layer Name':<40} | {'Wanda Score':<12} | {'Kept/Total':<15} | {'Ratio'}")
        print("-" * 85)
        
        scores = list(self.layer_avg_scores.values())
        max_s = max(scores) if scores else 1.0

        for name, m in self.model.named_modules():
            if hasattr(m, 'interaction_mask'):
                kept = int(m.interaction_mask.sum())
                total = m.out_channels
                avg_s = self.layer_avg_scores.get(name, 0.0)
                bar = "█" * int((avg_s / max_s) * 10) if max_s > 0 else ""
                print(f"{name:40} | {avg_s:12.4f} | {kept:>4}/{total:<4} | {kept/total:>6.1%} {bar}")
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





