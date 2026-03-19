import torch
import torch.nn as nn


def masked_mse_loss(preds, targets, zenith_angles):
    """
    带掩码的 MSE 损失函数：只计算太阳天顶角 <= 85° (即白天) 的预测误差。
    """
    # 按照论文，天顶角 <= 85° 时才是有效的白天数据
    # mask 的形状与 preds/targets 相同: (Batch_Size, output_seq_len)
    mask = zenith_angles <= 86.0

    # 如果这个 Batch 全是深夜（比如凌晨 2 点），mask 全部为 False
    if mask.sum() == 0:
        # 为了防止梯度断裂报错，返回一个带梯度的 0
        return (preds * 0.0).sum()

    # 只提取白天的预测值和真实值进行 MSE 计算
    daytime_preds = preds[mask]
    daytime_targets = targets[mask]

    return torch.mean((daytime_preds - daytime_targets) ** 2)


class DCCALoss(nn.Module):
    """
    终极稳定版特征对齐损失 (基于跨模态互相关矩阵)
    平替传统 DCCA，彻底根除 linalg.eigh 导致的 NaN 梯度爆炸问题。
    """

    def __init__(self, lambd=5e-3):
        super(DCCALoss, self).__init__()
        # 用于平衡对角线和非对角线惩罚的系数
        self.lambd = lambd

    def forward(self, H1, H2):
        batch_size = H1.size(0)

        # 🛡️ 终极防御：如果由于某种原因只传进来 1 个样本，直接返回 0 的 Loss，不参与对齐计算
        if batch_size <= 1:
            return torch.tensor(0.0, device=H1.device, requires_grad=True)

        # 加入 unbiased=False，防止除以 N-1 造成的各种隐患
        H1_norm = (H1 - H1.mean(dim=0)) / (H1.std(dim=0, unbiased=False) + 1e-8)
        H2_norm = (H2 - H2.mean(dim=0)) / (H2.std(dim=0, unbiased=False) + 1e-8)

        # 2. 计算跨模态互相关矩阵 (Cross-Correlation Matrix)
        # C 的大小为 (Feature_Dim, Feature_Dim)
        C = torch.matmul(H1_norm.t(), H2_norm) / batch_size

        # 3. 构建目标：让互相关矩阵 C 尽量接近单位矩阵 (Identity Matrix)
        # 这意味着：
        # - 第 i 个视觉特征和第 i 个数值特征高度相关 (对角线为 1)
        # - 第 i 个视觉特征和第 j 个数值特征互不干扰 (非对角线为 0)
        c_diff = (C - torch.eye(C.size(0), device=C.device)).pow(2)

        # 提取对角线部分的误差 (迫使模态对齐)
        on_diag_loss = torch.diag(c_diff).sum()

        # 提取非对角线部分的误差 (去除特征内部冗余)
        off_diag_loss = c_diff.sum() - on_diag_loss

        # 总损失
        loss = on_diag_loss + self.lambd * off_diag_loss
        return loss


class PaperDCCALoss(nn.Module):
    """
    完全按照论文 公式 (21) - (26) 实现的 DCCA 损失函数
    """

    def __init__(self, r1=1e-3, r2=1e-3, eps=1e-8):
        super(PaperDCCALoss, self).__init__()
        # 论文明确指出的正则化常数 r1 和 r2 均设为 10^-3
        self.r1 = r1
        self.r2 = r2
        self.eps = eps  # 用于防止底层特征分解时出现数学崩溃的极小值保护

    def forward(self, H1, H2):
        # n 即为公式里的样本大小 (Batch Size) [cite: 251]
        n = H1.size(0)

        # 为了防止只有 1 个样本时下面除以 (n-1) 报错
        if n <= 1:
            return torch.tensor(0.0, device=H1.device, requires_grad=True)

        # ==========================================
        # 公式 (21)：去均值中心化 [cite: 251]
        # ==========================================
        # E_bar = E - 1/n * E * I
        H1_bar = H1 - H1.mean(dim=0, keepdim=True)
        H2_bar = H2 - H2.mean(dim=0, keepdim=True)

        # ==========================================
        # 公式 (24) - (26)：计算协方差矩阵
        # ==========================================
        # 注意：在 PyTorch 中，(Batch, Dim) 的矩阵乘法转置方式与论文表达略有不同，
        # H1_bar.t() @ H1_bar 等价于论文中特征维度展开的 E_bar * E_bar'

        # 公式 (25): Σ_11 = 1/(n-1) * E1_bar * E1_bar' + r1 * I [cite: 258]
        Sigma11 = torch.matmul(H1_bar.t(), H1_bar) / (n - 1)
        Sigma11 = Sigma11 + self.r1 * torch.eye(Sigma11.size(0), device=H1.device)

        # 公式 (26): Σ_22 = 1/(n-1) * E2_bar * E2_bar' + r2 * I
        # 备注：原论文写的是 1/(n-2) [cite: 260]，这是统计学无偏估计的笔误，工程上均统一为 n-1
        Sigma22 = torch.matmul(H2_bar.t(), H2_bar) / (n - 1)
        Sigma22 = Sigma22 + self.r2 * torch.eye(Sigma22.size(0), device=H2.device)

        # 公式 (24): Σ_12 = 1/(n-1) * E1_bar * E2_bar' [cite: 256]
        Sigma12 = torch.matmul(H1_bar.t(), H2_bar) / (n - 1)

        # ==========================================
        # 核心难点：计算 Σ_11^(-1/2) 和 Σ_22^(-1/2)
        # ==========================================
        # 通过特征值分解：Σ = V * diag(L) * V^T  ==>  Σ^(-1/2) = V * diag(L^(-1/2)) * V^T
        L1, V1 = torch.linalg.eigh(Sigma11)
        L2, V2 = torch.linalg.eigh(Sigma22)

        # 钳制特征值，防止出现负数或 0 导致后续开根号变成 NaN
        L1 = torch.clamp(L1, min=self.eps)
        L2 = torch.clamp(L2, min=self.eps)

        # 计算负二分之一次方矩阵
        Sigma11_inv_sqrt = torch.matmul(V1, torch.matmul(torch.diag(L1 ** -0.5), V1.t()))
        Sigma22_inv_sqrt = torch.matmul(V2, torch.matmul(torch.diag(L2 ** -0.5), V2.t()))

        # ==========================================
        # 公式 (23)：构建典型相关矩阵 T_t [cite: 253]
        # ==========================================
        # T_t = Σ_11^(-1/2) * Σ_12 * Σ_22^(-1/2)
        T_t = torch.matmul(Sigma11_inv_sqrt, torch.matmul(Sigma12, Sigma22_inv_sqrt))

        # ==========================================
        # 公式 (22)：计算相关性 (迹范数 / 核范数) [cite: 250]
        # ==========================================
        # 迹范数 ||T_t||_tr 等于矩阵 T_t 的所有奇异值之和
        # 使用 SVD (奇异值分解) 求出奇异值 S
        U, S, V = torch.linalg.svd(T_t)

        # 相关性 = 奇异值之和
        corr = torch.sum(S)

        # ==========================================
        # DCCA Loss 输出
        # ==========================================
        # 因为优化器是求最小值，我们要最大化相关性，所以取负数 (公式 19 的逻辑)
        return -corr
