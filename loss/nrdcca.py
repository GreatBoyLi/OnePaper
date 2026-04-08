import torch
import torch.nn as nn


def compute_cca_correlation(H1, H2, r=1e-3):
    """
    计算两组特征之间的典型相关性 (CCA Correlation)
    🚨 加入了极高数值稳定性的保护机制 (Float64 求导 + 特征预标准化)
    """
    N = H1.size(0)

    # 极小 Batch 保护
    if N < 2:
        return torch.tensor(0.0, device=H1.device, requires_grad=True)

    # ---------------------------------------------------------
    # 🛡️ 保护 1: 特征中心化与标准化 (CCA对缩放不变，但能救 GPU 的命)
    # ---------------------------------------------------------
    H1_bar = H1 - H1.mean(dim=0)
    H2_bar = H2 - H2.mean(dim=0)

    # 除以标准差，将特征压缩到均值为0，方差约等于1的空间
    H1_bar = H1_bar / (H1_bar.std(dim=0) + 1e-8)
    H2_bar = H2_bar / (H2_bar.std(dim=0) + 1e-8)

    # 清洗掉可能出现的极端异常值 (NaN / Inf)
    H1_bar = torch.nan_to_num(H1_bar, nan=0.0, posinf=0.0, neginf=0.0)
    H2_bar = torch.nan_to_num(H2_bar, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------
    # 2. 计算自协方差与跨协方差矩阵
    # ---------------------------------------------------------
    # 适度调大 Ridge 参数 r (从 1e-4 提升到 1e-3)，为低秩矩阵提供强有力的对角线支撑
    Sigma11 = (1.0 / (N - 1)) * torch.matmul(H1_bar.t(), H1_bar) + r * torch.eye(H1.size(1), device=H1.device)
    Sigma22 = (1.0 / (N - 1)) * torch.matmul(H2_bar.t(), H2_bar) + r * torch.eye(H2.size(1), device=H2.device)
    Sigma12 = (1.0 / (N - 1)) * torch.matmul(H1_bar.t(), H2_bar)

    # ---------------------------------------------------------
    # 🛡️ 保护 2: 矩阵负半次幂 (采用 Float64 降维打击 Error 129)
    # ---------------------------------------------------------
    def inv_sqrt(Sigma):
        # 强制转换为双精度 Float64，解决 LAPACK 底层 eigh 不收敛的死穴
        Sigma_d = Sigma.double()

        # 特征值分解
        L, V = torch.linalg.eigh(Sigma_d)

        # 🛡️ 保护 3: 暴力截断所有小于 1e-6 的特征值，防止后续除以 0 产生 NaN
        L = torch.clamp(L, min=1e-6)
        inv_sqrt_L = torch.diag(1.0 / torch.sqrt(L))

        # 重构矩阵，并转回 Float32 交还给 PyTorch 继续梯度反向传播
        res = torch.matmul(V, torch.matmul(inv_sqrt_L, V.t()))
        return res.float()

    Sigma11_inv_sqrt = inv_sqrt(Sigma11)
    Sigma22_inv_sqrt = inv_sqrt(Sigma22)

    # ---------------------------------------------------------
    # 4. 归一化互协方差矩阵 T 与最终相关性
    # ---------------------------------------------------------
    T = torch.matmul(Sigma11_inv_sqrt, torch.matmul(Sigma12, Sigma22_inv_sqrt))

    # 使用奇异值之和 (Nuclear Norm) 代表总体相关性
    S = torch.linalg.svdvals(T)
    return torch.sum(S)


class NRDCCALoss(nn.Module):
    """
    NeurIPS 2024: NR-DCCA (Noise Regularization DCCA)
    """

    def __init__(self, alpha=2.0, outdim=256):
        super(NRDCCALoss, self).__init__()
        self.alpha = alpha
        self.outdim = outdim

    def forward(self, X1, X2, Z1, Z2, A1, A2, Z_A1, Z_A2):
        # 1. 展平原始输入
        X1_flat = X1.view(X1.size(0), -1)
        X2_flat = X2.view(X2.size(0), -1)
        A1_flat = A1.view(A1.size(0), -1)
        A2_flat = A2.view(A2.size(0), -1)

        # 2. 截断加速：应对超高维的原始图像和噪声
        X1_sub = X1_flat[:, :self.outdim] if X1_flat.size(1) > self.outdim else X1_flat
        X2_sub = X2_flat[:, :self.outdim] if X2_flat.size(1) > self.outdim else X2_flat
        A1_sub = A1_flat[:, :self.outdim] if A1_flat.size(1) > self.outdim else A1_flat
        A2_sub = A2_flat[:, :self.outdim] if A2_flat.size(1) > self.outdim else A2_flat

        # 🛡️ 保护 4: 对特征 Z 也进行适当截断
        # 因为 DCCA 提取的特征维度 (384) 远大于 Batch Size (32)，算出的协方差极其稀疏
        # 截断到 outdim 可以进一步稳定矩阵分解，同时不会影响多模态对齐的核心表征
        Z1_sub = Z1[:, :self.outdim] if Z1.size(1) > self.outdim else Z1
        Z2_sub = Z2[:, :self.outdim] if Z2.size(1) > self.outdim else Z2
        Z_A1_sub = Z_A1[:, :self.outdim] if Z_A1.size(1) > self.outdim else Z_A1
        Z_A2_sub = Z_A2[:, :self.outdim] if Z_A2.size(1) > self.outdim else Z_A2

        # 3. 核心计算
        dcca_corr = compute_cca_correlation(Z1_sub, Z2_sub)

        corr_X1_A1 = compute_cca_correlation(X1_sub, A1_sub)
        corr_Z1_ZA1 = compute_cca_correlation(Z1_sub, Z_A1_sub)
        zeta_1 = torch.abs(corr_Z1_ZA1 - corr_X1_A1)

        corr_X2_A2 = compute_cca_correlation(X2_sub, A2_sub)
        corr_Z2_ZA2 = compute_cca_correlation(Z2_sub, Z_A2_sub)
        zeta_2 = torch.abs(corr_Z2_ZA2 - corr_X2_A2)

        # 4. 最终损失 = -DCCA(最大化) + alpha * CIP(最小化)
        loss = -dcca_corr + self.alpha * (zeta_1 + zeta_2)

        return loss