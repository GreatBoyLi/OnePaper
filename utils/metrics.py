import torch


def evaluate_metrics(preds, targets):
    """
    根据论文的公式 (27)-(30) 计算光伏预测的四个评估指标

    参数:
        preds: 模型的预测值张量 (Tensor)，形状通常为 (Batch_Size, k) 或打平的 1D 张量
        targets: 真实的光伏功率张量 (Tensor)，形状必须与 preds 相同

    返回:
        包含 RMSE, MAE, MAPE(%), R(%) 的字典
    """
    # 确保输入是 PyTorch 张量，并且维度一致
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    preds = preds.float()
    targets = targets.float()

    # 获取总元素个数 n*k (即公式分母中的 nk)
    nk = targets.numel()

    # ==========================================
    # 1. 公式 (27): RMSE (论文中符号写为 MSE，但公式带了根号，实际是均方根误差)
    # ==========================================
    # 公式: sqrt( 1/nk * sum((y - y_hat)^2) )
    rmse = torch.sqrt(torch.sum((targets - preds) ** 2) / nk)

    # ==========================================
    # 2. 公式 (28): MAE (平均绝对误差)
    # ==========================================
    # 公式: 1/nk * sum(|y - y_hat|)
    mae = torch.sum(torch.abs(targets - preds)) / nk

    # ==========================================
    # 3. 公式 (29): MAPE (%) (平均绝对百分比误差)
    # ==========================================
    # 论文中的指示函数 I(y >= 0.01) 要求只计算真实值 >= 0.01 的点
    mask = targets >= 0.01

    # 初始化一个与 targets 形状相同的全 0 张量
    mape_errors = torch.zeros_like(targets)

    # 只在 mask 为 True (即 target >= 0.01) 的位置计算百分比误差
    mape_errors[mask] = torch.abs((preds[mask] - targets[mask]) / targets[mask])

    # 按照公式：将符合条件的误差求和，除以总数 nk，最后乘以 100 变成百分比
    mape = (100.0 / nk) * torch.sum(mape_errors)

    # ==========================================
    # 4. 公式 (30): R (%) (皮尔逊相关系数)
    # ==========================================
    # 计算全局的均值 <y> 和 <y_hat>
    mean_targets = torch.mean(targets)
    mean_preds = torch.mean(preds)

    # 去均值中心化: (y - <y>) 和 (y_hat - <y_hat>)
    targets_centered = targets - mean_targets
    preds_centered = preds - mean_preds

    # 分子部分: 交叉相乘之和.
    numerator = torch.sum(targets_centered * preds_centered)

    # 分母部分: 各自平方和的平方根的乘积
    denominator = torch.sqrt(torch.sum(targets_centered ** 2)) * torch.sqrt(torch.sum(preds_centered ** 2))

    # 防止分母为 0 导致报错 (虽然在真实数据中几乎不可能发生)
    if denominator == 0:
        r = torch.tensor(0.0)
    else:
        # 公式乘以 100 将其转化为百分比格式
        r = (100.0 * numerator) / denominator

    # 将张量数值取出来，方便打印和保存
    return {
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'MAPE(%)': mape.item(),
        'R(%)': r.item()
    }


# ================= 演示测试 =================
if __name__ == "__main__":
    # 模拟 2 个样本，每个样本预测未来 4 个时间步的光伏功率
    dummy_targets = torch.tensor([[0.5, 0.6, 0.005, 0.0],
                                  [0.8, 0.7, 0.6, 0.5]])
    dummy_preds = torch.tensor([[0.45, 0.62, 0.01, 0.05],
                                [0.75, 0.65, 0.65, 0.45]])

    results = evaluate_metrics(dummy_preds, dummy_targets)

    print("模型评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")