import torch.optim as optim


def create_mamba_optimizer(model, lr=3e-4, weight_decay=1e-2):
    """
    为包含 Mamba 模块的模型创建分组优化的 AdamW 优化器。
    核心目的：将 Mamba 中对数值极其敏感的参数从权重衰减中“豁免”出来。
    """
    decay_parameters = []
    no_decay_parameters = []

    # 🌟 Mamba 内部对 Weight Decay 极度敏感的参数名关键字
    no_decay_keywords = [
        'bias',  # 所有的偏置项 (常规操作)
        'norm',  # 所有的归一化层权重 (LayerNorm, RMSNorm等)
        'dt_proj',  # SSM 时间步长 dt 的投影层 (极其敏感，控制时间离散化)
        'A_log',  # SSM 状态转移矩阵 A (决定了记忆的持久度，千万不能加衰减)
        'D',  # SSM 跨步连接权重 (Skip connection)
        'conv1d',  # 局部 1D 卷积的权重和偏置
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判断逻辑：如果是 1 维张量（通常是 bias 或 norm），或者名字中包含敏感关键字
        if param.ndim <= 1 or any(nd in name for nd in no_decay_keywords):
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)

    # 将参数分为两组送入优化器
    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,  # 🛑 核心：敏感参数完全豁免权重衰减
        },
    ]

    # 使用 AdamW 优化器
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # 打印一下分组情况，让你心里有数
    print(f"🔧 优化器配置完成:")
    print(f"   - 参与衰减的参数张量数量: {len(decay_parameters)}")
    print(f"   - 豁免衰减的参数张量数量 (Mamba核心/Bias/Norm): {len(no_decay_parameters)}")

    return optimizer