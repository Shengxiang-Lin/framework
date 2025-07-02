import torch

def recall(scores, labels, k):
    """
    计算Top-K召回率
    :param scores: 模型预测的分数矩阵 [batch_size, num_items]
    :param labels: 真实标签矩阵 [batch_size, num_items]
    :param k: 前K个推荐项
    :return: 平均召回率
    """
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    rank = (-scores).argsort(dim=1)  # 对分数降序排列获取索引（分数越高排名越前）
    cut = rank[:, :k]  # 截取每个样本的前k个推荐项索引
    hit = labels.gather(1, cut)  # 在标签中收集前k个位置的物品（1表示相关，0表示不相关）
     # 计算每个用户的召回率
    recall_per_user = hit.sum(1).float() / labels.sum(1).float()
    recall_per_user[labels.sum(1) == 0] = float('nan')
    return recall_per_user.mean().item()

def ndcg(scores, labels, k):
    """
    计算归一化折损累计增益(NDCG)
    :param scores: 模型预测的分数矩阵 [batch_size, num_items]
    :param labels: 真实标签矩阵 [batch_size, num_items]
    :param k: 前K个推荐项
    :return: 平均NDCG
    """
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    rank = (-scores).argsort(dim=1)  # 对分数降序排列获取索引
    cut = rank[:, :k]  # 截取每个样本的前k个推荐项索引
    hits = labels.gather(1, cut)  # 获取前k个位置的物品相关性（0/1）
    position = torch.arange(2, 2+k)  # 生成位置序列[2,3,...,k+1]
    weights = 1 / torch.log2(position.float())  # 计算DCG权重（位置增益）
    dcg = (hits.float() * weights).sum(1)  # 计算每个用户的DCG（折损累计增益）
    # 计算理想DCG（IDCG）：对每个用户取其前min(相关物品数,k)个位置的理想增益
    idcg = torch.zeros_like(dcg)
    for i, n in enumerate(labels.sum(1)):
        n = min(n.item(), k)
        if n > 0:
            idcg[i] = weights[:n].sum()
    ndcg_val = dcg / idcg  # 计算每个用户的NDCG（归一化折损累计增益）
    ndcg_val[idcg == 0] = 0.0
    return ndcg_val.mean().item()  # 返回所有用户的平均NDCG

def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """
    批量计算多个K值的召回率和NDCG指标
    :param scores: 模型预测的分数矩阵 [batch_size, num_items]
    :param labels: 真实标签矩阵 [batch_size, num_items]
    :param ks: K值列表，如[5, 10, 20]
    :return: 包含各指标值的字典
    """
    metrics = {}  # 初始化指标字典
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    answer_count = labels.sum(1)  # 计算每个用户的相关物品总数
    answer_count_float = answer_count.float()  # 转换为浮点数类型
    labels_float = labels.float()  # 标签转换为浮点数类型
    rank = (-scores).argsort(dim=1)  # 对分数降序排列获取索引
    cut = rank  # 初始化截取索引（后续动态裁剪）
    # 按k值从大到小遍历（优化：复用已计算的rank）
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]  # 截取当前k值对应的前k个索引
        hits = labels_float.gather(1, cut)  # 获取前k个位置的物品相关性
        # 计算并存储召回率@k（相关命中数/总相关数）
        recall_per_user = hits.sum(1) / answer_count_float
        recall_per_user[answer_count == 0] = float('nan')
        metrics['Recall@%d' % k] = recall_per_user.mean().item()
        
        position = torch.arange(2, 2+k)  # 生成位置序列
        weights = 1 / torch.log2(position.float())  # 计算DCG权重
        dcg = (hits * weights).sum(1)  # 计算DCG
        # 计算IDCG（考虑用户实际相关物品数n）
        idcg = torch.zeros_like(dcg)
        for i, n in enumerate(answer_count):
            n = min(n.item(), k)
            if n > 0:
                idcg[i] = weights[:n].sum()
        ndcg_val = dcg / idcg
        ndcg_val[idcg == 0] = 0.0
        metrics['NDCG@%d' % k] = ndcg_val.mean().item()  # 计算平均NDCG
    
    return metrics  # 返回包含所有指标的字典

def split_at_index(dim, index, t):
    """
    在指定维度分割张量
    :param dim: 要分割的维度
    :param index: 分割点的索引
    :param t: 输入张量
    :return: 分割后的两个张量
    """
    pre_slices = (slice(None),) * dim  # 生成前置切片元组（如dim=2时：(slice(None), slice(None))）
    l = (*pre_slices, slice(None, index))  # 构建左分块切片（0到index-1）
    r = (*pre_slices, slice(index, None))  # 构建右分块切片（index到末尾）
    return t[l], t[r]  # 返回按指定维度在index处分割的两个张量