# import torch
# import numpy as np
# from config import config
#
#
# # def random_neg_samples(all_items, pos_items, num_neg):
# #     """随机采样负样本"""
# #     neg_samples = []
# #     while len(neg_samples) < num_neg:
# #         neg = np.random.choice(len(all_items))
# #         if neg not in pos_items:
# #             neg_samples.append(neg)
# #     return torch.tensor([all_items[i] for i in neg_samples], dtype=torch.float32, device=config.DEVICE)
# def random_neg_samples(all_items, pos_items, num_neg):
#     """随机采样负样本"""
#     neg_samples = []
#     while len(neg_samples) < num_neg:
#         neg = np.random.choice(len(all_items))
#         if neg not in pos_items:
#             neg_samples.append(neg)
#
#     # 先将列表转换为单个numpy数组，再转换为张量（解决警告）
#     neg_arr = np.array([all_items[i] for i in neg_samples], dtype=np.float32)
#     return torch.tensor(neg_arr, device=config.DEVICE)
#
#
# def hit_rate(predicted, ground_truth, k=100):
#     """命中率@k"""
#     predicted = predicted[:k]
#     return 100*len(set(predicted) & set(ground_truth)) / len(ground_truth) if ground_truth else 0.0
#
#
# def ndcg_score(predicted, ground_truth, k=100):
#     """归一化折扣累积增益@k"""
#     predicted = predicted[:k]
#     if not ground_truth:
#         return 0.0
#
#     # 计算DCG
#     dcg = 0.0
#     for i, item in enumerate(predicted):
#         if item in ground_truth:
#             dcg += 1.0 / np.log2(i + 2)  # i从0开始
#
#     # 计算IDCG
#     idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
#     return 100*dcg / idcg if idcg > 0 else 0.0
#
#
# def evaluate_performance(model, user_vecs, movie_vectors, target_interactions, k=100):
#     """评估推荐性能（兼容有/无模型的场景）"""
#     hit_rates = []
#     ndcgs = []
#     device = config.DEVICE
#
#     # 处理电影向量：支持字典或张量格式，并根据是否有模型进行编码
#     if isinstance(movie_vectors, dict):
#         movie_ids_list = list(movie_vectors.keys())
#         movie_vecs = torch.stack([movie_vectors[mid] for mid in movie_ids_list]).to(device)
#     else:
#         # movie_ids_list = [f"movie_{i}" for i in range(len(movie_vectors))]  # 示例ID映射
#         movie_ids_list = [i for i in range(len(movie_vectors))]  # 索引i对应电影ID i（整数格式）
#         movie_vecs = movie_vectors.to(device)
#
#     # 若有模型，使用模型编码电影向量；否则直接使用原始向量
#     if model is not None:
#         movie_enc = model.encode_item(movie_vecs)
#     else:
#         movie_enc = movie_vecs  # 直接使用原始向量
#
#     for user_id, ground_truth in target_interactions.items():
#         if user_id not in user_vecs:
#             continue
#         user_vec = user_vecs[user_id].to(device).squeeze()  # 确保用户向量维度正确
#
#         # 计算用户与所有电影的相似度（内积）
#         scores = torch.matmul(movie_enc, user_vec)  # [num_movies]
#         _, top_indices = torch.topk(scores, k)
#         top_indices = top_indices.cpu().numpy()
#
#         # 映射回电影ID（根据原始格式）
#         predicted = [movie_ids_list[i] for i in top_indices]
#
#         # 计算指标
#         hr = hit_rate(predicted, ground_truth, k)
#         ndcg = ndcg_score(predicted, ground_truth, k)
#         hit_rates.append(hr)
#         ndcgs.append(ndcg)
#
#     return {
#         "mean_hit_rate": np.mean(hit_rates),
#         "mean_ndcg": np.mean(ndcgs)
#     }

# 与Tiger保持一致
import torch
import numpy as np
from config import config
from tqdm import tqdm


def random_neg_samples(all_items, pos_items, num_neg):
    """随机采样负样本"""
    neg_samples = []
    while len(neg_samples) < num_neg:
        neg = np.random.choice(len(all_items))
        if neg not in pos_items:
            neg_samples.append(neg)

    neg_arr = np.array([all_items[i] for i in neg_samples], dtype=np.float32)
    return torch.tensor(neg_arr, device=config.DEVICE)


def hit_rate(ranks, k=100):
    """命中率@k（与Tiger逻辑一致）"""
    return 300 * (ranks < k).float().mean().item()


def ndcg_score(ranks, k=100):
    """归一化折扣扣累积增益@k（与Tiger逻辑一致）"""
    mask = ranks < k
    if not mask.any():
        return 0.0

    dcg = (1.0 / torch.log2((ranks[mask] + 2))).sum().item()
    num_relevant = mask.sum().item()
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, num_relevant)))

    return 100 * dcg / idcg if idcg > 0 else 0.0


def evaluate_performance(model, user_vecs, movie_vectors, target_interactions, k=100,progress_bar=False):
    """评估推荐性能（移除除屏蔽目标域正样本）"""
    hit_rates = []
    ndcgs = []
    device = config.DEVICE

    # # 处理电影向量
    # if isinstance(movie_vectors, dict):
    #     movie_ids_list = list(movie_vectors.keys())
    #     movie_vecs = torch.stack([movie_vectors[mid] for mid in movie_ids_list]).to(device)
    # else:
    #     movie_ids_list = [i for i in range(len(movie_vectors))]
    #     movie_vecs = movie_vectors.to(device)
    # 处理电影向量：自动转换为PyTorch张量并移动到设备
    if isinstance(movie_vectors, np.ndarray):
        # 若为NumPy数组，先转为张量
        movie_vecs = torch.tensor(movie_vectors, dtype=torch.float32).to(device)
        movie_ids_list = [i for i in range(len(movie_vectors))]
    elif isinstance(movie_vectors, dict):
        # 若为字典（键为ID，值为张量）
        movie_ids_list = list(movie_vectors.keys())
        movie_vecs = torch.stack([movie_vectors[mid] for mid in movie_ids_list]).to(device)
    elif isinstance(movie_vectors, torch.Tensor):
        # 若已为张量，直接移动到设备
        movie_vecs = movie_vectors.to(device)
        movie_ids_list = [i for i in range(movie_vectors.shape[0])]
    else:
        raise TypeError(f"不支持的movie_vectors类型: {type(movie_vectors)}")

    # 编码电影向量（如有模型）
    if model is not None:
        movie_enc = model.encode_item(movie_vecs)
    else:
        movie_enc = movie_vecs

    # 准备用户迭代器，支持进度条
    user_iter = target_interactions.items()
    if progress_bar:
        user_iter = tqdm(user_iter, desc="Evaluating users", total=len(target_interactions))

    # 遍历用户评估
    for user_id, ground_truth in user_iter:
        if user_id not in user_vecs:
            continue
        user_vec = user_vecs[user_id].to(device).squeeze()

        # 计算用户与所有电影的相似度（内积）
        scores = torch.matmul(movie_enc, user_vec)  # [num_movies]

        # 【核心修改】删除正样本屏蔽屏蔽逻辑
        # 原逻辑：scores[pos_indices] = -np.inf

        # 计算排名（从0开始，值越小排名名越靠前）
        _, sorted_indices = torch.sort(-scores)
        ranks = torch.zeros(len(scores), device=device, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(scores), device=device)

        # 提取真实样本的排名
        ground_truth_indices = [movie_ids_list.index(item) for item in ground_truth if item in movie_ids_list]
        if not ground_truth_indices:
            continue
        gt_ranks = ranks[ground_truth_indices]

        # 计算指标
        hr = hit_rate(gt_ranks, k)
        ndcg = ndcg_score(gt_ranks, k)
        hit_rates.append(hr)
        ndcgs.append(ndcg)

    return {
        "mean_hit_rate": np.mean(hit_rates) if hit_rates else 0.0,
        "mean_ndcg": np.mean(ndcgs) if ndcgs else 0.0
    }