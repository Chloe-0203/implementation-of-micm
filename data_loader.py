import numpy as np
import json
import csv
import ast  # 用于解析字符串形式的列表
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import config
from clustering_cosine import GSOM_Cosine, calculate_cluster_weights

# 全局缓存字典
USER_WEIGHTED_VECTORS_CACHE = {}
CLUSTER_MODEL = GSOM_Cosine(config.CLUSTER_CONFIG)  # 全局聚类模型实例，避免重复初始化


# ------------------------------
# 新增：自定义批次处理函数（填充对齐）
# ------------------------------
def collate_user_sequences(batch):
    """
    处理用户交互序列的批次，用0填充使长度一致
    batch格式：[(user_id1, weighted_vecs1, original_vecs1), (user_id2, ...), ...]
    """
    user_ids, weighted_seqs, original_seqs = zip(*batch)  # 解包批次数据

    # 1. 确定批次中最长的序列长度
    max_seq_len = max(len(seq) for seq in weighted_seqs)
    vec_dim = weighted_seqs[0].shape[1]  # 向量维度（如384）
    device = weighted_seqs[0].device  # 保持与原向量相同的设备

    # 2. 对每个序列进行填充
    padded_weighted = []
    padded_original = []
    seq_masks = []  # 掩码：1表示有效元素，0表示填充元素

    for w_seq, o_seq in zip(weighted_seqs, original_seqs):
        seq_len = len(w_seq)
        pad_len = max_seq_len - seq_len

        # 填充加权向量
        w_padded = torch.cat([w_seq, torch.zeros(pad_len, vec_dim, device=device)], dim=0)
        # 填充原始向量
        o_padded = torch.cat([o_seq, torch.zeros(pad_len, vec_dim, device=device)], dim=0)
        # 创建掩码
        mask = torch.cat([torch.ones(seq_len, device=device), torch.zeros(pad_len, device=device)], dim=0)

        padded_weighted.append(w_padded)
        padded_original.append(o_padded)
        seq_masks.append(mask)

    # 3. 堆叠成批次张量
    return {
        "user_ids": torch.tensor(user_ids, device=device),
        "weighted_vectors": torch.stack(padded_weighted),
        "original_vectors": torch.stack(padded_original),
        "masks": torch.stack(seq_masks)
    }


class CrossDomainDataset(Dataset):
    """跨域对比学习数据集（支持从config读取样本数量）"""

    def __init__(self, book_vectors, movie_vectors, pairs_path):
        self.book_vectors = book_vectors
        self.movie_vectors = movie_vectors
        self.pairs = []
        # 从config获取正负样本数量（新增）
        self.num_pos = config.CONTRASTIVE_POSITIVES  # 需要在config中新增该参数
        self.num_neg = config.CONTRASTIVE_NEGATIVES  # 需要在config中新增该参数

        # 读取正负样本对（处理新格式）
        with open(pairs_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_idx = int(row['movie_idx'])

                # 解析正样本列表并截断/补齐到config指定数量
                pos_idxs = ast.literal_eval(row['positive_indices'])
                # 取前num_pos个正样本（若不足则全部保留，实际应用中可能需要补齐策略）
                pos_idxs = pos_idxs[:self.num_pos]
                # 校验最终数量（可选，根据需求决定是否严格校验）
                if len(pos_idxs) < self.num_pos:
                    # 这里简单警告，也可根据需求抛出异常或用随机样本补齐
                    print(f"Warning: book {source_idx} has only {len(pos_idxs)} positive samples (need {self.num_pos})")

                # 解析负样本列表并截断/补齐到config指定数量
                neg_idxs = ast.literal_eval(row['negative_indices'])
                neg_idxs = neg_idxs[:self.num_neg]
                if len(neg_idxs) < self.num_neg:
                    print(f"Warning: book {source_idx} has only {len(neg_idxs)} negative samples (need {self.num_neg})")

                self.pairs.append((source_idx, pos_idxs, neg_idxs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_idx, pos_idxs, neg_idxs = self.pairs[idx]
        source_vec = self.book_vectors[source_idx]
        pos_vecs = self.movie_vectors[pos_idxs]  # 数量由config控制
        neg_vecs = self.movie_vectors[neg_idxs]  # 数量由config控制

        return (
            torch.tensor(source_vec, dtype=torch.float32),
            torch.tensor(pos_vecs, dtype=torch.float32),
            torch.tensor(neg_vecs, dtype=torch.float32)
        )


# 修改 __init__ 方法，增加预计算文件路径参数（提前预处理加权向量，不用每次训练都计算）
class InteractionDataset(Dataset):
    """用户-物品交互数据集（支持预加载加权向量）"""
    def __init__(self, interactions, item_vectors, is_source=True, cache_key="default", precomputed_path=None):
        self.interactions = interactions
        self.item_vectors = item_vectors
        self.user_ids = list(interactions.keys())
        self.is_source = is_source
        self.cache_key = cache_key
        self.epoch = 0
        self.cache_update_interval = config.CACHE_UPDATE_INTERVAL
        self.precomputed_path = precomputed_path  # 新增：预计算文件路径
        self.precomputed_data = {}  # 新增：存储预加载的加权向量

        # 加载预计算数据（如果提供）
        if self.precomputed_path:
            self._load_precomputed()
        else:
            # 保持原有缓存逻辑
            if self.cache_key not in USER_WEIGHTED_VECTORS_CACHE:
                USER_WEIGHTED_VECTORS_CACHE[self.cache_key] = {}
            self._initialize_cache()

    # 添加__len__方法
    def __len__(self):
        """返回数据集中的用户数量"""
        return len(self.user_ids)


    def _load_precomputed(self):
        """加载预计算的加权向量（假设为npz格式）"""
        print(f"Loading precomputed weighted vectors from {self.precomputed_path}")
        data = np.load(self.precomputed_path, allow_pickle=True)
        for user_id in self.user_ids:
            if str(user_id) in data:
                # 格式：(weighted_vecs, original_vecs, mask)
                weighted = torch.tensor(data[str(user_id)][0], dtype=torch.float32, device=config.DEVICE)
                original = torch.tensor(data[str(user_id)][1], dtype=torch.float32, device=config.DEVICE)
                self.precomputed_data[user_id] = (weighted, original)
            else:
                raise ValueError(f"User {user_id} not found in precomputed data")

    def _initialize_cache(self):
        """初始化用户加权向量缓存（使用物品原始向量）"""
        for user_id in self.user_ids:
            interactions = self.interactions[user_id]
            # 获取用户交互的所有物品原始向量
            item_vecs = [self.item_vectors[item_id] for item_id in interactions]
            # 转换为张量（保持原始向量）
            original_vecs = torch.tensor(item_vecs, dtype=torch.float32, device=config.DEVICE)
            # 由于你不需要加权，weighted_vecs 可直接等于原始向量
            weighted_vecs = original_vecs  # 无需 clone，直接引用原始向量
            # 存入缓存
            USER_WEIGHTED_VECTORS_CACHE[self.cache_key][user_id] = (weighted_vecs, original_vecs)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        if self.precomputed_path:
            # 从预计算数据获取
            return user_id, self.precomputed_data[user_id][0], self.precomputed_data[user_id][1]
        else:
            # 原有缓存逻辑
            weighted_vecs, original_vecs = USER_WEIGHTED_VECTORS_CACHE[self.cache_key][user_id]
            return user_id, weighted_vecs, original_vecs

    # 移除缓存更新相关方法（如果完全依赖预计算）
    def set_epoch(self, epoch):
        self.epoch = epoch
        # 若使用预计算数据，不更新缓存
        if not self.precomputed_path and epoch % self.cache_update_interval == 0 and epoch > 0:
            self._update_cache()

def load_vectors(path):
    """加载物品向量"""
    return np.load(path)


def load_interactions(path):
    """加载用户交互数据"""
    # with open(path, 'r') as f:
    #     return json.load(f)
    """加载用户交互数据，将用户ID转为整数"""
    with open(path, 'r') as f:
        interactions = json.load(f)

     # 处理带前缀的用户ID（如"user_474" -> 474）
    processed = {}
    for user_id_str, items in interactions.items():
        # 提取数字部分（假设格式固定为"user_数字"）
        try:
            # 分割字符串并取最后一部分转换为整数
            user_id = int(user_id_str.split("_")[-1])
            processed[user_id] = items
        except (ValueError, IndexError):
            # 处理异常格式（可选：跳过或记录日志）
            print(f"警告：无效的用户ID格式 '{user_id_str}'，已跳过")

    return processed

class ValInteractionDataset(Dataset):
    """验证集数据集（复用训练集的部分数据）验证集所加"""
    def __init__(self, user_ids, precomputed_data):
        self.user_ids = user_ids
        self.precomputed_data = precomputed_data  # 复用训练集的预处理数据

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return user_id, self.precomputed_data[user_id][0], self.precomputed_data[user_id][1]

def get_data_loaders(precomputed_paths=None):
    """获取所有数据加载器（带缓存支持）"""
    precomputed_paths = precomputed_paths or {}

    # 加载向量
    book_vectors = load_vectors(f"{config.DATA_DIR}/{config.BOOK_VECTORS}")
    movie_vectors = load_vectors(f"{config.DATA_DIR}/{config.MOVIE_VECTORS}")

    # 加载交互数据
    source_interactions_5 = load_interactions(f"{config.DATA_DIR}/{config.SOURCE_INTERACTIONS_5}")
    source_interactions_13 = load_interactions(f"{config.DATA_DIR}/{config.SOURCE_INTERACTIONS_13}")
    target_interactions = load_interactions(f"{config.DATA_DIR}/{config.TARGET_INTERACTIONS}")

    # 跨域对比学习数据集
    cross_domain_dataset = CrossDomainDataset(
        book_vectors,
        movie_vectors,
        f"{config.DATA_DIR}/{config.CROSS_DOMAIN_PAIRS}"
    )
    cross_domain_loader = DataLoader(
        cross_domain_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )


    # 源域交互数据集（阶段1）- 使用不同的缓存键
    stage1_dataset = InteractionDataset(
        source_interactions_5,
        book_vectors,
        is_source=True,
        cache_key="stage1",
        # 预处理文件路径
        precomputed_path=precomputed_paths.get("stage1")
    )
    stage1_loader = DataLoader(
        stage1_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn = collate_user_sequences  # 应用填充函数
    )

    # 元学习数据集（阶段2）
    maml_dataset = InteractionDataset(
        source_interactions_13,
        book_vectors,
        is_source=True,
        cache_key="maml",
        # 预处理文件路径
        precomputed_path=precomputed_paths.get("maml")
    )
    maml_loader = DataLoader(
        maml_dataset,
        batch_size=config.META_BATCH_SIZE,  # 元学习每个任务n个用户
        shuffle=True,
        collate_fn = collate_user_sequences  # 应用填充函数
    )

    # 微调数据集（阶段3）
    finetune_dataset = InteractionDataset(
        source_interactions_5,
        book_vectors,
        is_source=True,
        cache_key="finetune",
        # 预处理文件路径
        precomputed_path=precomputed_paths.get("finetune")
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=config.META_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_user_sequences  # 应用填充函数
    )

    # 验证集模块添加（开始）
    # 从目标域数据中划分20%用户作为验证集（测试时仍用完整数据）
    target_users = list(target_interactions.keys())
    np.random.seed(42)  # 固定随机种子确保划分稳定
    val_size = int(0.4 * len(target_users))
    val_users = np.random.choice(target_users, val_size, replace=False)
    val_interactions = {u: target_interactions[u] for u in val_users}

    # # 修改后：从源域交互数据划分验证集（以阶段1使用的源域数据为例）
    # source_users = list(source_interactions_5.keys())  # 源域交互数据（阶段1的源数据）
    # np.random.seed(42)  # 固定随机种子确保划分稳定
    # val_size = int(0.2 * len(source_users))  # 取源域用户的20%作为验证集
    # val_users = np.random.choice(source_users, val_size, replace=False)
    # val_interactions = {u: source_interactions_5[u] for u in val_users}  # 验证集交互数据来自源域

    # 构建验证集数据加载器（复用目标域用户的预处理数据）
    # 注意：验证集仅用于评估，不参与训练
    val_dataset = ValInteractionDataset(
        user_ids=val_users,
        precomputed_data={}  # 动态填充
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_user_sequences
    )
    # 验证集模块添加（结束）

    return {
        "cross_domain": cross_domain_loader,
        "stage1": stage1_loader,
        "maml": maml_loader,
        "finetune": finetune_loader,
        "target_interactions": target_interactions,
        # 验证集模块添加（开始）
        "val_interactions": val_interactions,  # 20%验证集
        "val_loader": val_loader,
        # 验证集模块添加（结束）
        "movie_vectors": movie_vectors,
        "book_vectors": book_vectors
    }
