import numpy as np
import json
import csv
import ast
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import config
from clustering_MI import GSOM_MI, calculate_cluster_weights

# 全局缓存：仅当不使用预计算文件时生效
USER_WEIGHTED_VECTORS_CACHE = {}

# ------------------------------
# 批次填充collate函数
# ------------------------------
def collate_user_sequences(batch):
    user_ids, weighted_seqs, original_seqs = zip(*batch)

    max_seq_len = max(len(seq) for seq in weighted_seqs)
    vec_dim = weighted_seqs[0].shape[1]
    device = weighted_seqs[0].device

    padded_weighted = []
    padded_original = []
    seq_masks = []

    for w_seq, o_seq in zip(weighted_seqs, original_seqs):
        seq_len = len(w_seq)
        pad_len = max_seq_len - seq_len

        w_padded = torch.cat([w_seq, torch.zeros(pad_len, vec_dim, device=device)], dim=0)
        o_padded = torch.cat([o_seq, torch.zeros(pad_len, vec_dim, device=device)], dim=0)
        mask = torch.cat([torch.ones(seq_len, device=device), torch.zeros(pad_len, device=device)], dim=0)

        padded_weighted.append(w_padded)
        padded_original.append(o_padded)
        seq_masks.append(mask)

    return {
        "user_ids": torch.tensor(user_ids, device=device),
        "weighted_vectors": torch.stack(padded_weighted),
        "original_vectors": torch.stack(padded_original),
        "masks": torch.stack(seq_masks)
    }


# ------------------------------
# 跨域对比学习数据集
# ------------------------------
class CrossDomainDataset(Dataset):
    def __init__(self, book_vectors, movie_vectors, pairs_path):
        self.book_vectors = book_vectors
        self.movie_vectors = movie_vectors
        self.num_pos = config.CONTRASTIVE_POSITIVES
        self.num_neg = config.CONTRASTIVE_NEGATIVES
        self.pairs = []

        with open(pairs_path, 'r', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_idx = int(row['movie_idx'])
                pos_idxs = ast.literal_eval(row['positive_indices'])[:self.num_pos]
                neg_idxs = ast.literal_eval(row['negative_indices'])[:self.num_neg]

                if len(pos_idxs) < self.num_pos:
                    print(f"Warning: item {source_idx} pos sample missing, got {len(pos_idxs)} vs need {self.num_pos}")
                if len(neg_idxs) < self.num_neg:
                    print(f"Warning: item {source_idx} neg sample missing, got {len(neg_idxs)} vs need {self.num_neg}")
                self.pairs.append((source_idx, pos_idxs, neg_idxs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_idx, pos_idxs, neg_idxs = self.pairs[idx]
        src_vec = torch.tensor(self.book_vectors[source_idx], dtype=torch.float32)
        pos_vecs = torch.tensor(self.movie_vectors[pos_idxs], dtype=torch.float32)
        neg_vecs = torch.tensor(self.movie_vectors[neg_idxs], dtype=torch.float32)
        return src_vec, pos_vecs, neg_vecs


# ------------------------------
# 用户交互序列数据集（统一训练/验证集）
# ------------------------------
class InteractionDataset(Dataset):
    def __init__(self, interactions, item_vectors, cache_key, precomputed_path=None):
        self.interactions = interactions
        self.item_vectors = item_vectors
        self.user_ids = list(interactions.keys())
        self.cache_key = cache_key
        self.precomputed_path = precomputed_path
        self.precomputed_data = dict()

        # 预计算文件优先加载
        if self.precomputed_path is not None:
            self._load_precomputed()
        else:
            if self.cache_key not in USER_WEIGHTED_VECTORS_CACHE:
                USER_WEIGHTED_VECTORS_CACHE[self.cache_key] = dict()
            self._init_memory_cache()

    def __len__(self):
        return len(self.user_ids)

    def _load_precomputed(self):
        print(f"Load precomputed vec from: {self.precomputed_path}")
        data = np.load(self.precomputed_path, allow_pickle=True)
        for uid in self.user_ids:
            key = str(uid)
            if key not in data:
                raise KeyError(f"User {uid} missing in precomputed npz")
            weighted = torch.tensor(data[key][0], dtype=torch.float32, device=config.DEVICE)
            original = torch.tensor(data[key][1], dtype=torch.float32, device=config.DEVICE)
            self.precomputed_data[uid] = (weighted, original)

    def _init_memory_cache(self):
        cache = USER_WEIGHTED_VECTORS_CACHE[self.cache_key]
        dev = config.DEVICE
        for uid in self.user_ids:
            item_ids = self.interactions[uid]
            orig_vec = torch.tensor([self.item_vectors[i] for i in item_ids], dtype=torch.float32, device=dev)
            cache[uid] = (orig_vec, orig_vec)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        if self.precomputed_path is not None:
            w, o = self.precomputed_data[uid]
        else:
            w, o = USER_WEIGHTED_VECTORS_CACHE[self.cache_key][uid]
        return uid, w, o


# ------------------------------
# 数据加载工具函数
# ------------------------------
def load_vectors(path: str):
    return np.load(path)

def load_interactions(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        raw = json.load(f)
    processed = dict()
    for uid_str, item_list in raw.items():
        try:
            uid = int(uid_str.split("_")[-1])
            processed[uid] = item_list
        except Exception as e:
            print(f"Skipped invalid user id {uid_str}: {e}")
    return processed


# ------------------------------
# 统一构建全部DataLoader
# 核心逻辑：source_interactions_5 切分8:2，80%训练，20%验证
# ------------------------------
def get_data_loaders(precomputed_paths=None):
    precomputed_paths = precomputed_paths or dict()
    rng = np.random.default_rng(seed=42)

    # 1. 加载全局物品向量
    book_vecs = load_vectors(f"{config.DATA_DIR}/{config.BOOK_VECTORS}")
    movie_vecs = load_vectors(f"{config.DATA_DIR}/{config.MOVIE_VECTORS}")

    # 2. 加载全部交互数据
    src_inter_5 = load_interactions(f"{config.DATA_DIR}/{config.SOURCE_INTERACTIONS_5}")
    src_inter_13 = load_interactions(f"{config.DATA_DIR}/{config.SOURCE_INTERACTIONS_13}")
    tgt_inter = load_interactions(f"{config.DATA_DIR}/{config.TARGET_INTERACTIONS}")

    # --------------------------
    # 关键：源域source_inter_5 划分 80%训练 / 20%验证
    # --------------------------
    all_src_users = list(src_inter_5.keys())
    val_num = int(len(all_src_users) * 0.2)
    val_user_list = rng.choice(all_src_users, size=val_num, replace=False).tolist()
    train_user_list = [u for u in all_src_users if u not in val_user_list]

    train_src_inter5 = {u: src_inter_5[u] for u in train_user_list}
    val_src_inter5 = {u: src_inter_5[u] for u in val_user_list}

    # 3. 跨域对比学习 loader
    cross_dataset = CrossDomainDataset(
        book_vecs, movie_vecs, f"{config.DATA_DIR}/{config.CROSS_DOMAIN_PAIRS}"
    )
    cross_loader = DataLoader(cross_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 4. 阶段1训练集（80%源域用户）
    stage1_train_ds = InteractionDataset(
        interactions=train_src_inter5,
        item_vectors=book_vecs,
        cache_key="stage1_train",
        precomputed_path=precomputed_paths.get("stage1")
    )
    stage1_loader = DataLoader(
        stage1_train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_user_sequences
    )

    # 5. 阶段1验证集（切分出的20%源域用户）
    val_ds = InteractionDataset(
        interactions=val_src_inter5,
        item_vectors=book_vecs,
        cache_key="stage1_val",
        precomputed_path=precomputed_paths.get("stage1")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_user_sequences
    )

    # 6. 元学习阶段maml（完整source_inter_13不切分）
    maml_ds = InteractionDataset(
        src_inter_13, book_vecs,
        cache_key="maml",
        precomputed_path=precomputed_paths.get("maml")
    )
    maml_loader = DataLoader(
        maml_ds,
        batch_size=config.META_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_user_sequences
    )

    # 7. 微调finetune（同阶段1训练集80%）
    finetune_ds = InteractionDataset(
        train_src_inter5, book_vecs,
        cache_key="finetune",
        precomputed_path=precomputed_paths.get("finetune")
    )
    finetune_loader = DataLoader(
        finetune_ds,
        batch_size=config.META_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_user_sequences
    )

    # 整合所有返回数据，变量无缺失、无未定义
    return {
        "cross_domain_loader": cross_loader,
        "stage1_loader": stage1_loader,
        "maml_loader": maml_loader,
        "finetune_loader": finetune_loader,
        "val_loader": val_loader,
        "train_source_inter5": train_src_inter5,
        "val_source_inter5": val_src_inter5,
        "source_inter13": src_inter_13,
        "target_interactions": tgt_inter,
        "book_vectors": book_vecs,
        "movie_vectors": movie_vecs
    }
