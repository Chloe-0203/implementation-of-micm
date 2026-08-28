import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from torch.amp import GradScaler, autocast
from contextlib import contextmanager

from config import config
from models_server import CrossDomainRecommender
from data_loader import collate_user_sequences, get_data_loaders
from utils import evaluate_performance, random_neg_samples


# 新增：设置随机种子的函数
def set_random_seed(seed):
    """设置所有可能的随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时使用
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""序列数据增强函数"""


def augment_sequence(sequence, mask, aug_prob=0.15):
    """
    对序列进行数据增强:
    1. 随机裁剪
    2. 随机替换
    3. 随机mask
    """
    seq_len = mask.sum().int().item()
    if seq_len <= 5:  # 短序列不增强，避免信息丢失过多
        return sequence, mask

    augmented_seq = sequence.clone()
    augmented_mask = mask.clone()

    # 1. 随机裁剪 (20%概率)
    if np.random.random() < 0.15:
        crop_ratio = np.random.uniform(0.6, 0.9)
        crop_len = max(3, int(seq_len * crop_ratio))
        start_idx = np.random.randint(0, seq_len - crop_len + 1)

        # 保留裁剪部分，其余置零
        augmented_seq[start_idx + crop_len:] = 0
        augmented_mask[start_idx + crop_len:] = 0

    # 2. 随机替换 (20%概率替换部分元素)
    replace_mask = torch.rand(seq_len) < 0.20
    replace_indices = torch.where(replace_mask)[0]
    if len(replace_indices) > 0:
        # 从序列中随机选择其他元素进行替换
        for idx in replace_indices:
            random_pos = np.random.randint(0, seq_len)
            while random_pos == idx:  # 确保不是同一个位置
                random_pos = np.random.randint(0, seq_len)
            augmented_seq[idx] = sequence[random_pos]

    # 3. 随机mask (10%概率将元素置零)
    mask_mask = torch.rand(seq_len) < 0.09
    mask_indices = torch.where(mask_mask)[0]
    if len(mask_indices) > 0:
        augmented_seq[mask_indices] = 0

    return augmented_seq, augmented_mask


"""临时替换模型参数的上下文管理器"""


@contextmanager
def _set_params(model, params):
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    try:
        for name, param in model.named_parameters():
            if name in params:
                param.data = params[name].data
        yield
    finally:
        for name, param in model.named_parameters():
            if name in original_params:
                param.data = original_params[name].data


"""通用测试函数：计算目标域全量测试集指标（支持多K值）"""


def test_target_domain(model, data_loaders, stage_name, version="v11"):
    print(f"\n=== {stage_name} - 开始目标域全量测试（K=10/20/50） ===")
    model.eval()
    with torch.no_grad():
        current_user_vecs = {}
        if stage_name == "Stage1":
            loader = data_loaders["stage1"]
        elif stage_name == "MAML":
            loader = data_loaders["maml"]
        elif stage_name == "Finetune":
            current_user_vecs = model.save_user_vectors
        else:
            loader = data_loaders["stage1"]

        if stage_name != "Finetune":
            for batch in tqdm(loader, desc=f"{stage_name} - 生成用户向量"):
                user_ids = batch["user_ids"]
                weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
                masks = batch["masks"].to(config.DEVICE)
                user_vecs = model(weighted_vecs, masks)

                for i, user_id in enumerate(user_ids):
                    uid = user_id.item()
                    current_user_vecs[uid] = user_vecs[i]

        if stage_name == "Finetune":
            movie_encs = model.save_movie_vectors
        else:
            movie_tensor = torch.tensor(
                data_loaders["movie_vectors"],
                dtype=torch.float32,
                device=config.DEVICE
            )
            movie_encs = model.encode_item(movie_tensor).detach()

        # 定义需要测试的K值列表
        k_list = [10, 20, 50]
        all_test_metrics = {}
        # 循环计算每个K值的指标
        for k in k_list:
            test_metrics = evaluate_performance(
                model=None,
                user_vecs=current_user_vecs,
                movie_vectors=movie_encs,
                target_interactions=data_loaders["target_interactions"],
                k=k,
                progress_bar=True if k == 20 else False  # 仅20时显示进度条
            )
            all_test_metrics[k] = test_metrics

    # 输出所有K值的测试结果
    print(f"\n{stage_name} 目标域全量测试结果:")
    for k in k_list:
        metrics = all_test_metrics[k]
        print(f"K={k}: Mean Hit Rate@{k}: {metrics['mean_hit_rate']:.4f}, Mean NDCG@{k}: {metrics['mean_ndcg']:.4f}")
    print(f"=== {stage_name} - 测试结束 ===\n")

    # 保存时存储所有K值的指标
    model_save_path = f"{version}_{stage_name.lower()}_best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "user_vectors": current_user_vecs,
        "movie_vectors": movie_encs,
        "test_metrics": all_test_metrics,  # 存储字典（key=K，value=指标）
        "stage": stage_name
    }, model_save_path)
    print(f"{stage_name} 最优模型已保存至: {model_save_path}\n")

    return all_test_metrics


"""自适应损失权重计算"""


def adaptive_loss_weights(contrast_loss, bpr_loss, contrast_mean, bpr_mean, alpha_t=0.9):
    """
    计算自适应损失权重
    contrast_loss: 当前对比损失
    bpr_loss: 当前BPR损失
    contrast_mean: 对比损失移动平均值
    bpr_mean: BPR损失移动平均值
    alpha_t: 温度参数，控制权重平衡敏感度
    """
    # 计算损失比率
    if bpr_mean == 0 or contrast_mean == 0:
        return 1.0, 1.0

    ratio = (contrast_loss / contrast_mean) / (bpr_loss / bpr_mean + 1e-8)

    # 计算自适应权重（使用softmax-like归一化）
    lambda1 = torch.exp(-alpha_t * ratio)
    lambda2 = torch.exp(-alpha_t / (ratio + 1e-8))

    # 归一化确保权重和为1（可选）
    total = lambda1 + lambda2
    lambda1 = lambda1 / total
    lambda2 = lambda2 / total

    return lambda1, lambda2


"""第一阶段：基础特征学习（带早停机制+阶段末测试）"""


def stage1_training(model, data_loaders, optimizer, version="v11"):
    model.train()
    cross_domain_loader = data_loaders["cross_domain"]
    stage1_loader = data_loaders["stage1"]
    dataset = stage1_loader.dataset
    book_vectors = data_loaders["book_vectors"]
    val_interactions = data_loaders["val_interactions"]
    movie_vectors = data_loaders["movie_vectors"]

    scaler = GradScaler('cuda')
    best_val_hr = 0.0  # 以K=10的HR作为早停标准
    patience = config.EARLY_STOPPING_PATIENCE
    counter = 0
    best_model_params = None

    # 用于损失归一化的移动平均值跟踪（使用requires_grad=False确保它们不参与梯度计算）
    contrast_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    bpr_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    alpha = 0.9  # 移动平均系数
    alpha_t = 0.3  # 自适应权重温度参数

    cross_domain_iter = iter(cross_domain_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.LR_DECAY_T_MAX,
        eta_min=config.LR_MIN
    )

    # 获取用户编码器参数名称
    user_encoder_param_names = {name for name, _ in model.user_encoder.named_parameters()}

    for epoch in range(config.MAX_EPOCHS):
        dataset.set_epoch(epoch)
        total_loss = 0.0
        epoch_pbar = tqdm(
            stage1_loader,
            desc=f"Stage1 Epoch {epoch + 1}/{config.MAX_EPOCHS}",
            leave=True
        )

        for batch in epoch_pbar:
            user_ids = batch["user_ids"]
            weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
            original_vecs = batch["original_vectors"].to(config.DEVICE)
            masks = batch["masks"].to(config.DEVICE)

            # 应用数据增强
            augmented_weighted = []
            augmented_masks = []
            for i in range(weighted_vecs.shape[0]):
                seq = weighted_vecs[i]
                mask = masks[i]
                aug_seq, aug_mask = augment_sequence(seq, mask)
                augmented_weighted.append(aug_seq.unsqueeze(0))
                augmented_masks.append(aug_mask.unsqueeze(0))

            weighted_vecs = torch.cat(augmented_weighted, dim=0)
            masks = torch.cat(augmented_masks, dim=0)

            optimizer.zero_grad()
            with autocast('cuda'):
                user_vecs = model(weighted_vecs, masks)

                batch_size = user_vecs.shape[0]
                bpr_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)

                for i in range(batch_size):
                    user_id = user_ids[i].item()
                    user_vec = user_vecs[i]
                    seq_len = masks[i].sum().int().item()
                    valid_original = original_vecs[i, :seq_len]

                    # 改进正样本采样：结合顺序和随机采样
                    num_pos = min(50, seq_len)
                    if seq_len <= num_pos:
                        # 短序列使用全部样本
                        pos_indices = torch.arange(seq_len, device=config.DEVICE)
                    else:
                        # 长序列结合顺序和随机采样
                        # 1. 取最近的20%样本
                        recent_ratio = 0.2
                        recent_count = max(1, int(num_pos * recent_ratio))
                        recent_indices = torch.arange(seq_len - recent_count, seq_len, device=config.DEVICE)

                        # 2. 随机采样剩余样本
                        remaining_count = num_pos - recent_count
                        random_indices = torch.randint(0, seq_len - recent_count, (remaining_count,),
                                                       device=config.DEVICE)
                        pos_indices = torch.cat([recent_indices, random_indices])

                    pos_items = valid_original[pos_indices]
                    pos_encs = model.encode_item(pos_items)

                    neg_items = random_neg_samples(
                        book_vectors,
                        stage1_loader.dataset.interactions[user_id],
                        config.NEGATIVE_SAMPLES
                    )
                    neg_encs = model.encode_item(neg_items)
                    # 累加损失时确保操作在计算图内
                    bpr_loss = bpr_loss + model.bpr_loss(user_vec, pos_encs, neg_encs)
                bpr_loss = bpr_loss / batch_size

                try:
                    source_vec, pos_vec, neg_vec = next(cross_domain_iter)
                except StopIteration:
                    cross_domain_iter = iter(cross_domain_loader)
                    source_vec, pos_vec, neg_vec = next(cross_domain_iter)
                source_vec = source_vec.to(config.DEVICE)
                pos_vec = pos_vec.to(config.DEVICE)
                neg_vec = neg_vec.to(config.DEVICE)
                contrast_loss = model.contrastive_loss(source_vec, pos_vec, neg_vec)

                # 更新损失的移动平均值
                with torch.no_grad():
                    contrast_loss_mean = alpha * contrast_loss_mean + (1 - alpha) * contrast_loss.detach()
                    bpr_loss_mean = alpha * bpr_loss_mean + (1 - alpha) * bpr_loss.detach()

                # 计算归一化损失（在计算图内进行）
                contrast_loss_norm = contrast_loss / (contrast_loss_mean + 1e-8)
                bpr_loss_norm = bpr_loss / (bpr_loss_mean + 1e-8)

                # 计算自适应权重
                lambda1, lambda2 = adaptive_loss_weights(
                    contrast_loss, bpr_loss,
                    contrast_loss_mean, bpr_loss_mean,
                    alpha_t=alpha_t
                )

                # 总损失（使用自适应权重）
                loss = lambda1 * contrast_loss_norm + lambda2 * bpr_loss_norm

            # 反向传播
            scaler.scale(loss).backward()

            # 清除用户编码器参数的梯度（仅保留BPR损失的梯度）
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in user_encoder_param_names:
                        # 仅保留BPR损失对用户编码器的梯度
                        param.grad *= (lambda2 * bpr_loss_norm / loss).detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            epoch_pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "BPR Loss": f"{bpr_loss.item():.4f}",
                "Contrast Loss": f"{contrast_loss.item():.4f}",
                "λ1": f"{lambda1.item():.3f}",
                "λ2": f"{lambda2.item():.3f}"
            })

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(stage1_loader)
        print(f"Stage1 Epoch {epoch + 1}/{config.MAX_EPOCHS}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # 验证阶段
        model.eval()
        with torch.no_grad():
            current_user_vecs = {}
            for batch in data_loaders["stage1"]:
                user_ids = batch["user_ids"]
                weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
                masks = batch["masks"].to(config.DEVICE)
                user_vecs = model(weighted_vecs, masks)

                for i, user_id in enumerate(user_ids):
                    uid = user_id.item()
                    if uid in val_interactions:
                        current_user_vecs[uid] = user_vecs[i]

            # 验证时计算多K值
            val_metrics_all = {}
            for k in [10, 20, 50]:
                val_metrics_all[k] = evaluate_performance(
                    model=model,
                    user_vecs=current_user_vecs,
                    movie_vectors=movie_vectors,
                    target_interactions=val_interactions,
                    k=k
                )
            # 打印所有K值的验证结果
            print(f"Stage1 Epoch {epoch + 1} 验证结果:")
            for k in [10, 20, 50]:
                print(
                    f"  K={k}: Val HR@{k}: {val_metrics_all[k]['mean_hit_rate']:.4f}, Val NDCG@{k}: {val_metrics_all[k]['mean_ndcg']:.4f}")
            # 早停判断用K=10的HR
            current_val_hr = val_metrics_all[10]['mean_hit_rate']

        # 早停机制与最优模型保存
        if current_val_hr > best_val_hr:
            best_val_hr = current_val_hr
            best_model_params = deepcopy(model.state_dict())
            counter = 0
            print(f"Stage1: Best Val HR@10 updated to {best_val_hr:.4f}")
        else:
            counter += 1
            print(f"Stage1: No improvement, counter={counter}/{patience}")
            if counter >= patience:
                print(f"Stage1: Early stopping at epoch {epoch + 1}")
                break

        model.train()

    # 加载第一阶段最优模型
    if best_model_params is not None:
        model.load_state_dict(best_model_params)
        print(f"Stage1: Loaded best model with Val HR@10: {best_val_hr:.4f}")

    # 第一阶段结束：测试目标域全量数据并保存模型
    test_target_domain(model, data_loaders, stage_name="Stage1", version=version)

    return model


"""第二阶段：元学习（MAML）- 带早停机制+阶段末测试"""


def maml_training(model, data_loaders, meta_optimizer, version="v11"):
    model.train()
    maml_loader = data_loaders["maml"]
    cross_domain_loader = data_loaders["cross_domain"]
    dataset = maml_loader.dataset
    book_vectors = data_loaders["book_vectors"]
    val_interactions = data_loaders["val_interactions"]
    movie_vectors = data_loaders["movie_vectors"]

    scaler = GradScaler('cuda')
    best_val_hr = 0.0  # 以K=10的HR作为早停标准
    patience = config.EARLY_STOPPING_PATIENCE
    counter = 0
    best_model_params = None

    # 用于损失归一化的移动平均值跟踪
    contrast_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    bpr_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    alpha = 0.9  # 移动平均系数
    alpha_t = 0.3  # 自适应权重温度参数

    cross_domain_iter = iter(cross_domain_loader)
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer,
        T_max=config.LR_DECAY_T_MAX,
        eta_min=config.LR_MIN
    )

    # 获取用户编码器参数名称
    user_encoder_param_names = {name for name, _ in model.user_encoder.named_parameters()}

    for epoch in range(config.MAML_EPOCHS):
        dataset.set_epoch(epoch)
        total_meta_loss = 0.0
        maml_pbar = tqdm(maml_loader, desc=f"MAML Epoch {epoch + 1}/{config.MAML_EPOCHS}")

        for batch in maml_pbar:
            user_ids = batch["user_ids"].to(config.DEVICE)
            weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
            original_vecs = batch["original_vectors"].to(config.DEVICE)
            masks = batch["masks"].to(config.DEVICE)

            # 应用数据增强
            augmented_weighted = []
            augmented_masks = []
            for i in range(weighted_vecs.shape[0]):
                seq = weighted_vecs[i]
                mask = masks[i]
                aug_seq, aug_mask = augment_sequence(seq, mask)
                augmented_weighted.append(aug_seq.unsqueeze(0))
                augmented_masks.append(aug_mask.unsqueeze(0))

            weighted_vecs = torch.cat(augmented_weighted, dim=0)
            masks = torch.cat(augmented_masks, dim=0)

            meta_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)
            inner_lr = config.MAML_LR
            meta_params = {name: param for name, param in model.named_parameters()}

            with autocast('cuda'):
                for i in range(weighted_vecs.shape[0]):
                    user_id = user_ids[i].item()
                    wv = weighted_vecs[i]
                    ov = original_vecs[i]
                    mask = masks[i]
                    seq_len = mask.sum().int().item()

                    if seq_len == 0:
                        continue
                    wv = wv[:seq_len]
                    ov = ov[:seq_len]

                    split_idx = seq_len // 2
                    support_weighted = wv[:split_idx]
                    query_weighted = wv[split_idx:]
                    support_original = ov[:split_idx]
                    query_original = ov[split_idx:]

                    support_mask = torch.ones(len(support_weighted), device=config.DEVICE)
                    query_mask = torch.ones(len(query_weighted), device=config.DEVICE)

                    # 确保快速参数有梯度
                    fast_params = {
                        name: param.clone().detach().requires_grad_(True)
                        for name, param in meta_params.items()
                    }

                    with _set_params(model, fast_params):
                        user_vec_support = model(support_weighted.unsqueeze(0), support_mask.unsqueeze(0))

                        # 改进正样本采样
                        num_pos_support = min(50, len(support_original))
                        if len(support_original) <= num_pos_support:
                            pos_indices_support = torch.arange(len(support_original), device=config.DEVICE)
                        else:
                            # 结合最近样本和随机样本
                            recent_count = max(1, int(num_pos_support * 0.2))
                            recent_indices = torch.arange(len(support_original) - recent_count,
                                                          len(support_original), device=config.DEVICE)
                            remaining_count = num_pos_support - recent_count
                            random_indices = torch.randint(0, len(support_original) - recent_count,
                                                           (remaining_count,), device=config.DEVICE)
                            pos_indices_support = torch.cat([recent_indices, random_indices])

                        pos_items_support = support_original[pos_indices_support]
                        pos_encs_support = model.encode_item(pos_items_support)
                        neg_items_support = random_neg_samples(
                            book_vectors,
                            maml_loader.dataset.interactions[user_id],
                            config.NEGATIVE_SAMPLES
                        )
                        neg_encs_support = model.encode_item(neg_items_support)
                        bpr_loss_support = model.bpr_loss(
                            user_vec_support.squeeze(0),
                            pos_encs_support,
                            neg_encs_support
                        )

                        try:
                            source_vec_s, pos_vec_s, neg_vec_s = next(cross_domain_iter)
                        except StopIteration:
                            cross_domain_iter = iter(cross_domain_loader)
                            source_vec_s, pos_vec_s, neg_vec_s = next(cross_domain_iter)
                        source_vec_s = source_vec_s.to(config.DEVICE)
                        pos_vec_s = pos_vec_s.to(config.DEVICE)
                        neg_vec_s = neg_vec_s.to(config.DEVICE)
                        contrast_loss_support = model.contrastive_loss(source_vec_s, pos_vec_s, neg_vec_s)

                        # 更新支持集损失的移动平均值
                        with torch.no_grad():
                            batch_contrast_mean = alpha * contrast_loss_mean + (
                                        1 - alpha) * contrast_loss_support.detach()
                            batch_bpr_mean = alpha * bpr_loss_mean + (1 - alpha) * bpr_loss_support.detach()

                        # 计算支持集自适应权重
                        lambda1_s, lambda2_s = adaptive_loss_weights(
                            contrast_loss_support, bpr_loss_support,
                            batch_contrast_mean, batch_bpr_mean,
                            alpha_t=alpha_t
                        )

                        loss_support = (lambda1_s * contrast_loss_support
                                        + lambda2_s * bpr_loss_support)

                    # 计算支持集损失梯度，允许未使用的参数
                    grads = torch.autograd.grad(
                        loss_support,
                        fast_params.values(),
                        create_graph=True,
                        allow_unused=True  # 新增：允许未使用的参数
                    )

                    # 处理可能为None的梯度（未使用的参数）
                    adjusted_grads = []
                    for idx, (name, param) in enumerate(fast_params.items()):
                        grad = grads[idx]
                        if grad is None:
                            # 为未使用的参数创建零梯度
                            adjusted_grads.append(torch.zeros_like(param))
                        elif name in user_encoder_param_names:
                            # 仅使用BPR损失的梯度更新用户编码器
                            bpr_grads = torch.autograd.grad(
                                bpr_loss_support,
                                param,
                                retain_graph=True,
                                allow_unused=True
                            )[0]
                            adjusted_grads.append(bpr_grads if bpr_grads is not None else torch.zeros_like(param))
                        else:
                            adjusted_grads.append(grad)

                    # 更新快速参数
                    with torch.no_grad():
                        updated_params = {
                            name: param - inner_lr * grad
                            for name, param, grad in zip(fast_params.keys(), fast_params.values(), adjusted_grads)
                        }

                    # 查询集计算
                    with _set_params(model, updated_params):
                        user_vec_query = model(query_weighted.unsqueeze(0), query_mask.unsqueeze(0))

                        # 改进正样本采样
                        num_pos_query = min(50, len(query_original))
                        if len(query_original) <= num_pos_query:
                            pos_indices_query = torch.arange(len(query_original), device=config.DEVICE)
                        else:
                            recent_count = max(1, int(num_pos_query * 0.2))
                            recent_indices = torch.arange(len(query_original) - recent_count,
                                                          len(query_original), device=config.DEVICE)
                            remaining_count = num_pos_query - recent_count
                            random_indices = torch.randint(0, len(query_original) - recent_count,
                                                           (remaining_count,), device=config.DEVICE)
                            pos_indices_query = torch.cat([recent_indices, random_indices])

                        pos_items_query = query_original[pos_indices_query]
                        pos_encs_query = model.encode_item(pos_items_query)

                        neg_items_query = random_neg_samples(
                            book_vectors,
                            maml_loader.dataset.interactions[user_id],
                            config.NEGATIVE_SAMPLES
                        )
                        neg_encs_query = model.encode_item(neg_items_query)
                        bpr_loss_query = model.bpr_loss(
                            user_vec_query.squeeze(0),
                            pos_encs_query,
                            neg_encs_query
                        )

                        try:
                            source_vec_q, pos_vec_q, neg_vec_q = next(cross_domain_iter)
                        except StopIteration:
                            cross_domain_iter = iter(cross_domain_loader)
                            source_vec_q, pos_vec_q, neg_vec_q = next(cross_domain_iter)
                        source_vec_q = source_vec_q.to(config.DEVICE)
                        pos_vec_q = pos_vec_q.to(config.DEVICE)
                        neg_vec_q = neg_vec_q.to(config.DEVICE)
                        contrast_loss_query = model.contrastive_loss(source_vec_q, pos_vec_q, neg_vec_q)

                        # 计算查询集自适应权重
                        lambda1_q, lambda2_q = adaptive_loss_weights(
                            contrast_loss_query, bpr_loss_query,
                            batch_contrast_mean, batch_bpr_mean,
                            alpha_t=alpha_t
                        )

                        task_meta_loss = (lambda1_q * contrast_loss_query
                                          + lambda2_q * bpr_loss_query)

                    meta_loss = meta_loss + task_meta_loss

                meta_loss = meta_loss / weighted_vecs.shape[0]

                # 更新元损失的移动平均值并归一化
                with torch.no_grad():
                    contrast_loss_mean = alpha * contrast_loss_mean + (1 - alpha) * contrast_loss_query.detach()
                    bpr_loss_mean = alpha * bpr_loss_mean + (1 - alpha) * bpr_loss_query.detach()

            meta_optimizer.zero_grad()
            scaler.scale(meta_loss).backward()

            # 调整用户编码器参数的梯度（仅保留BPR损失部分）
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in user_encoder_param_names and param.grad is not None:
                        # 计算仅BPR损失的梯度比例
                        bpr_ratio = (lambda2_q * bpr_loss_query) / task_meta_loss
                        param.grad *= bpr_ratio.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(meta_optimizer)
            scaler.update()

            total_meta_loss += meta_loss.item()
            maml_pbar.set_postfix({
                "Meta Loss": f"{meta_loss.item():.4f}",
                "λ1": f"{lambda1_q.item():.3f}",
                "λ2": f"{lambda2_q.item():.3f}"
            })

        meta_scheduler.step()
        current_meta_lr = meta_optimizer.param_groups[0]['lr']
        avg_meta_loss = total_meta_loss / len(maml_loader)
        print(
            f"MAML Epoch {epoch + 1}/{config.MAML_EPOCHS}, Avg Meta Loss: {avg_meta_loss:.4f}, Meta LR: {current_meta_lr:.6f}")

        # 验证阶段
        model.eval()
        with torch.no_grad():
            current_user_vecs = {}
            for batch in data_loaders["maml"]:
                user_ids = batch["user_ids"]
                weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
                masks = batch["masks"].to(config.DEVICE)
                user_vecs = model(weighted_vecs, masks)

                for i, user_id in enumerate(user_ids):
                    uid = user_id.item()
                    if uid in val_interactions:
                        current_user_vecs[uid] = user_vecs[i]

            # 计算多K值验证指标
            val_metrics_all = {}
            for k in [10, 20, 50]:
                val_metrics_all[k] = evaluate_performance(
                    model=model,
                    user_vecs=current_user_vecs,
                    movie_vectors=movie_vectors,
                    target_interactions=val_interactions,
                    k=k
                )
            # 打印所有K值的验证结果
            print(f"MAML Epoch {epoch + 1} 验证结果:")
            for k in [10, 20, 50]:
                print(
                    f"  K={k}: Val HR@{k}: {val_metrics_all[k]['mean_hit_rate']:.4f}, Val NDCG@{k}: {val_metrics_all[k]['mean_ndcg']:.4f}")
            # 早停判断用K=10的HR
            current_val_hr = val_metrics_all[10]['mean_hit_rate']

        # 早停机制与最优模型保存
        if current_val_hr > best_val_hr:
            best_val_hr = current_val_hr
            best_model_params = deepcopy(model.state_dict())
            counter = 0
            print(f"MAML: Best Val HR@10 updated to {best_val_hr:.4f}")
        else:
            counter += 1
            print(f"MAML: No improvement, counter={counter}/{patience}")
            if counter >= patience:
                print(f"MAML: Early stopping at epoch {epoch + 1}")
                break

        model.train()

    # 加载MAML阶段最优模型
    if best_model_params is not None:
        model.load_state_dict(best_model_params)
        print(f"MAML: Loaded best model with Val HR@10: {best_val_hr:.4f}")

    # MAML阶段结束：测试目标域全量数据并保存模型
    test_target_domain(model, data_loaders, stage_name="MAML", version=version)

    return model


"""第三阶段：批次微调"""


def finetune_and_save(model, data_loaders, version="v11"):
    model.train()
    finetune_loader = data_loaders["finetune"]
    cross_domain_loader = data_loaders["cross_domain"]
    book_vectors = data_loaders["book_vectors"]
    val_interactions = data_loaders["val_interactions"]
    best_val_hr = 0.0  # 以K=10的HR作为早停标准

    # 初始化微调优化器和调度器
    ft_optimizer = optim.Adam(
        model.parameters(),
        lr=config.FINETUNE_LR,
        weight_decay=config.REGULARIZATION_LAMBDA * 2  # 增加权重衰减
    )
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer,
        T_max=config.LR_DECAY_T_MAX // 3,
        eta_min=config.LR_MIN
    )
    ft_scaler = GradScaler('cuda')
    cross_domain_iter = iter(cross_domain_loader)

    # 用于损失归一化的移动平均值跟踪
    contrast_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    bpr_loss_mean = torch.tensor(0.0, device=config.DEVICE, requires_grad=False)
    alpha = 0.9  # 移动平均系数
    alpha_t = 0.3  # 自适应权重温度参数

    # 获取用户编码器参数名称
    user_encoder_param_names = {name for name, _ in model.user_encoder.named_parameters()}

    # 预生成并保存电影向量
    movie_tensor = torch.tensor(
        data_loaders["movie_vectors"],
        dtype=torch.float32,
        device=config.DEVICE
    )
    model.save_movie_vectors = model.encode_item(movie_tensor).detach()

    # 微调训练循环
    for ft_epoch in range(config.FINETUNE_EPOCHS):
        total_ft_loss = 0.0
        ft_pbar = tqdm(
            finetune_loader,
            desc=f"Finetune Epoch {ft_epoch + 1}/{config.FINETUNE_EPOCHS}",
            leave=True
        )

        for batch in ft_pbar:
            user_ids = batch["user_ids"]
            weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
            original_vecs = batch["original_vectors"].to(config.DEVICE)
            masks = batch["masks"].to(config.DEVICE)
            batch_size = user_ids.shape[0]

            # 应用数据增强
            augmented_weighted = []
            augmented_masks = []
            for i in range(weighted_vecs.shape[0]):
                seq = weighted_vecs[i]
                mask = masks[i]
                aug_seq, aug_mask = augment_sequence(seq, mask)
                augmented_weighted.append(aug_seq.unsqueeze(0))
                augmented_masks.append(aug_mask.unsqueeze(0))

            weighted_vecs = torch.cat(augmented_weighted, dim=0)
            masks = torch.cat(augmented_masks, dim=0)

            ft_optimizer.zero_grad()
            with autocast('cuda'):
                # 1. 计算BPR损失（批次级）
                user_vecs = model(weighted_vecs, masks)
                bpr_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)

                for i in range(batch_size):
                    user_id = user_ids[i].item()
                    user_vec = user_vecs[i]
                    seq_len = masks[i].sum().int().item()

                    if seq_len == 0:
                        continue

                    valid_original = original_vecs[i, :seq_len]
                    num_pos = min(40, seq_len)

                    # 改进正样本采样
                    if seq_len <= num_pos:
                        pos_indices = torch.arange(seq_len, device=config.DEVICE)
                    else:
                        # 结合最近样本和随机样本
                        recent_count = max(1, int(num_pos * 0.2))
                        recent_indices = torch.arange(seq_len - recent_count, seq_len, device=config.DEVICE)
                        remaining_count = num_pos - recent_count
                        random_indices = torch.randint(0, seq_len - recent_count,
                                                       (remaining_count,), device=config.DEVICE)
                        pos_indices = torch.cat([recent_indices, random_indices])

                    pos_items = valid_original[pos_indices]
                    pos_encs = model.encode_item(pos_items)

                    # 生成负样本
                    neg_items = random_neg_samples(
                        book_vectors,
                        finetune_loader.dataset.interactions[user_id],
                        config.NEGATIVE_SAMPLES
                    )
                    neg_encs = model.encode_item(neg_items)
                    bpr_loss = bpr_loss + model.bpr_loss(user_vec, pos_encs, neg_encs)

                bpr_loss = bpr_loss / batch_size

                # 2. 计算跨域对比损失
                try:
                    source_vec, pos_vec, neg_vec = next(cross_domain_iter)
                except StopIteration:
                    cross_domain_iter = iter(cross_domain_loader)
                    source_vec, pos_vec, neg_vec = next(cross_domain_iter)
                source_vec = source_vec.to(config.DEVICE)
                pos_vec = pos_vec.to(config.DEVICE)
                neg_vec = neg_vec.to(config.DEVICE)
                contrast_loss = model.contrastive_loss(source_vec, pos_vec, neg_vec)

                # 更新损失的移动平均值并归一化
                with torch.no_grad():
                    contrast_loss_mean = alpha * contrast_loss_mean + (1 - alpha) * contrast_loss.detach()
                    bpr_loss_mean = alpha * bpr_loss_mean + (1 - alpha) * bpr_loss.detach()

                # 计算归一化损失（在计算图内）
                contrast_loss_norm = contrast_loss / (contrast_loss_mean + 1e-8)
                bpr_loss_norm = bpr_loss / (bpr_loss_mean + 1e-8)

                # 计算自适应权重
                lambda1, lambda2 = adaptive_loss_weights(
                    contrast_loss, bpr_loss,
                    contrast_loss_mean, bpr_loss_mean,
                    alpha_t=alpha_t
                )

                # 3. 总损失（使用自适应权重）
                loss = lambda1 * contrast_loss_norm + lambda2 * bpr_loss_norm

            # 反向传播与参数更新
            ft_scaler.scale(loss).backward()

            # 清除用户编码器参数中来自对比损失的梯度
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in user_encoder_param_names and param.grad is not None:
                        # 仅保留BPR损失对用户编码器的梯度
                        param.grad *= (lambda2 * bpr_loss_norm / loss).detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            ft_scaler.step(ft_optimizer)
            ft_scaler.update()

            total_ft_loss += loss.item()
            current_lr = ft_optimizer.param_groups[0]['lr']
            ft_pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "BPR Loss": f"{bpr_loss.item():.4f}",
                "Contrast Loss": f"{contrast_loss.item():.4f}",
                "λ1": f"{lambda1.item():.3f}",
                "λ2": f"{lambda2.item():.3f}",
                "LR": f"{current_lr:.6f}"
            })

        # 学习率调度
        ft_scheduler.step()
        avg_ft_loss = total_ft_loss / len(finetune_loader)
        print(
            f"Finetune Epoch {ft_epoch + 1}/{config.FINETUNE_EPOCHS}, Avg Loss: {avg_ft_loss:.4f}, LR: {current_lr:.6f}")

        # 微调阶段验证
        if (ft_epoch + 1) % 5 == 0 or ft_epoch == config.FINETUNE_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                current_user_vecs = {}
                for batch in tqdm(finetune_loader, desc="Finetune - 生成验证用户向量"):
                    user_ids = batch["user_ids"]
                    weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
                    masks = batch["masks"].to(config.DEVICE)
                    user_vecs = model(weighted_vecs, masks)

                    for i, user_id in enumerate(user_ids):
                        uid = user_id.item()
                        current_user_vecs[uid] = user_vecs[i]

                model.save_user_vectors = current_user_vecs

                # 计算多K值验证指标
                val_metrics_all = {}
                for k in [10, 20, 50]:
                    val_metrics_all[k] = evaluate_performance(
                        model=None,
                        user_vecs=current_user_vecs,
                        movie_vectors=model.save_movie_vectors,
                        target_interactions=val_interactions,
                        k=k
                    )
                # 打印所有K值的验证结果
                print(f"Finetune Epoch {ft_epoch + 1} 验证结果:")
                for k in [10, 20, 50]:
                    print(
                        f"  K={k}: Val HR@{k}: {val_metrics_all[k]['mean_hit_rate']:.4f}, Val NDCG@{k}: {val_metrics_all[k]['mean_ndcg']:.4f}")
                # 早停判断用K=10的HR
                current_val_hr = val_metrics_all[10]['mean_hit_rate']

                if current_val_hr > best_val_hr:
                    best_val_hr = current_val_hr
                    print(f"Finetune: Best Val HR@10 updated to {best_val_hr:.4f}")

            model.train()

    # 微调阶段结束：生成最终用户向量并保存
    model.eval()
    with torch.no_grad():
        final_user_vecs = {}
        for batch in tqdm(finetune_loader, desc="Finetune - 生成最终用户向量"):
            user_ids = batch["user_ids"]
            weighted_vecs = batch["weighted_vectors"].to(config.DEVICE)
            masks = batch["masks"].to(config.DEVICE)
            user_vecs = model(weighted_vecs, masks)

            for i, user_id in enumerate(user_ids):
                uid = user_id.item()
                final_user_vecs[uid] = user_vecs[i]

        model.save_user_vectors = final_user_vecs
        print(f"Finetune: 已生成 {len(final_user_vecs)} 个用户的最终向量")

    # 微调阶段结束：测试目标域全量数据并保存模型
    test_target_domain(model, data_loaders, stage_name="Finetune", version=version)

    return model


def main(precomputed_paths=None, version="v11"):
    set_random_seed(config.SEED)
    # 打印配置参数
    print("=" * 50)
    print("当前训练配置参数:")
    print("=" * 50)
    # 打印关键配置参数
    print(f"设备: {config.DEVICE}")
    print(f"随机种子: {config.SEED}")
    print(f"向量维度: 输入={config.ITEM_DIM}, 输出={config.EMBED_DIM}")
    print(f"滑动窗口大小: {config.WINDOW_SIZE}")
    print(
        f"对比学习参数: 正样本数={config.CONTRASTIVE_POSITIVES}, 负样本数={config.CONTRASTIVE_NEGATIVES}, 温度={config.CONTRASTIVE_TEMP}")
    print(f"自适应损失权重温度参数: {0.3}")  # 显示自适应权重参数
    print(f"训练轮数: 第一阶段={config.MAX_EPOCHS}, 元学习={config.MAML_EPOCHS}, 微调={config.FINETUNE_EPOCHS}")
    print(f"学习率: 基础={config.LEARNING_RATE}, 元学习={config.MAML_LR}, 微调={config.FINETUNE_LR}")
    print(f"早停耐心值: {config.EARLY_STOPPING_PATIENCE}")
    print(f"批大小: 基础={config.BATCH_SIZE}, 元学习={config.META_BATCH_SIZE}")
    print("=" * 50 + "\n")

    precomputed_paths = precomputed_paths or {}
    data_loaders = get_data_loaders(precomputed_paths)
    model = CrossDomainRecommender().to(config.DEVICE)

    # 第一阶段训练
    print("=" * 50)
    print(f"Starting {version} - Stage 1 Training...")
    print("=" * 50)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.REGULARIZATION_LAMBDA * 2)
    model = stage1_training(model, data_loaders, optimizer, version=version)

    # 第二阶段训练
    print("\n" + "=" * 50)
    print(f"Starting {version} - MAML Training...")
    print("=" * 50)
    meta_optimizer = optim.Adam(model.parameters(), lr=config.MAML_LR, weight_decay=config.REGULARIZATION_LAMBDA * 2)
    model = maml_training(model, data_loaders, meta_optimizer, version=version)

    # 第三阶段训练
    print("\n" + "=" * 50)
    print(f"Starting {version} - Finetuning...")
    print("=" * 50)
    model = finetune_and_save(model, data_loaders, version=version)

    # 最终汇总测试结果
    print("\n" + "=" * 60)
    print(f"{version} - 所有阶段训练完成！各阶段最优模型测试结果汇总：")
    print("=" * 60)
    # 加载各阶段模型并打印关键指标
    stage1_model = torch.load(f"{version}_stage1_best_model.pt", weights_only=False)
    maml_model = torch.load(f"{version}_maml_best_model.pt", weights_only=False)
    finetune_model = torch.load(f"{version}_finetune_best_model.pt", weights_only=False)

    # 打印所有K值的结果
    k_list = [10, 20, 50]
    for stage, model_data in [("Stage1", stage1_model), ("MAML", maml_model), ("Finetune", finetune_model)]:
        print(f"\n{stage} 阶段指标：")
        for k in k_list:
            metrics = model_data["test_metrics"][k]
            print(f"  K={k}: HR@{k}={metrics['mean_hit_rate']:.4f}, NDCG@{k}={metrics['mean_ndcg']:.4f}")

    # 保存最终综合模型
    torch.save({
        "version": version,
        "final_model_state_dict": model.state_dict(),
        "final_user_vectors": model.save_user_vectors,
        "final_movie_vectors": model.save_movie_vectors,
        "stage1_model_path": f"{version}_stage1_best_model.pt",
        "maml_model_path": f"{version}_maml_best_model.pt",
        "finetune_model_path": f"{version}_finetune_best_model.pt"
    }, f"{version}_final_ensemble_model.pt")

    print(f"\n最终综合模型已保存至: {version}_final_ensemble_model.pt")


if __name__ == "__main__":
    precomputed = {
        # "stage1": "data/Books_user_interactions_seq_5.npz",
        # "maml": "data/Books_user_interactions_seq_13.npz",
        # "finetune": "data/Books_user_interactions_seq_5.npz"
        # "stage1": "data/Movies_and_TV_user_interactions_seq_5.npz",
        # "maml": "data/Movies_and_TV_user_interactions_seq_13.npz",
        # "finetune": "data/Movies_and_TV_user_interactions_seq_5.npz"
        "stage1": "data/Books_user_interactions_weighted_5.npz",
        "maml": "data/Books_user_interactions_weighted_13.npz",
        "finetune": "data/Books_user_interactions_weighted_5.npz"
        # "stage1": "data/Movies_and_TV_user_interactions_weighted_5.npz",
        # "maml": "data/Movies_and_TV_user_interactions_weighted_13.npz",
        # "finetune": "data/Movies_and_TV_user_interactions_weighted_5.npz"
        # "stage1": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\CDs_and_Vinyl_user_interactions_weighted_5.npz",
        # "maml": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\CDs_and_Vinyl_user_interactions_weighted_13.npz",
        # "finetune": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\CDs_and_Vinyl_user_interactions_weighted_5.npz"
        # "stage1": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\Books_user_interactions_weighted_5.npz",
        # "maml": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\Books_user_interactions_weighted_13.npz",
        # "finetune": "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed\Books_user_interactions_weighted_5.npz"
        # "stage1": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Toys_and_Games_user_interactions_weighted_5.npz",
        # "maml": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Toys_and_Games_user_interactions_weighted_13.npz",
        # "finetune": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Toys_and_Games_user_interactions_weighted_5.npz"
        # "stage1": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Books_user_interactions_weighted_5.npz",
        # "maml": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Books_user_interactions_weighted_13.npz",
        # "finetune": "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Books_user_interactions_weighted_5.npz"
    }
    main(precomputed_paths=precomputed, version="v1_AB_time_Adaptive")