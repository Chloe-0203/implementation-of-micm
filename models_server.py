import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class ItemEncoder(nn.Module):
    """物品向量编码器（MLP）"""

    #老模型用的nn.BatchNorm1d(dim)，适合BM
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ItemEncoder, self).__init__()
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.2))  # 增加dropout比例
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Dropout(0.2))  # 增加输出层dropout比例
        self.mlp = nn.Sequential(*layers)

        # 添加权重初始化
        self._initialize_weights()
    #新模型用的是nn.LayerNorm(dim)，适合BC和BT
    # def __init__(self, input_dim, hidden_dims, output_dim):
    #     super(ItemEncoder, self).__init__()
    #     layers = []
    #     in_dim = input_dim
    #     for dim in hidden_dims:
    #         layers.append(nn.Linear(in_dim, dim))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.LayerNorm(dim))
    #         layers.append(nn.Dropout(0.2))  # 增加dropout比例
    #         in_dim = dim
    #     layers.append(nn.Linear(in_dim, output_dim))
    #     layers.append(nn.Dropout(0.2))  # 增加输出层dropout比例
    #     self.mlp = nn.Sequential(*layers)
    #
    #     # 添加权重初始化
    #     self._initialize_weights()

    def _initialize_weights(self):
        """初始化MLP中的线性层权重"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class SequenceEncoder(nn.Module):
    """子序列编码器（LSTM）"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):  # 减少LSTM层数
        super(SequenceEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2  # 增加LSTM层间dropout
        )
        self.dropout = nn.Dropout(0.2)  # 增加dropout比例
        self.l2_norm = nn.LayerNorm(hidden_dim * 2)  # 新增LayerNorm
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.l2_norm(last_out)  # 应用LayerNorm
        return self.fc(self.dropout(last_out))


class UserEncoder(nn.Module):
    """用户向量编码器（批量处理+Transformer）"""

    def __init__(self, item_encoder, seq_encoder, window_size, num_heads, num_layers=1):  # 减少Transformer层数
        super(UserEncoder, self).__init__()
        self.item_encoder = item_encoder
        self.seq_encoder = seq_encoder
        self.window_size = window_size

        # 增强Transformer正则化
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.2,  # 增加dropout比例
            batch_first=True,
            layer_norm_eps=1e-5  # 调整LayerNorm参数
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.l2_norm = nn.LayerNorm(config.EMBED_DIM)  # 新增输出LayerNorm

        self._initialize_transformer_weights()

    def _initialize_transformer_weights(self):
        for layer in self.transformer.layers:
            attn = layer.self_attn
            
            if hasattr(attn, 'in_proj_weight'):
                nn.init.xavier_uniform_(attn.in_proj_weight)
            else:
                for param in [attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight]:
                    if param is not None:
                        nn.init.xavier_uniform_(param)

            nn.init.xavier_uniform_(attn.out_proj.weight)

            if hasattr(attn, 'in_proj_bias'):
                if attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)
            else:
                for param in [attn.q_proj_bias, attn.k_proj_bias, attn.v_proj_bias]:
                    if param is not None:
                        nn.init.zeros_(param)

            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

            # 前馈网络正则化增强
            ffn = layer.linear1
            nn.init.xavier_uniform_(ffn.weight)
            nn.init.zeros_(ffn.bias)
            
            ffn_out = layer.linear2
            nn.init.xavier_uniform_(ffn_out.weight)
            nn.init.zeros_(ffn_out.bias)

    def forward(self, weighted_items, mask):
        batch_size, max_seq_len, item_dim = weighted_items.shape

        # 1. 物品向量编码
        flat_items = weighted_items.reshape(-1, item_dim)
        flat_encoded = self.item_encoder(flat_items)
        encoded_items = flat_encoded.reshape(batch_size, max_seq_len, config.EMBED_DIM)

        # 2. 生成滑动窗口子序列
        actual_window_size = min(self.window_size, max_seq_len)
        window_steps = max_seq_len - actual_window_size + 1
        if window_steps <= 0:
            window_steps = 1

        windows = encoded_items.unfold(
            dimension=1,
            size=actual_window_size,
            step=1
        )

        # 3. LSTM子序列编码
        batch_windows = windows.reshape(-1, actual_window_size, config.EMBED_DIM)
        seq_encoded = self.seq_encoder(batch_windows)
        seq_encoded = seq_encoded.reshape(batch_size, window_steps, config.EMBED_DIM)

        # 4. Transformer全局编码
        if window_steps > 1:
            window_mask = mask[:, :-actual_window_size + 1]
        else:
            window_mask = mask[:, 0:1]
        transformer_mask = (window_mask == 0).bool()

        transformer_out = self.transformer(seq_encoded, mask=None, src_key_padding_mask=transformer_mask) + seq_encoded
        transformer_out = self.l2_norm(transformer_out)  # 应用LayerNorm

        # 5. 聚合生成用户向量
        valid_counts = window_mask.sum(dim=1, keepdim=True).clamp(min=1)
        user_vecs = (transformer_out * window_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        
        return user_vecs


class CrossDomainRecommender(nn.Module):
    """跨域推荐模型（整合新编码器）"""

    def __init__(self):
        super(CrossDomainRecommender, self).__init__()
        # 物品编码器
        self.item_encoder = ItemEncoder(
            input_dim=config.ITEM_DIM,
            hidden_dims=config.MLP_HIDDEN_DIMS,
            output_dim=config.EMBED_DIM
        )
        # LSTM子序列编码器
        self.seq_encoder = SequenceEncoder(
            input_dim=config.EMBED_DIM,
            hidden_dim=64,  # 减小隐藏维度
            output_dim=config.EMBED_DIM
        )
        # Transformer用户编码器
        self.user_encoder = UserEncoder(
            self.item_encoder,
            self.seq_encoder,
            window_size=config.WINDOW_SIZE,
            num_heads=config.NUM_HEADS,
            num_layers=1  # 减少Transformer层数
        )

        self.save_user_vectors = {}
        self.save_movie_vectors = None

    def forward(self, weighted_items, mask):
        return self.user_encoder(weighted_items, mask)

    def encode_item(self, item_vectors):
        return self.item_encoder(item_vectors)


    def contrastive_loss(self, source_vec, pos_vecs, neg_vecs):
        source_enc = self.encode_item(source_vec)

        batch_size, num_pos, item_dim = pos_vecs.shape
        pos_flat = pos_vecs.view(-1, item_dim)
        pos_enc_flat = self.encode_item(pos_flat)
        pos_enc = pos_enc_flat.view(batch_size, num_pos, -1)

        batch_size, num_neg, item_dim = neg_vecs.shape
        neg_flat = neg_vecs.view(-1, item_dim)
        neg_enc_flat = self.encode_item(neg_flat)
        neg_enc = neg_enc_flat.view(batch_size, num_neg, -1)

        # 计算源向量与所有正样本的相似度（[batch_size, num_pos]）
        pos_sim = F.cosine_similarity(
            source_enc.unsqueeze(1),
            pos_enc,
            dim=2
        )

        # 计算源向量与所有负样本的相似度（[batch_size, num_neg]）
        neg_sim = F.cosine_similarity(
            source_enc.unsqueeze(1),
            neg_enc,
            dim=2
        )

        # 拼接正负样本相似度（[batch_size, num_pos + num_neg]）
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        # 温度系数缩放
        logits = logits / config.CONTRASTIVE_TEMP

        # 正确构造标签：前num_pos列为1，后num_neg列为0
        labels = torch.cat([
            torch.ones(batch_size, num_pos, device=config.DEVICE),
            torch.zeros(batch_size, num_neg, device=config.DEVICE)
        ], dim=1)

        # 使用二元交叉熵损失（每个位置独立判断是否为正样本）
        return F.binary_cross_entropy_with_logits(logits, labels)

    def bpr_loss(self, user_vec, pos_item_vecs, neg_item_vecs):
        pos_scores = torch.matmul(pos_item_vecs, user_vec.unsqueeze(1)).squeeze(1)
        neg_scores = torch.matmul(neg_item_vecs, user_vec.unsqueeze(1)).squeeze(1)
        
        pos_scores = pos_scores.unsqueeze(1)
        neg_scores = neg_scores.unsqueeze(0)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss
    