import torch


class Config:
    # 数据路径
    # DATA_DIR = "./data"
    # BOOK_VECTORS = "Books_initial_embeddings.npy"
    # MOVIE_VECTORS = "Movies_and_TV_initial_embeddings.npy"
    # CROSS_DOMAIN_PAIRS = "book_samples_30.csv"#书籍的30个正负样本
    # SOURCE_INTERACTIONS_5 = "Books_user_interactions_5.json"  # 第一阶段训练数据
    # SOURCE_INTERACTIONS_13 = "Books_user_interactions_13.json"  # 元学习数据
    # TARGET_INTERACTIONS_13 = "Movies_and_TV_user_interactions_13.json"
    # TARGET_INTERACTIONS = "Movies_and_TV_user_interactions_5.json"  # 测试数据

    DATA_DIR = "./data"
    BOOK_VECTORS = "Movies_and_TV_initial_embeddings.npy"
    MOVIE_VECTORS = "Books_initial_embeddings.npy"
    CROSS_DOMAIN_PAIRS = "movie_sampless_euclidean_30.csv"  # 书籍的30个正负样本
    SOURCE_INTERACTIONS_5 = "Movies_and_TV_user_interactions_5.json"  # 第一阶段训练数据
    SOURCE_INTERACTIONS_13 = "Movies_and_TV_user_interactions_13.json"  # 元学习数据
    TARGET_INTERACTIONS_13 = "Books_user_interactions_13.json"
    TARGET_INTERACTIONS = "Books_user_interactions_5.json"  # 测试数据

    # DATA_DIR = "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed"
    # BOOK_VECTORS = "CDs_and_Vinyl_initial_embeddings.npy"
    # MOVIE_VECTORS = "Books_initial_embeddings.npy"
    # CROSS_DOMAIN_PAIRS = "cd_samples_30.csv"#书籍的30个正负样本
    # SOURCE_INTERACTIONS_5 = "CDs_and_Vinyl_user_interactions_5.json"  # 第一阶段训练数据
    # SOURCE_INTERACTIONS_13 = "CDs_and_Vinyl_user_interactions_13.json"  # 元学习数据
    # TARGET_INTERACTIONS_13 = "Books_user_interactions_13.json"
    # TARGET_INTERACTIONS = "Books_user_interactions_5.json"  # 测试数据

    # DATA_DIR = "D:\CODE\MICM\Dataset\Amazon_BC\Data_Processed"
    # BOOK_VECTORS = "Books_initial_embeddings.npy"
    # MOVIE_VECTORS = "CDs_and_Vinyl_initial_embeddings.npy"
    # CROSS_DOMAIN_PAIRS = "book_samples_30.csv"  # 书籍的30个正负样本
    # SOURCE_INTERACTIONS_5 = "Books_user_interactions_5.json"  # 第一阶段训练数据
    # SOURCE_INTERACTIONS_13 = "Books_user_interactions_13.json"  # 元学习数据
    # TARGET_INTERACTIONS_13 = "CDs_and_Vinyl_user_interactions_13.json"
    # TARGET_INTERACTIONS = "CDs_and_Vinyl_user_interactions_5.json"  # 测试数据

    # DATA_DIR = "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed"
    # BOOK_VECTORS = "Books_initial_embeddings.npy"
    # MOVIE_VECTORS = "Toys_and_Games_initial_embeddings.npy"
    # CROSS_DOMAIN_PAIRS = "book_samples_30.csv"  # 书籍的30个正负样本
    # SOURCE_INTERACTIONS_5 = "Books_user_interactions_5.json"  # 第一阶段训练数据
    # SOURCE_INTERACTIONS_13 = "Books_user_interactions_13.json"  # 元学习数据
    # TARGET_INTERACTIONS_13 = "Toys_and_Games_user_interactions_13.json"
    # TARGET_INTERACTIONS = "Toys_and_Games_user_interactions_5.json"  # 测试数据

    # DATA_DIR = "D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed"
    # BOOK_VECTORS = "Toys_and_Games_initial_embeddings.npy"
    # MOVIE_VECTORS = "Books_initial_embeddings.npy"
    # CROSS_DOMAIN_PAIRS = "toy_samples_30.csv"  # 书籍的30个正负样本
    # SOURCE_INTERACTIONS_5 = "Toys_and_Games_user_interactions_5.json"  # 第一阶段训练数据
    # SOURCE_INTERACTIONS_13 = "Toys_and_Games_user_interactions_13.json"  # 元学习数据
    # TARGET_INTERACTIONS_13 = "Books_user_interactions_13.json"
    # TARGET_INTERACTIONS = "Books_user_interactions_5.json"  # 测试数据
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    SEED = 18

    # 模型参数
    ITEM_DIM = 384  # 物品初始向量维度
    EMBED_DIM = 64  # 编码后向量维度
    MLP_HIDDEN_DIMS = [128,64]  # 物品编码器MLP隐藏层
    SEQ_MLP_HIDDEN_DIMS = [32,64]  # 子序列编码MLP
    CONTRASTIVE_POSITIVES = 15  # 正样本数量（默认10，可修改）
    CONTRASTIVE_NEGATIVES = 20  # 负样本数量（默认20，可修改）
    NUM_HEADS = 2  # 多头注意力头数
    WINDOW_SIZE = 5  # 滑动窗口大小
    ALPHA = 0.5  # 类别权重计算参数

    # 训练配置
    EPOCHS = 10
    BATCH_SIZE = 128
    META_BATCH_SIZE = 64
    LEARNING_RATE = 5e-5
    MAML_LR = 5e-6  # 元学习学习率
    LR_DECAY_T_MAX = 50  # 余弦退火周期（完整周期的迭代次数）
    LR_MIN = 1e-6  # 最小学习率
    MAX_EPOCHS = 1  # 第一阶段总轮数
    MAML_EPOCHS = 100  # 元学习总轮数
    FINETUNE_LR = 1e-6  # 微调学习率
    FINETUNE_EPOCHS = 1  # 微调总轮数
    CONTRASTIVE_TEMP = 0.7  # 对比学习温度参数
    # LOSS_LAMBDA1_CONTRAST = 0.5  #对比损失约0.7，bpr损失约2.5
    # LOSS_LAMBDA2_BPR = 0.6  # 对比损失约0.7，bpr损失约2.5
    # LOSS_LAMBDA_CONTRAST = 0.3
    REGULARIZATION_LAMBDA = 5e-2  # 正则化系数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # 聚类配置（GSOM_MI参数）
    CLUSTER_CONFIG = {
        "spread_factor": 0.4,
        "max_iter": 10,#模型中用户兴趣聚类
        "learning_rate": 0.015,
        "radius": 0.6,
        "dim": ITEM_DIM,
        "delete_factor": 4,
        'batch_size': 256,  # 批量处理大小
        'adaptive_iter': True,
        "device": "cpu"
    }

    # 负采样配置
    NEGATIVE_SAMPLES = 20  # BPR损失负样本数量
    CACHE_UPDATE_INTERVAL = 100  # 缓存更新间隔（轮数）


config = Config()
