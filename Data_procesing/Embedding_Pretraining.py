import pandas as pd
import numpy as np
import torch
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)


class ItemEmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """初始化embedding生成器"""
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device
        print(f"使用模型: {model_name}，计算设备: {self.device}")

        # 存储结果
        self.embeddings = None
        self.asin_to_index = None

        # 获取模型输出维度并固定
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"模型输出维度: {self.embedding_dim}")

    def process_category(self, category_list):
        """处理category字段，生成固定维度的平均embedding"""
        # 确保返回向量维度始终为self.embedding_dim
        try:
            if not category_list or pd.isna(category_list):
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # 确保输入是列表
            if isinstance(category_list, str):
                try:
                    # 尝试解析字符串为列表
                    category_list = eval(category_list)
                    # 确保解析后是列表类型
                    if not isinstance(category_list, list):
                        category_list = [category_list]
                except:
                    category_list = [category_list]

            # 如果解析后不是列表，转为列表
            if not isinstance(category_list, list):
                category_list = [str(category_list)]

            # 过滤空字符串
            category_list = [cat.strip() for cat in category_list if cat.strip()]

            if not category_list:  # 如果列表为空，返回零向量
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # 生成每个名词短语的embedding并平均
            embeddings = self.model.encode(category_list, convert_to_numpy=True, show_progress_bar=False)
            avg_emb = np.mean(embeddings, axis=0)

            # 确保输出维度正确
            if avg_emb.shape[0] != self.embedding_dim:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            return avg_emb

        except Exception as e:
            # 任何异常情况下都返回固定维度的零向量
            print(f"处理category时出错: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def process_text_field(self, text):
        """处理title和description字段，生成固定维度的句子平均embedding"""
        try:
            if not text or pd.isna(text):
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # 简单句子分割
            sentences = [s.strip() for s in str(text).split('.') if s.strip()]
            if not sentences:  # 如果没有句子，直接编码整个文本
                emb = self.model.encode([str(text)], convert_to_numpy=True, show_progress_bar=False)
                return emb[0]

            # 生成每个句子的embedding并平均
            embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            avg_emb = np.mean(embeddings, axis=0)

            return avg_emb if avg_emb.shape[0] == self.embedding_dim else np.zeros(self.embedding_dim, dtype=np.float32)
        except Exception as e:
            print(f"处理文本时出错: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def process_rank(self, rank_value):
        """处理rank字段，生成固定维度的数字embedding"""
        try:
            if rank_value is None or pd.isna(rank_value):
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # 将数字转换为字符串后编码
            if isinstance(rank_value, list):
                rank_str = ', '.join(map(str, rank_value))
            else:
                rank_str = str(rank_value)

            emb = self.model.encode([rank_str], convert_to_numpy=True, show_progress_bar=False)
            return emb[0] if emb.shape[1] == self.embedding_dim else np.zeros(self.embedding_dim, dtype=np.float32)
        except Exception as e:
            print(f"处理rank时出错: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def reduce_dimension(self, combined_embeddings, target_dim=384):
        """使用PCA将拼接后的高维向量降维"""
        print(f"将embedding从 {combined_embeddings.shape[1]} 维降维至 {target_dim} 维...")

        # 标准化
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(combined_embeddings)

        # PCA降维
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(scaled_embeddings)

        print(f"降维后解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        return reduced_embeddings

    def generate_embeddings(self, input_path, output_embedding_path,
                            output_index_path, target_dim=384):
        """生成物品embedding的主函数"""
        # 读取数据
        print(f"读取数据: {input_path}")
        df = pd.read_csv(input_path)

        # 确保只处理唯一的asin
        unique_items = df.drop_duplicates('asin').copy()
        print(f"共发现 {len(unique_items)} 个唯一物品")

        # 创建asin到索引的映射
        self.asin_to_index = {asin: idx for idx, asin in enumerate(unique_items['asin'].tolist())}

        # 存储各字段的embedding
        category_embeddings = []
        title_embeddings = []
        rank_embeddings = []
        description_embeddings = []

        # 处理每个物品，显示进度
        for idx, (_, row) in enumerate(tqdm(unique_items.iterrows(), total=len(unique_items), desc="生成embedding")):
            # 处理category
            cat_emb = self.process_category(row['category'])
            category_embeddings.append(cat_emb)

            # 处理title
            title_emb = self.process_text_field(row['title'])
            title_embeddings.append(title_emb)

            # 处理rank
            rank_emb = self.process_rank(row['rank'])
            rank_embeddings.append(rank_emb)

            # 处理description
            desc_emb = self.process_text_field(row['description'])
            description_embeddings.append(desc_emb)

            # 每处理1000个物品检查一次维度是否一致
            if idx % 1000 == 0 and idx > 0:
                assert all(len(emb) == self.embedding_dim for emb in
                           category_embeddings[-1000:]), f"第{idx}个物品的category维度不一致"
                assert all(len(emb) == self.embedding_dim for emb in
                           title_embeddings[-1000:]), f"第{idx}个物品的title维度不一致"

        # 转换为numpy数组（现在应该不会报错了）
        try:
            category_embeddings = np.array(category_embeddings, dtype=np.float32)
            title_embeddings = np.array(title_embeddings, dtype=np.float32)
            rank_embeddings = np.array(rank_embeddings, dtype=np.float32)
            description_embeddings = np.array(description_embeddings, dtype=np.float32)

            # 验证所有数组维度
            print(f"category_embeddings形状: {category_embeddings.shape}")
            print(f"title_embeddings形状: {title_embeddings.shape}")
            print(f"rank_embeddings形状: {rank_embeddings.shape}")
            print(f"description_embeddings形状: {description_embeddings.shape}")
        except ValueError as e:
            print(f"转换为numpy数组时出错: {e}")
            # 查找有问题的索引
            for i, emb in enumerate(category_embeddings):
                if len(emb) != self.embedding_dim:
                    print(f"问题索引 {i}: 维度 {len(emb)} (预期 {self.embedding_dim})")
            return None, None

        # 拼接所有字段的embedding
        combined_embeddings = np.concatenate([
            category_embeddings,
            title_embeddings,
            rank_embeddings,
            description_embeddings
        ], axis=1)

        # 降维
        self.embeddings = self.reduce_dimension(combined_embeddings, target_dim)

        # 保存结果
        print(f"保存embedding至: {output_embedding_path}")
        np.save(output_embedding_path, self.embeddings)

        print(f"保存asin索引字典至: {output_index_path}")
        with open(output_index_path, 'w', encoding='utf-8') as f:
            json.dump(self.asin_to_index, f, ensure_ascii=False, indent=2)

        print("所有处理完成!")
        return self.embeddings, self.asin_to_index



if __name__ == "__main__":
    # 配置文件路径
    INPUT_CSV = r"D:\CODE\MICM\Dataset\Amazon_BT\AB_and_AT\BT_meta_Books_5cores_cleaned.csv"
    OUTPUT_EMBEDDING = r"D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Books_initial_embeddings.npy"
    OUTPUT_INDEX = r"D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Books_asin_to_index.json"
    # INPUT_CSV = r"D:\CODE\MICM\Dataset\Amazon_BT\AB_and_AT\BT_meta_Toys_and_Games_5cores_cleaned.csv"
    # OUTPUT_EMBEDDING = r"D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Toys_and_Games_initial_embeddings.npy"
    # OUTPUT_INDEX = r"D:\CODE\MICM\Dataset\Amazon_BT\Data_Processed\Toys_and_Games_asin_to_index.json"

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDING), exist_ok=True)

    # 初始化生成器，可根据需要更换模型
    generator = ItemEmbeddingGenerator(model_name='all-MiniLM-L6-v2')

    # 生成embedding，目标维度设为384
    generator.generate_embeddings(
        input_path=INPUT_CSV,
        output_embedding_path=OUTPUT_EMBEDDING,
        output_index_path=OUTPUT_INDEX,
        target_dim=384
    )
