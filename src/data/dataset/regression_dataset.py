import numpy as np
import json
import torch
from typing import List, Optional, Dict, Any, Union
from torch.utils.data import Dataset
from pathlib import Path
import os
from src.model.regress_lm import core
from src.model.regress_lm.vocabs import SentencePieceVocab


class RegressionDataset(Dataset):
    """读取回归数据集的.npy文件和info.json"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train", "val", "test"
        max_samples: Optional[int] = None,
        random_seed: int = 42,
        dataset_name: Optional[str] = None,  # 如果指定，只加载特定数据集
        compute_max_seq_len: bool = True  # 是否计算最大序列长度
    ):
        """
        初始化回归数据集
        
        Args:
            data_dir: 数据集目录路径
            split: 数据分割类型 ("train", "val", "test")
            max_samples: 最大样本数，如果为None则不限制
            random_seed: 随机种子
            dataset_name: 特定数据集名称，如果为None则加载所有数据集
            compute_max_seq_len: 是否计算sentence piece tokenizer下的最大序列长度
        """
        # 使用绝对路径来避免PYTHONPATH的影响
        if os.path.isabs(data_dir):
            self.data_dir = Path(data_dir)
        else:
            # 如果是相对路径，转换为绝对路径
            self.data_dir = Path(os.path.abspath(data_dir))
        self.split = split
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.compute_max_seq_len = compute_max_seq_len
        
        # 初始化tokenizer（如果需要计算序列长度）
        self.tokenizer = None
        if self.compute_max_seq_len:
            try:
                self.tokenizer = SentencePieceVocab.from_t5()
                print("成功加载SentencePiece tokenizer")
            except Exception as e:
                print(f"警告：无法加载SentencePiece tokenizer: {e}")
                print("将跳过序列长度计算")
                self.compute_max_seq_len = False
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载.npy文件和info.json"""
        print(f"正在加载数据集目录: {self.data_dir}")
        print(f"数据分割: {self.split}")
        
        if self.dataset_name:
            # 加载特定数据集
            self.load_single_dataset(self.dataset_name)
        else:
            # 加载所有数据集
            self.load_all_datasets()
    
    def load_single_dataset(self, dataset_name: str):
        """加载单个数据集"""
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"找不到数据集目录: {dataset_dir}")
        
        # 读取info.json
        info_file = dataset_dir / "info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"找不到info.json文件: {info_file}")
        
        with open(info_file, 'r', encoding='utf-8') as f:
            self.info = json.load(f)
        
        # 提取metadata
        self.metadata = {
            "name": self.info.get("name", dataset_name),
            "dimension": self.info.get("n_num_features", 0) + self.info.get("n_cat_features", 0)
        }
        if self.metadata["dimension"] > 200:
            print(f"跳过 {dataset_name}: 维度大于200")
            return
        
        # 获取特征名映射
        self.num_feature_names = self.info.get("num_feature_intro", {})
        self.cat_feature_names = self.info.get("cat_feature_intro", {})
        
        # 读取.npy文件
        x_num_file = dataset_dir / f"N_{self.split}.npy"
        x_cat_file = dataset_dir / f"C_{self.split}.npy"
        y_file = dataset_dir / f"y_{self.split}.npy"
        
        # 检查至少有一个特征文件存在
        if not x_num_file.exists() and not x_cat_file.exists():
            raise FileNotFoundError(f"找不到特征文件: {x_num_file} 或 {x_cat_file}")
        if not y_file.exists():
            raise FileNotFoundError(f"找不到标签文件: {y_file}")
        
        # 加载数据
        self.x_num_data = None
        self.x_cat_data = None
        
        if x_num_file.exists():
            self.x_num_data = np.load(x_num_file, allow_pickle=True)
            # 确保数据形状正确
            if len(self.x_num_data.shape) == 1:
                self.x_num_data = self.x_num_data.reshape(-1, 1)
        
        if x_cat_file.exists():
            self.x_cat_data = np.load(x_cat_file, allow_pickle=True)
            # 确保数据形状正确
            if len(self.x_cat_data.shape) == 1:
                self.x_cat_data = self.x_cat_data.reshape(-1, 1)
        
        self.y_data = np.load(y_file, allow_pickle=True)
        if len(self.y_data.shape) == 1:
            self.y_data = self.y_data.reshape(-1, 1)
        
        # 合并特征数据
        feature_parts = []
        if self.x_num_data is not None:
            feature_parts.append(self.x_num_data)
        if self.x_cat_data is not None:
            feature_parts.append(self.x_cat_data)
        
        if feature_parts:
            self.x_data = np.concatenate(feature_parts, axis=1)
        else:
            raise ValueError("没有找到任何特征数据")

        
        # 预处理特征字符串
        self.x_str_data = []
        for i in range(len(self.x_data)):
            self.x_str_data.append(self.format_features(i))
        
        # 计算最大序列长度
        # self.max_seq_len = None
        # if self.compute_max_seq_len and self.tokenizer is not None:
        #     max_len = 0
        #     for x_str in self.x_str_data:
        #         try:
        #             token_ids = self.tokenizer.to_token_ids(x_str)
        #             max_len = max(max_len, len(token_ids))
        #         except Exception as e:
        #             print(f"警告：计算序列长度时出错: {e}")
        #             continue
        #     self.max_seq_len = max_len
        #     print(f"  SentencePiece tokenizer下的最大序列长度: {self.max_seq_len}")
        
        print(f"数据加载完成:")
        print(f"  数据集名称: {self.metadata['name']}")
        print(f"  特征维度: {self.metadata['dimension']}")
        print(f"  样本数量: {len(self.x_data)}")
        print(f"  特征形状: {self.x_data.shape}")
        print(f"  标签形状: {self.y_data.shape}")
        if self.x_num_data is not None:
            print(f"  数值特征数量: {self.x_num_data.shape[1]}")
        if self.x_cat_data is not None:
            print(f"  分类特征数量: {self.x_cat_data.shape[1]}")
    
    def load_all_datasets(self):
        """加载所有数据集"""
        self.datasets = {}
        self.dataset_info = {}
        self.x_data = []
        self.y_data = []
        self.x_str_data = []

        # 遍历所有子目录
        max_dim = 0
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dataset_name = item.name
                
                # 检查是否有info.json文件
                info_file = item / "info.json"
                if not info_file.exists():
                    print(f"跳过 {dataset_name}: 没有info.json文件")
                    continue
                
                try:
                    # 读取info.json
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    
                    metadata = {
                        "name": info.get("name", dataset_name),
                        "dimension": info.get("n_num_features", 0) + info.get("n_cat_features", 0)
                    }

                    if metadata["dimension"] > 200:
                        print(f"跳过 {dataset_name}: 维度大于200")
                        continue
                    
                    # 检查是否是回归任务
                    if info.get("task_type") != "regression":
                        print(f"跳过 {dataset_name}: 不是回归任务")
                        continue
                    
                    # 检查是否有对应的.npy文件
                    x_num_file = item / f"N_{self.split}.npy"
                    x_cat_file = item / f"C_{self.split}.npy"
                    y_file = item / f"y_{self.split}.npy"
                    
                    if not x_num_file.exists() and not x_cat_file.exists():
                        print(f"跳过 {dataset_name}: 缺少{self.split}分割的.npy文件")
                        continue
                    if not y_file.exists():
                        print(f"跳过 {dataset_name}: 缺少{self.split}分割的y.npy文件")
                        continue
                    
                    # 加载数据
                    x_num_data = None
                    x_cat_data = None
                    
                    if x_num_file.exists():
                        x_num_data = np.load(x_num_file, allow_pickle=True)
                        # 确保数据形状正确
                        print(f"x_num_data.shape: {x_num_data.shape}")
                        if len(x_num_data.shape) == 1:
                            x_num_data = x_num_data.reshape(-1, 1)
                    
                    if x_cat_file.exists():
                        x_cat_data = np.load(x_cat_file, allow_pickle=True)
                        # 确保数据形状正确
                        print(f"x_cat_data.shape: {x_cat_data.shape}")
                        if len(x_cat_data.shape) == 1:
                            x_cat_data = x_cat_data.reshape(-1, 1)
                    
                    y_data = np.load(y_file, allow_pickle=True)
                    if len(y_data.shape) == 1:
                        y_data = y_data.reshape(-1, 1)
                    
                    # 合并特征数据
                    feature_parts = []
                    if x_num_data is not None:
                        feature_parts.append(x_num_data)
                    if x_cat_data is not None:
                        feature_parts.append(x_cat_data)
                    
                    if not feature_parts:
                        print(f"跳过 {dataset_name}: 没有找到任何特征数据")
                        continue
                    
                    x_data = np.concatenate(feature_parts, axis=1)
                    max_dim = max(max_dim, x_data.shape[1])
                    print(f"x_data.shape: {x_data.shape}")

                    
                    # 预处理特征字符串
                    num_feature_names = info.get("num_feature_intro", {})
                    cat_feature_names = info.get("cat_feature_intro", {})
                    x_str_data = []
                    
                    for i in range(len(x_data)):
                        feature_strs = []
                        feature_idx = 0
                        
                        # 处理数值特征
                        if x_num_data is not None:
                            for j in range(x_num_data.shape[1]):
                                feature_key = f"x{feature_idx + 1}"
                                feature_name = num_feature_names.get(feature_key, feature_key)
                                value = x_num_data[i, j]
                                feature_strs.append(f"{feature_name}: {float(value):.4f}")
                                feature_idx += 1
                        
                        # 处理分类特征
                        if x_cat_data is not None:
                            for j in range(x_cat_data.shape[1]):
                                feature_key = f"x{feature_idx + 1}"
                                feature_name = cat_feature_names.get(feature_key, feature_key)
                                value = x_cat_data[i, j]
                                # 分类特征可能是字符串，需要特殊处理
                                if isinstance(value, str):
                                    feature_strs.append(f"{feature_name}: {value}")
                                else:
                                    # 如果是数值，转换为字符串
                                    feature_strs.append(f"{feature_name}: {str(value)}")
                                feature_idx += 1
                        
                        feature_strs = ', '.join([f"{k}: {v}" for k, v in metadata.items()]) + ", " + ', '.join(feature_strs)
                        x_str_data.append(feature_strs)
                    
                    # 添加到全局列表
                    for i in range(len(x_data)):
                        self.x_data.append(x_data[i])
                        self.y_data.append(y_data[i])
                        self.x_str_data.append(x_str_data[i])
                    
                    # 存储数据集
                    self.datasets[dataset_name] = {
                        'x_data': x_data,
                        'y_data': y_data,
                        'x_str_data': x_str_data,
                        'x_num_data': x_num_data,
                        'x_cat_data': x_cat_data,
                        'info': info,
                        'metadata': metadata
                    }
                    
                    self.dataset_info[dataset_name] = {
                        "name": info.get("name", dataset_name),
                        "dimension": info.get("n_num_features", 0) + info.get("n_cat_features", 0),
                        "split": self.split,
                        "num_samples": len(x_data),
                        "feature_names": num_feature_names
                    }
                    
                    print(f"加载数据集: {dataset_name} - {len(x_data)} 样本")
                    
                except Exception as e:
                    print(f"加载数据集 {dataset_name} 时出错: {e}")
                    continue
        
        # 计算所有数据集的最大序列长度
        print(f"max_dim: {max_dim}")
        self.max_seq_len = None
        if self.compute_max_seq_len and self.tokenizer is not None:
            max_len = 0
            for x_str in self.x_str_data:
                try:
                    token_ids = self.tokenizer.to_token_ids(x_str)
                    max_len = max(max_len, len(token_ids))
                except Exception as e:
                    print(f"警告：计算序列长度时出错: {e}")
                    continue
            self.max_seq_len = max_len
            print(f"SentencePiece tokenizer下的最大序列长度: {self.max_seq_len}")
        print(f"\n总共加载了 {len(self.datasets)} 个数据集")
        
        # 如果没有加载任何数据集，抛出异常
        if not self.datasets:
            raise ValueError("没有找到任何有效的回归数据集")
    
    def format_features(self, idx: int) -> str:
        """将特征数组格式化为字符串，使用info.json中的特征名"""
        feature_strs = []
        feature_idx = 0
        
        # 处理数值特征
        if self.x_num_data is not None:
            for i in range(self.x_num_data.shape[1]):
                feature_key = f"x{feature_idx + 1}"
                feature_name = self.num_feature_names.get(feature_key, feature_key)
                value = self.x_num_data[idx, i]
                # 数值特征直接转换为float
                feature_strs.append(f"{feature_name}: {float(value):.4f}")
                feature_idx += 1
        
        # 处理分类特征
        if self.x_cat_data is not None:
            for i in range(self.x_cat_data.shape[1]):
                feature_key = f"x{feature_idx + 1}"
                feature_name = self.cat_feature_names.get(feature_key, feature_key)
                value = self.x_cat_data[idx, i]
                # 分类特征可能是字符串，需要特殊处理
                if isinstance(value, str):
                    feature_strs.append(f"{feature_name}: {value}")
                else:
                    # 如果是数值，转换为字符串
                    feature_strs.append(f"{feature_name}: {str(value)}")
                feature_idx += 1
        
        return ", ".join(feature_strs)
    
    def format_features_single(self, x: np.ndarray, feature_names: Dict[str, str]) -> str:
        """将特征数组格式化为字符串，使用指定的特征名映射"""
        feature_strs = []
        for i, value in enumerate(x):
            # 获取特征名，如果没有对应的映射就使用默认的x{i+1}
            feature_key = f"x{i+1}"
            feature_name = feature_names.get(feature_key, feature_key)
            feature_strs.append(f"{feature_name}: {float(value):.4f}")
        
        return ", ".join(feature_strs)
    
    def __len__(self):
        if self.dataset_name:
            return len(self.x_data)
        else:
            # 返回所有数据集的样本总数
            return sum(len(dataset['x_data']) for dataset in self.datasets.values())
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if self.dataset_name:
            # 单个数据集模式
            x = self.x_str_data[idx]
            y = self.y_data[idx]
            
            return core.Example(
                x=x,
                y=float(y[0]) if len(y.shape) > 0 else float(y)
            )
        else:
            # 多数据集模式 - 需要找到对应的数据集和索引
            x = self.x_str_data[idx]
            y = self.y_data[idx]
            return core.Example(
                x=x,
                y=float(y[0]) if len(y.shape) > 0 else float(y)
            )

    def get_item_with_metadata(self, idx):
        """获取包含metadata的样本"""
        if self.dataset_name:
            # 单个数据集模式
            x = self.x_str_data[idx]
            y = self.y_data[idx]
            
            return {
                'x': x,
                'y': float(y[0]) if len(y.shape) > 0 else float(y),
                'metadata': self.metadata,
                'x_raw': self.x_data[idx],
                'y_raw': y,
                'dataset_name': self.dataset_name,
                'x_num_raw': self.x_num_data[idx] if self.x_num_data is not None else None,
                'x_cat_raw': self.x_cat_data[idx] if self.x_cat_data is not None else None
            }
        else:
            # 多数据集模式
            current_idx = 0
            for dataset_name, dataset in self.datasets.items():
                dataset_size = len(dataset['x_data'])
                if current_idx + dataset_size > idx:
                    # 找到对应的数据集
                    local_idx = idx - current_idx
                    x = dataset['x_str_data'][local_idx]
                    y = dataset['y_data'][local_idx]
                    
                    return {
                        'x': x,
                        'y': float(y[0]) if len(y.shape) > 0 else float(y),
                        'metadata': dataset['metadata'],
                        'x_raw': dataset['x_data'][local_idx],
                        'y_raw': y,
                        'dataset_name': dataset_name,
                        'x_num_raw': dataset['x_num_data'][local_idx] if dataset['x_num_data'] is not None else None,
                        'x_cat_raw': dataset['x_cat_data'][local_idx] if dataset['x_cat_data'] is not None else None
                    }
                current_idx += dataset_size
            
            raise IndexError(f"索引 {idx} 超出范围")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        if self.dataset_name:
            return {
                "name": self.metadata["name"],
                "dimension": self.metadata["dimension"],
                "split": self.split,
                "num_samples": len(self.x_data),
                "num_feature_names": self.num_feature_names,
                "cat_feature_names": self.cat_feature_names,
                "info": self.info
            }
        else:
            return {
                "total_datasets": len(self.datasets),
                "split": self.split,
                "total_samples": sum(len(dataset['x_data']) for dataset in self.datasets.values()),
                "datasets": self.dataset_info
            }
    
    def get_dataset_names(self) -> List[str]:
        """获取所有数据集名称"""
        if self.dataset_name:
            return [self.dataset_name]
        else:
            return list(self.datasets.keys())
    
    def get_max_seq_len(self) -> Optional[int]:
        """获取SentencePiece tokenizer下的最大序列长度"""
        return self.max_seq_len


def load_regression_dataset(
    data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    dataset_name: Optional[str] = None,
    compute_max_seq_len: bool = True
) -> RegressionDataset:
    """
    便捷函数：加载回归数据集
    
    Args:
        data_dir: 数据集目录路径
        split: 数据分割类型
        max_samples: 最大样本数
        random_seed: 随机种子
        dataset_name: 特定数据集名称，如果为None则加载所有数据集
        compute_max_seq_len: 是否计算sentence piece tokenizer下的最大序列长度
    
    Returns:
        RegressionDataset实例
    """
    return RegressionDataset(
        data_dir=data_dir,
        split=split,
        max_samples=max_samples,
        random_seed=random_seed,
        dataset_name=dataset_name,
        compute_max_seq_len=compute_max_seq_len
    )


if __name__ == "__main__":
    
    # 测试代码 - 加载所有数据集
    print("\n=== 测试所有数据集 ===")
    dataset_all = RegressionDataset(
        data_dir="data/regression_data",
        split="test",
        max_samples=2,  # 每个数据集最多2个样本
        random_seed=42
    )
    
    print("\n所有数据集信息:")
    info = dataset_all.get_dataset_info()
    print(f"  总数据集数: {info['total_datasets']}")
    print(f"  总样本数: {info['total_samples']}")
    print(f"  数据分割: {info['split']}")
    if info.get("max_seq_len") is not None:
        print(f"  SentencePiece tokenizer下的最大序列长度: {info['max_seq_len']}")
    
    print("\n数据集列表:")
    for name, details in info['datasets'].items():
        print(f"  {name}: {details['num_samples']} 样本, {details['dimension']} 维度")
    
    print(f"\n前5个样本 (来自不同数据集):")
    for i in range(min(5, len(dataset_all))):
        sample = dataset_all.get_item_with_metadata(i)
        print(f"样本 {i+1} (来自 {sample['dataset_name']}):")
        print(f"  x: {sample['x'][:100]}...")  # 只显示前100个字符
        print(f"  y: {sample['y']:.6f}")
        print()
