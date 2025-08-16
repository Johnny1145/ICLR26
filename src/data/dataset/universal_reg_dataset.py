import json
import os
import re
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
from src.model.regress_lm import core


class FunctionTrainingDataset(Dataset):
    """读取function_training_data.json文件中的函数训练数据"""
    
    def __init__(
        self,
        json_file_path: str = "function_training_data.json",
        function_types: Optional[List[str]] = None,
        max_samples_per_type: Optional[int] = None,
        random_seed: int = 42,
        load_subset: bool = False,
    ):
        """
        初始化函数训练数据集
        
        Args:
            json_file_path: JSON文件路径
            function_types: 要包含的函数类型列表，如果为None则包含所有类型
            max_samples_per_type: 每种函数类型的最大样本数，如果为None则不限制
            random_seed: 随机种子
        """
        self.json_file_path = json_file_path
        self.function_types = function_types
        self.max_samples_per_type = max_samples_per_type
        self.random_seed = random_seed
        self.load_subset = load_subset
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 读取和处理数据
        if self.load_subset:
            self.load_data_subset()
        else:
            self.load_data()
    
    def load_data(self):
        """加载和处理JSON数据"""
        print(f"正在加载数据文件: {self.json_file_path}")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 按函数类型分组数据
        self.data_by_type = {}
        
        for item in raw_data:
            func_type = item["type"]
            
            # 如果指定了函数类型过滤，则跳过不在列表中的类型
            if self.function_types is not None and func_type not in self.function_types:
                continue
            
            if func_type not in self.data_by_type:
                self.data_by_type[func_type] = []
            
            # 提取metadata和data
            metadata = item["function"]["metadata"]
            x_data = item["function"]["data"]["x"]
            y_data = item["function"]["data"]["y"]
            params_data = item["function"]["data"]["params"]
            
            # 将x和y配对，每个样本都包含对应的metadata
            for x, y in zip(x_data, y_data):
                self.data_by_type[func_type].append({
                    "x": x,
                    "y": y,
                    "params": params_data,
                    "metadata": metadata  # 每个样本都有自己的metadata
                })
        
        # 限制每种类型的样本数
        if self.max_samples_per_type is not None:
            for func_type in self.data_by_type:
                if len(self.data_by_type[func_type]) > self.max_samples_per_type:
                    # 随机采样
                    indices = np.random.choice(
                        len(self.data_by_type[func_type]),
                        self.max_samples_per_type,
                        replace=False
                    )
                    self.data_by_type[func_type] = [
                        self.data_by_type[func_type][i] for i in indices
                    ]
        
        # 合并所有数据
        self.x_values = []
        self.x_num_values = []
        self.y_values = []
        self.normalized_y_values = []
        self.function_types_list = []
        self.metadata_list = []
        self.params_list = []
        self.function_ids_list = []
        
        for func_type, data_list in self.data_by_type.items():
            for item in data_list:
                metadata = item["metadata"]  # 使用每个样本自己的metadata
                str_metadata = json.dumps(metadata)
                str_x = str_metadata + " x:" + str(item["x"])
                self.x_values.append(str_x)
                self.x_num_values.append(item["x"])
                self.y_values.append(item["y"])
                self.params_list.append(item["params"])
                self.function_types_list.append(func_type)
                self.metadata_list.append(metadata)
                self.function_ids_list.append(metadata["function_id"])
        # print(self.function_ids_list)
        # assert 0
        
        print(f"数据加载完成:")
        for func_type in self.data_by_type:
            print(f"  {func_type}: {len(self.data_by_type[func_type])} 个样本")
        print(f"总样本数: {len(self.x_values)}")
        # assert 0

    def load_data_subset(self):
        """加载数据，但只加载func_id为1-5的任务"""
        print(f"正在加载数据文件: {self.json_file_path}")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 按函数类型分组数据
        self.data_by_type = {}
        
        for item in raw_data:
            func_type = item["type"]
            
            # 如果指定了函数类型过滤，则跳过不在列表中的类型
            if self.function_types is not None and func_type not in self.function_types:
                continue
            
            if func_type not in self.data_by_type:
                self.data_by_type[func_type] = []
            
            # 提取metadata和data
            metadata = item["function"]["metadata"]
            x_data = item["function"]["data"]["x"]
            y_data = item["function"]["data"]["y"]
            params_data = item["function"]["data"]["params"]
            
            # 将x和y配对，每个样本都包含对应的metadata
            for x, y in zip(x_data, y_data):
                self.data_by_type[func_type].append({
                    "x": x,
                    "y": y,
                    "params": params_data,
                    "metadata": metadata  # 每个样本都有自己的metadata
                })
        
        # 限制每种类型的样本数
        if self.max_samples_per_type is not None:
            for func_type in self.data_by_type:
                if len(self.data_by_type[func_type]) > self.max_samples_per_type:
                    # 随机采样
                    indices = np.random.choice(
                        len(self.data_by_type[func_type]),
                        self.max_samples_per_type,
                        replace=False
                    )
                    self.data_by_type[func_type] = [
                        self.data_by_type[func_type][i] for i in indices
                    ]
        
        # 合并所有数据
        self.x_values = []
        self.x_num_values = []
        self.y_values = []
        self.normalized_y_values = []
        self.function_types_list = []
        self.metadata_list = []
        self.params_list = []
        self.function_ids_list = []
        
        for func_type, data_list in self.data_by_type.items():
            for item in data_list:
                metadata = item["metadata"]  # 使用每个样本自己的metadata
                if metadata["function_id"][-1] not in ['1', '2', '3', '4', '5']:
                    continue
                str_metadata = json.dumps(metadata)
                str_x = str_metadata + " x:" + str(item["x"])
                self.x_values.append(str_x)
                self.x_num_values.append(item["x"])
                self.y_values.append(item["y"])
                self.params_list.append(item["params"])
                self.function_types_list.append(func_type)
                self.metadata_list.append(metadata)
                self.function_ids_list.append(metadata["function_id"])
        # print(self.function_ids_list)
        # assert 0
        
        print(f"数据加载完成:")
        for func_type in self.data_by_type:
            print(f"  {func_type}: {len(self.data_by_type[func_type])} 个样本")
        print(f"总样本数: {len(self.x_values)}")
    
    def get_function_types(self) -> List[str]:
        """获取所有函数类型"""
        return list(self.data_by_type.keys())
    
    def get_metadata_by_type(self) -> Dict[str, List[Dict]]:
        """获取每种函数类型的metadata列表"""
        metadata_by_type = {}
        for func_type, data_list in self.data_by_type.items():
            metadata_by_type[func_type] = []
            seen_function_ids = set()
            for item in data_list:
                metadata = item["metadata"]
                function_id = metadata["function_id"]
                if function_id not in seen_function_ids:
                    metadata_by_type[func_type].append(metadata)
                    seen_function_ids.add(function_id)
        return metadata_by_type
    
    def get_data_by_type(self) -> Dict[str, List[Dict]]:
        """获取按类型分组的数据"""
        return self.data_by_type.copy()
    
    def __len__(self):
        return len(self.x_values)
    
    def __getitem__(self, idx):
        return core.Example(
            x=self.x_values[idx],
            y=self.y_values[idx]
        )
    
    def get_item_with_metadata(self, idx):
        """获取包含metadata的样本"""
        return {
            "x": self.x_values[idx],
            "x_num": self.x_num_values[idx],
            "y": self.y_values[idx],
            "params": self.params_list[idx],
            "function_type": self.function_types_list[idx],
            "function_id": self.function_ids_list[idx],
            "metadata": self.metadata_list[idx]
        }
    
    def save_to_json(self, file_path: str, include_metadata: bool = False):
        """将数据集保存为JSON格式"""
        data = []
        for i in range(len(self)):
            if include_metadata:
                item = self.get_item_with_metadata(i)
                data.append({
                    "x": item["x"],
                    "y": item["y"],
                    "function_type": item["function_type"],
                    "metadata": item["metadata"]
                })
            else:
                item = self[i]
                data.append({
                    "x": item.x,
                    "y": item.y
                })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"数据集已保存到: {file_path}")


if __name__ == "__main__":
    print("函数训练数据集示例:")
    
    # 测试函数训练数据集
    function_dataset = FunctionTrainingDataset(
        json_file_path="function_training_data.json",
        function_types=None,  # 只加载这两种类型
        max_samples_per_type=None,  # 每种类型最多100个样本
        random_seed=42
    )
    
    # 显示前5个样本
    for i in range(5):
        sample = function_dataset.get_item_with_metadata(i)
        print(f"样本 {i+1}: x={sample['x']}, y={sample['y']:.8f}, 类型={sample['function_type']}")
    
    # 显示每种类型的metadata
    print("\n函数类型metadata:")
    metadata_by_type = function_dataset.get_metadata_by_type()
    for func_type, metadata_list in metadata_by_type.items():
        print(f"{func_type}: {metadata_list}")
    
    # 保存处理后的数据
    function_dataset.save_to_json("processed_function_data.json", include_metadata=True)
