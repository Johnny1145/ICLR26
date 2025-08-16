import h5py
import numpy as np
import torch
import yaml
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model.regress_lm import core


class BBOBDataset(Dataset):
    """读取BBOB数据集的HDF5文件"""
    
    def __init__(
        self,
        hdf5_file_path: str = "data/bbob_data.h5",
        function_types: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        max_samples_per_type: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        初始化BBOB数据集
        
        Args:
            hdf5_file_path: HDF5文件路径
            function_types: 要包含的函数类型列表，如果为None则包含所有类型
            dimensions: 要包含的维度列表，如果为None则包含所有维度
            max_samples_per_type: 每种函数类型的最大样本数，如果为None则不限制
            random_seed: 随机种子
        """
        self.hdf5_file_path = hdf5_file_path
        self.function_types = function_types
        self.dimensions = dimensions if dimensions else list(range(2, 7))
        self.max_samples_per_type = max_samples_per_type
        self.random_seed = random_seed
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        self.load_data()
    
    def load_data(self):
        """加载和处理HDF5数据"""
        print(f"正在加载数据文件: {self.hdf5_file_path}")
        
        self.data_by_type = {}
        self.x_values = []
        self.x_num_values = []
        self.y_values = []
        self.function_types_list = []
        self.metadata_list = []
        
        try:
            with h5py.File(self.hdf5_file_path, 'r', locking=False) as f:
                for dim_str in f.keys():
                    dim = int(dim_str.split('_')[1])
                    if dim not in self.dimensions:
                        continue
                    dim_group = f[dim_str]
                    for func_type in dim_group.keys():
                        if self.function_types is not None and func_type not in self.function_types:
                            continue
                        func_group = dim_group[func_type]
                        x_data = func_group['x'][:]
                        y_data = func_group['y'][:]
                        shift_data = func_group['shift'][:]
                        metadata = {
                            'function_name': func_group.attrs['function_name'],
                            'dimension': func_group.attrs['dimension'],
                        }
                        
                        if func_type not in self.data_by_type:
                            self.data_by_type[func_type] = []
                        
                        for i in range(len(x_data)):
                            new_metadata = metadata.copy()
                            new_metadata['shift'] = [float(s) for s in shift_data[i]]
                            new_metadata['shift'] = '[' + ', '.join([f'{s:.2f}' for s in new_metadata['shift']]) + ']'
                            for j in range(len(x_data[i])):
                                str_x = ', '.join([f'x{k}: {float(x_data[i][j][k]):.4f}' for k in range(len(x_data[i][j]))])
                                str_x = ', '.join([f'{k}: {v}' for k, v in new_metadata.items()]) + ", " + str_x
                                self.data_by_type[func_type].append({
                                    'x': str_x,
                                    'x_num': x_data[i][j],
                                    'y': y_data[i][j],
                                    'metadata': new_metadata
                                })
                                self.x_values.append(str_x)
                                self.y_values.append(y_data[i][j])
                                self.x_num_values.append(x_data[i][j])
                                self.function_types_list.append(func_type)
                                self.metadata_list.append(new_metadata)
        except BlockingIOError as e:
            print(f"无法 无法打开文件: {e}")
            print("请确保没有其他进程正在使用该文件，或者尝试重新启动程序。")
            raise
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            raise
        
        # 限制每种类型的样本数
        if self.max_samples_per_type is not None:
            for func_type in self.data_by_type:
                if len(self.data_by_type[func_type]) > self.max_samples_per_type:
                    indices = np.random.choice(
                        len(self.data_by_type[func_type]),
                        self.max_samples_per_type,
                        replace=False
                    )
                    self.data_by_type[func_type] = [
                        self.data_by_type[func_type][i] for i in indices
                    ]
        
        print(f"数据加载完成:")
        for func_type in self.data_by_type:
            print(f"  {func_type}: {len(self.data_by_type[func_type])} 个样本")
        print(f"总样本数: {len(self.x_values)}")
    
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
            'x': self.x_values[idx],
            'y': self.y_values[idx],
            'function_type': self.function_types_list[idx],
            'metadata': self.metadata_list[idx]
        }

if __name__ == "__main__":
    dataset = BBOBDataset(
        hdf5_file_path="data/bbob_data.h5",
        function_types=None,
        dimensions=None,
        max_samples_per_type=None,
        random_seed=42
    )
    for i in range(5):
        sample = dataset.get_item_with_metadata(i)
        print(f"样本 {i+1}: x={sample['x']} y={sample['y']:.8f}, 类型={sample['function_type'], sample['metadata']}")
