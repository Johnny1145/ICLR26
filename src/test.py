import os
import time

import pandas as pd

import pmlb
from pmlb import regression_dataset_names

for name in regression_dataset_names:
    print(name)
    success = False
    while not success:
        try:
            path = f"./pmlb/datasets/{name}/{name}.tsv.gz"
            # if os.path.exists(path):
            #     os.remove(path)
            X, y = pmlb.fetch_data(
                name, return_X_y=True, local_cache_dir="./pmlb/datasets"
            )
            print(X.shape, y.shape)
            success = True
        except Exception as e:
            print(e)
            time.sleep(1)
exit()
# 使用 PMLB API 直接获取数据，而不是读取 Git LFS 文件
# 获取所有可用的回归数据集名称
print("可用的回归数据集:")
print(regression_dataset_names[:10])  # 显示前10个数据集名称

X, y = pmlb.fetch_data(
    regression_dataset_names[0], return_X_y=True, local_cache_dir="./pmlb/datasets"
)
print(X.shape, y.shape)
exit()

# 直接获取 cmc 数据集
try:
    # 方法1: 使用 fetch_data 获取 X 和 y
    X, y = pmlb.fetch_data("cmc", return_X_y=True)
    print(f"\n使用 fetch_data 获取的数据:")
    print(f"X 形状: {X.shape}")
    print(f"y 形状: {y.shape}")
    print(f"X 前5行:\n{X[:5]}")
    print(f"y 前5个值: {y[:5]}")

    # 方法2: 获取完整的 DataFrame
    df = pmlb.fetch_data("cmc")
    print(f"\n使用 fetch_data 获取的完整 DataFrame:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())

    # 查看数据类型
    print("\n数据类型:")
    print(df.dtypes)

    # 基本统计信息
    print("\n基本统计:")
    print(df.describe())

except Exception as e:
    print(f"获取 cmc 数据集时出错: {e}")

    # 尝试其他数据集
    print("\n尝试获取其他数据集...")
    available_datasets = pmlb.dataset_names
    print(f"所有可用数据集数量: {len(available_datasets)}")

    # 查找包含 'cmc' 的数据集
    cmc_datasets = [name for name in available_datasets if "cmc" in name.lower()]
    print(f"包含 'cmc' 的数据集: {cmc_datasets}")

    if cmc_datasets:
        try:
            df = pmlb.fetch_data(cmc_datasets[0])
            print(f"\n成功获取数据集: {cmc_datasets[0]}")
            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            print("\n前5行数据:")
            print(df.head())
        except Exception as e2:
            print(f"获取 {cmc_datasets[0]} 时出错: {e2}")
