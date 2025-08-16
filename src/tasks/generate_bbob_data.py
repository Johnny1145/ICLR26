import h5py
import numpy as np
from tqdm import tqdm
from src.tasks.functions.bbob import (
    Sphere, Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector,
    StepEllipsoidal, RosenbrockRotated, Ellipsoidal, Discus, BentCigar,
    SharpRidge, DifferentPowers, Weierstrass, SchaffersF7,
    SchaffersF7IllConditioned, GriewankRosenbrock, Schwefel, Katsuura,
    Lunacek, Gallagher101Me, Gallagher21Me, NegativeSphere,
    NegativeMinDifference, FonsecaFleming
)

def generate_bbob_data(output_file: str, seed: int):
    """
    为BBOB函数从2D到6D生成数据，每个维度生成8000个函数，每个函数采样30个点，
    并将结果保存为HDF5格式，包含采样点对和metadata。
    
    Args:
        output_file (str): 输出HDF5文件路径
    """
    # 定义函数类列表
    np.random.seed(seed)
    function_classes = [
        Sphere, Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector,
        StepEllipsoidal, RosenbrockRotated, Ellipsoidal, Discus, BentCigar,
        SharpRidge, DifferentPowers, Weierstrass, SchaffersF7,
        SchaffersF7IllConditioned, GriewankRosenbrock, Schwefel, Katsuura,
        Lunacek, Gallagher101Me, Gallagher21Me, NegativeSphere,
        NegativeMinDifference, FonsecaFleming
    ]
    
    # 创建HDF5文件
    with h5py.File(output_file, 'w') as f:
        # 对每个维度从2到6
        for dim in range(2, 7):
            dim_group = f.create_group(f"dim_{dim}")
            print(f"处理维度 {dim}...")
            # 对每类函数
            for func_class in function_classes:
                func_name = func_class.__name__
                func_group = dim_group.create_group(func_name)
                print(f"  处理函数 {func_name}...")
                # 使用tqdm创建进度条，生成8000个函数实例
                x_data_all = []
                y_data_all = []
                shift_data_all = []
                for _ in tqdm(range(1), desc=f"    生成 {func_name} 函数实例"):
                    func = func_class(dim)
                    # 获取shift大小
                    shift = func.c.tolist() if hasattr(func.c, 'tolist') else func.c
                    # 采样30个点
                    x_data = []
                    y_data = []
                    for _ in range(30):
                        x = np.random.uniform(-5, 5, dim)  # BBOB函数通常在[-5, 5]范围内采样
                        y = func(x)
                        x_data.append(x)
                        y_data.append(float(y))
                    x_data_all.append(x_data)
                    y_data_all.append(y_data)
                    shift_data_all.append(shift)
                # 保存数据到HDF5
                func_group.create_dataset('x', data=np.array(x_data_all))
                func_group.create_dataset('y', data=np.array(y_data_all))
                func_group.create_dataset('shift', data=np.array(shift_data_all))
                # 保存metadata
                func_group.attrs['function_name'] = func_name
                func_group.attrs['dimension'] = dim
    
    print(f"数据已成功保存到 {output_file}")

if __name__ == "__main__":
    generate_bbob_data("data/bbob_test_data.h5", 1145141919)
