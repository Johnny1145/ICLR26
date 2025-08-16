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

def generate_val_test_data(input_file: str, output_file: str, seed: int):
    """
    从现有的HDF5文件读取BBOB函数数据，为每个维度和函数随机选择一个偏移，
    生成验证集和测试集数据，每个函数采样30个点，并保存到新的HDF5文件。
    
    Args:
        input_file (str): 输入HDF5文件路径
        output_file (str): 输出HDF5文件路径
        seed (int): 随机种子
    """
    np.random.seed(seed)
    function_classes = [
        Sphere, Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector,
        StepEllipsoidal, RosenbrockRotated, Ellipsoidal, Discus, BentCigar,
        SharpRidge, DifferentPowers, Weierstrass, SchaffersF7,
        SchaffersF7IllConditioned, GriewankRosenbrock, Schwefel, Katsuura,
        Lunacek, Gallagher101Me, Gallagher21Me, NegativeSphere,
        NegativeMinDifference, FonsecaFleming
    ]
    function_dict = {cls.__name__: cls for cls in function_classes}
    
    # 创建输出HDF5文件
    with h5py.File(output_file, 'w') as f_out:
        # 读取输入HDF5文件
        with h5py.File(input_file, 'r') as f_in:
            # 对每个维度从2到6
            for dim in range(2, 7):
                dim_group_in = f_in[f"dim_{dim}"]
                dim_group_out = f_out.create_group(f"dim_{dim}")
                print(f"处理维度 {dim}...")
                # 对每类函数
                for func_name in dim_group_in.keys():
                    func_group_in = dim_group_in[func_name]
                    func_group_out = dim_group_out.create_group(func_name)
                    print(f"  处理函数 {func_name}...")
                    # 随机选择一个函数实例的偏移
                    x_val_data_all = []
                    y_val_data_all = []
                    selected_shift_all = []
                    for i in range(1000):
                        x_val_data = []
                        y_val_data = []
                        shift_data = func_group_in['shift'][:]
                        random_idx = np.random.randint(0, shift_data.shape[0])
                        selected_shift = shift_data[random_idx]
                        # 使用选定的偏移创建函数实例
                        func_class = function_dict[func_name]
                        func = func_class(dim)
                        if hasattr(func, 'c'):
                            func.c = selected_shift
                        # 采样30个点作为验证集
                        for _ in range(30):
                            x = np.random.uniform(-5, 5, dim)
                            y = func(x)
                            x_val_data.append(x)
                            y_val_data.append(float(y))
                        x_val_data_all.append(x_val_data)
                        y_val_data_all.append(y_val_data)
                        selected_shift_all.append(selected_shift)
                    # 保存数据到HDF5，格式与原始数据一致
                    func_group_out.create_dataset('x', data=np.array(x_val_data_all))
                    func_group_out.create_dataset('y', data=np.array(y_val_data_all))
                    func_group_out.create_dataset('shift', data=np.array(selected_shift_all))
                    func_group_out.attrs['function_name'] = func_name
                    func_group_out.attrs['dimension'] = dim
            
    print(f"验证集和测试集数据已成功保存到 {output_file}")

if __name__ == "__main__":
    generate_val_test_data("data/bbob_data.h5", "data/bbob_test_data.h5", 1145141919)
