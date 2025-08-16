import json
import numpy as np
from src.tasks.functions.bbob import (
    Sphere, Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector,
    StepEllipsoidal, RosenbrockRotated, Ellipsoidal, Discus, BentCigar,
    SharpRidge, DifferentPowers, Weierstrass, SchaffersF7,
    SchaffersF7IllConditioned, GriewankRosenbrock, Schwefel, Katsuura,
    Lunacek, Gallagher101Me, Gallagher21Me, NegativeSphere,
    NegativeMinDifference, FonsecaFleming
)

def generate_bbob_data(output_file: str):
    """
    为BBOB函数从2D到6D生成数据，每个维度生成8000个函数，每个函数采样30个点，
    并将结果保存为JSON格式，包含采样点对和metadata。
    
    Args:
        output_file (str): 输出JSON文件路径
    """
    # 定义函数类列表
    function_classes = [
        Sphere, Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector,
        StepEllipsoidal, RosenbrockRotated, Ellipsoidal, Discus, BentCigar,
        SharpRidge, DifferentPowers, Weierstrass, SchaffersF7,
        SchaffersF7IllConditioned, GriewankRosenbrock, Schwefel, Katsuura,
        Lunacek, Gallagher101Me, Gallagher21Me, NegativeSphere,
        NegativeMinDifference, FonsecaFleming
    ]
    
    # 初始化结果字典
    results = {}
    
    # 对每个维度从2到6
    for dim in range(2, 7):
        results[f"dim_{dim}"] = {}
        # 对每类函数
        for func_class in function_classes:
            func_name = func_class.__name__
            results[f"dim_{dim}"][func_name] = []
            # 生成8000个函数实例
            for _ in range(8000):
                func = func_class(dim)
                # 获取shift大小
                shift = func.c.tolist() if hasattr(func.c, 'tolist') else func.c
                # 采样30个点
                x_data = []
                y_data = []
                for _ in range(30):
                    x = np.random.uniform(-5, 5, dim)  # BBOB函数通常在[-5, 5]范围内采样
                    y = func(x)
                    x_data.append(x.tolist())
                    y_data.append(float(y))
                # 构建metadata
                metadata = {
                    "function_name": func_name,
                    "dimension": dim,
                    "shift": shift
                }
                # 保存实例数据
                instance_data = {
                    "metadata": metadata,
                    "x": x_data,
                    "y": y_data
                }
                results[f"dim_{dim}"][func_name].append(instance_data)
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"数据已成功保存到 {output_file}")

if __name__ == "__main__":
    generate_bbob_data("bbob_data.json")
