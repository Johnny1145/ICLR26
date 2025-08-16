import logging
from typing import Dict

import hydra
import numpy as np
import rootutils
import swanlab
import torch
import json
import os
from datetime import datetime
from accelerate.utils import tqdm as accelerate_tqdm
from omegaconf import DictConfig
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from .data.universal_reg_dataset import FunctionTrainingDataset
from .model.base_module import BaseModule
from .model.regress_lm import core
from .model.regress_lm.models.pytorch import model as torch_model_lib
from .model.regress_lm.tokenizers import P10Tokenizer
from .model.regress_lm.vocabs import DecoderVocab, SentencePieceVocab
from src.utils.number_token_loss import NumberTokenLoss
from src.utils.checkpoint import CheckpointManager
from .data.dataset.bbob_dataset import BBOBDataset

# Initialize vocabs
encoder_vocab = SentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=P10Tokenizer())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def collate_fn(examples, model):
    tensor_examples = model.convert_examples(examples)
    return tensor_examples


class RegressionModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        model = torch_model_lib.PyTorchModel(
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            max_input_len=cfg.model.max_input_len,
            max_num_objs=cfg.model.max_num_objs,
            d_model=cfg.model.d_model,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
        )

        criterion = None

        super().__init__(
            model=model,
            criterion=criterion,
            project_name=cfg.project_name,
            experiment_name=cfg.experiment_name,
            use_wandb=cfg.use_wandb,
            log_dir=cfg.log_dir,
        )
        self.cfg = cfg
        self.total_steps = None

    def set_total_steps(self, total_steps: int):
        """Set total training steps for scheduler configuration"""
        self.total_steps = total_steps

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        
        if not hasattr(self.cfg, 'scheduler') or self.total_steps is None:
            return {"optimizer": optimizer}
        
        # Create scheduler based on configuration
        scheduler_config = self.cfg.scheduler
        scheduler_type = scheduler_config.get('type', 'constant')
        
        if scheduler_type == 'cosine':
            # Calculate minimum learning rate
            min_lr = self.cfg.learning_rate * scheduler_config.get('min_lr_ratio', 0.1)
            
            # Create cosine annealing scheduler
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.total_steps - scheduler_config.get('warmup_steps', 0),
                eta_min=min_lr
            )
            
            # Add warmup if specified
            warmup_steps = scheduler_config.get('warmup_steps', 0)
            if warmup_steps > 0:
                def warmup_lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        return 1.0
                
                warmup_scheduler = lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=warmup_lr_lambda
                )
                
                # Use sequential scheduler to combine warmup and cosine
                scheduler = lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps]
                )
        
        elif scheduler_type == 'linear':
            # Linear decay scheduler
            def linear_lr_lambda(step):
                if step < scheduler_config.get('warmup_steps', 0):
                    return step / scheduler_config.get('warmup_steps', 0)
                else:
                    remaining_steps = self.total_steps - step
                    total_decay_steps = self.total_steps - scheduler_config.get('warmup_steps', 0)
                    return max(scheduler_config.get('min_lr_ratio', 0.1), 
                              remaining_steps / total_decay_steps)
            
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr_lambda)
        
        else:  # constant
            scheduler = lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        return {"optimizer": optimizer, "scheduler": scheduler}

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        if self.cfg.if_ntl:
            self.NTL = NumberTokenLoss(self.model.decoder_vocab, self.model.device)
            # assert False
        else:
            self.NTL = None

        progress_bar = accelerate_tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                loss, _ = self.model.compute_loss_and_metrics(batch, self.NTL)

                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if self.logger:
                    # self.logger.log_metrics(
                    self.logger.log(
                        {"train_step_loss": loss.item()}, step=self.global_step
                    )
                self.global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        if not hasattr(self, "val_loader"):
            return {}

        self.model.eval()
        total_loss = 0

        progress_bar = accelerate_tqdm(
            self.val_loader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            loss, _ = self.model.compute_loss_and_metrics(batch)

            total_loss += loss.item()

            if self.logger:
                # self.logger.log_metrics(
                self.logger.log(
                    {"val_step_loss": loss.item()}, step=self.val_global_step
                )
                self.val_global_step += 1

            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = total_loss / len(self.val_loader)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            if self.cfg.save_dir:
                self.checkpoint_manager.save_checkpoint(self, self.epoch_counter, {"val_loss": val_loss})
        else:
            self.early_stop_counter += 1


        return {"val_loss": val_loss}

    def fit(self, train_loader, raw_loader, num_epochs, val_loader, checkpoint_dir, save_every_n_epochs):
        self.raw_loader = raw_loader
        super().fit(train_loader=train_loader, num_epochs=num_epochs, val_loader=val_loader, checkpoint_dir=checkpoint_dir, save_every_n_epochs=save_every_n_epochs)
        with torch.no_grad():
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                loss, _ = self.model.compute_loss_and_metrics(batch)
                total_loss += loss.item() 
            print(total_loss / len(train_loader))

        return {"train_loss": total_loss / len(self.train_loader)}
    
    def eval(self, pretrain_ckpt_path, raw_loader, train_dataset):
        self.model.load_state_dict(torch.load(pretrain_ckpt_path))
        self.model.to(self.accelerator.device)
        self.model.eval()
        
        # 首先计算训练数据集中每种函数类型和函数ID的y值范围
        print("正在计算训练数据集中各函数类型和函数ID的y值范围...")
        func_type_ranges = {}
        func_id_ranges = {}
        func_type_y_values = {}
        func_id_y_values = {}
        
        # 遍历训练数据集获取每种函数类型和函数ID的y值
        for i in range(len(train_dataset)):
            sample_with_metadata = train_dataset.get_item_with_metadata(i)
            func_type = sample_with_metadata['function_type']
            func_id = sample_with_metadata.get('function_id', 'unknown')  # 获取函数ID
            y_value = sample_with_metadata['y']
            
            # 按函数类型收集
            if func_type not in func_type_y_values:
                func_type_y_values[func_type] = []
            func_type_y_values[func_type].append(y_value)
            
            # 按函数ID收集
            if func_id not in func_id_y_values:
                func_id_y_values[func_id] = []
            func_id_y_values[func_id].append(y_value)
        
        # 计算每种函数类型的范围
        for func_type, y_values in func_type_y_values.items():
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = y_max - y_min
            func_type_ranges[func_type] = y_range
            print(f"函数类型 {func_type}: y范围 = [{y_min:.6f}, {y_max:.6f}], 范围大小 = {y_range:.6f}")
        
        # 计算每种函数ID的范围
        for func_id, y_values in func_id_y_values.items():
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = y_max - y_min
            func_id_ranges[func_id] = y_range
            print(f"函数ID {func_id}: y范围 = [{y_min:.6f}, {y_max:.6f}], 范围大小 = {y_range:.6f}")
        
        # 初始化按函数类型和函数ID统计的MAE和标准化MAE
        mae_by_type = {}
        normalized_mae_by_type = {}
        count_by_type = {}
        
        mae_by_func_id = {}
        normalized_mae_by_func_id = {}
        count_by_func_id = {}
        
        # 初始化按func_type和func_id组织的预测结果
        predictions_by_type_and_id = {}
        
        total_mae = 0.0
        total_normalized_mae = 0.0
        total_count = 0
        
        # 获取数据集对象以访问函数类型信息
        dataset = raw_loader.dataset
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(raw_loader):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                _, output_floats = self.model.decode(batch, 128)
                output_floats = output_floats.squeeze()
                
                # 计算当前batch中每个样本的索引
                batch_size = len(batch['y'])
                start_idx = batch_idx * raw_loader.batch_size
                
                for i, (y_pred, y) in enumerate(zip(output_floats, batch['y'])):
                    # 计算样本在数据集中的实际索引
                    sample_idx = start_idx + i
                    if sample_idx >= len(dataset):
                        break
                    
                    # 获取函数类型和函数ID
                    sample_with_metadata = dataset.get_item_with_metadata(sample_idx)
                    func_type = sample_with_metadata['function_type']
                    func_id = sample_with_metadata.get('function_id', 'unknown')
                    
                    # 计算MAE
                    median_pred = np.median(y_pred)
                    y_true = y.item()
                    mae = abs(median_pred - y_true)
                    
                    # 计算标准化MAE（使用函数类型的范围）
                    y_range_type = func_type_ranges.get(func_type, 1.0)
                    normalized_mae_type = mae / y_range_type if y_range_type > 0 else mae
                    
                    # 计算标准化MAE（使用函数ID的范围）
                    y_range_id = func_id_ranges.get(func_id, 1.0)
                    normalized_mae_id = mae / y_range_id if y_range_id > 0 else mae
                    
                    # 按func_type和func_id组织预测结果
                    if func_type not in predictions_by_type_and_id:
                        predictions_by_type_and_id[func_type] = {}
                    if func_id not in predictions_by_type_and_id[func_type]:
                        predictions_by_type_and_id[func_type][func_id] = {
                            'y_true': [],
                            'y_pred': [],
                            'mae': [],
                            'normalized_mae_by_type': [],
                            'normalized_mae_by_func_id': []
                        }
                    
                    predictions_by_type_and_id[func_type][func_id]['y_true'].append(float(y_true))
                    predictions_by_type_and_id[func_type][func_id]['y_pred'].append(float(median_pred))
                    predictions_by_type_and_id[func_type][func_id]['mae'].append(float(mae))
                    predictions_by_type_and_id[func_type][func_id]['normalized_mae_by_type'].append(float(normalized_mae_type))
                    predictions_by_type_and_id[func_type][func_id]['normalized_mae_by_func_id'].append(float(normalized_mae_id))
                    
                    # 按函数类型累加MAE和标准化MAE
                    if func_type not in mae_by_type:
                        mae_by_type[func_type] = 0.0
                        normalized_mae_by_type[func_type] = 0.0
                        count_by_type[func_type] = 0
                    
                    mae_by_type[func_type] += mae
                    normalized_mae_by_type[func_type] += normalized_mae_type
                    count_by_type[func_type] += 1
                    
                    # 按函数ID累加MAE和标准化MAE
                    if func_id not in mae_by_func_id:
                        mae_by_func_id[func_id] = 0.0
                        normalized_mae_by_func_id[func_id] = 0.0
                        count_by_func_id[func_id] = 0
                    
                    mae_by_func_id[func_id] += mae
                    normalized_mae_by_func_id[func_id] += normalized_mae_id
                    count_by_func_id[func_id] += 1
                    
                    # 累加总体MAE和标准化MAE
                    total_mae += mae
                    total_normalized_mae += normalized_mae_type  # 使用函数类型的标准化
                    total_count += 1
                    
                    print(f"样本 {sample_idx}: Pred: {median_pred:.6f}, True: {y_true:.6f}, MAE: {mae:.6f}, 标准化MAE: {normalized_mae_type:.6f}, 类型: {func_type}, ID: {func_id}")
        
        # 计算并输出每种函数类型的平均MAE和标准化MAE
        print("\n=== 按函数类型的MAE统计 ===")
        for func_type in sorted(mae_by_type.keys()):
            avg_mae = mae_by_type[func_type] / count_by_type[func_type]
            avg_normalized_mae = normalized_mae_by_type[func_type] / count_by_type[func_type]
            y_range = func_type_ranges.get(func_type, 1.0)
            print(f"{func_type}: MAE = {avg_mae:.6f}, 标准化MAE = {avg_normalized_mae:.6f} (样本数: {count_by_type[func_type]}, y范围: {y_range:.6f})")
        
        # 计算并输出每种函数ID的平均MAE和标准化MAE
        print("\n=== 按函数ID的MAE统计 ===")
        for func_id in sorted(mae_by_func_id.keys()):
            avg_mae = mae_by_func_id[func_id] / count_by_func_id[func_id]
            avg_normalized_mae = normalized_mae_by_func_id[func_id] / count_by_func_id[func_id]
            y_range = func_id_ranges.get(func_id, 1.0)
            print(f"函数ID {func_id}: MAE = {avg_mae:.6f}, 标准化MAE = {avg_normalized_mae:.6f} (样本数: {count_by_func_id[func_id]}, y范围: {y_range:.6f})")
        
        # 计算并输出总体平均MAE和标准化MAE
        overall_mae = total_mae / total_count if total_count > 0 else 0.0
        overall_normalized_mae = total_normalized_mae / total_count if total_count > 0 else 0.0
        print(f"\n总体平均MAE: {overall_mae:.6f}")
        print(f"总体平均标准化MAE: {overall_normalized_mae:.6f} (总样本数: {total_count})")
        
        # 准备保存到JSON的数据
        results_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "overall_metrics": {
                "overall_mae": float(overall_mae),
                "overall_normalized_mae": float(overall_normalized_mae),
                "total_samples": total_count
            },
            "metrics_by_function_type": {
                func_type: {
                    "mae": float(mae_by_type[func_type] / count_by_type[func_type]),
                    "normalized_mae": float(normalized_mae_by_type[func_type] / count_by_type[func_type]),
                    "sample_count": count_by_type[func_type],
                    "y_range": float(func_type_ranges.get(func_type, 1.0))
                }
                for func_type in mae_by_type
            },
            "metrics_by_function_id": {
                func_id: {
                    "mae": float(mae_by_func_id[func_id] / count_by_func_id[func_id]),
                    "normalized_mae": float(normalized_mae_by_func_id[func_id] / count_by_func_id[func_id]),
                    "sample_count": count_by_func_id[func_id],
                    "y_range": float(func_id_ranges.get(func_id, 1.0))
                }
                for func_id in mae_by_func_id
            },
            "predictions_by_type_and_id": predictions_by_type_and_id,
            "training_ranges": {
                "func_type_ranges": {k: float(v) for k, v in func_type_ranges.items()},
                "func_id_ranges": {k: float(v) for k, v in func_id_ranges.items()}
            }
        }
        
        # 保存结果到JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"eval_regress_lm_results_{timestamp}.json"
        
        # 确保输出目录存在
        output_dir = "eval_results"
        os.makedirs(output_dir, exist_ok=True)
        
        json_filepath = os.path.join(output_dir, json_filename)
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {json_filepath}")
        
        return {
            "overall_mae": overall_mae,
            "overall_normalized_mae": overall_normalized_mae,
            "mae_by_type": {func_type: mae_by_type[func_type] / count_by_type[func_type] 
                           for func_type in mae_by_type},
            "normalized_mae_by_type": {func_type: normalized_mae_by_type[func_type] / count_by_type[func_type] 
                                     for func_type in normalized_mae_by_type},
            "mae_by_func_id": {func_id: mae_by_func_id[func_id] / count_by_func_id[func_id] 
                              for func_id in mae_by_func_id},
            "normalized_mae_by_func_id": {func_id: normalized_mae_by_func_id[func_id] / count_by_func_id[func_id] 
                                        for func_id in normalized_mae_by_func_id},
            "count_by_type": count_by_type,
            "count_by_func_id": count_by_func_id,
            "func_type_ranges": func_type_ranges,
            "func_id_ranges": func_id_ranges,
            "json_filepath": json_filepath
        }

    def cal_token_dis(self, raw_loader, check_point_path):
        self.model.load_state_dict(torch.load(check_point_path))
        self.model.to(self.accelerator.device)
        self.model.eval()
        token_distribution = [{} for _ in range(self.model.decode_len)]
        total_samples = 0
        
        with torch.no_grad():
            for batch in accelerate_tqdm(raw_loader, desc="Analyzing token distribution during decode"):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                decoded_ids, _ = self.model.decode(batch, num_samples=128)
                batch_size = decoded_ids.size(0)
                total_samples += batch_size * 128  # 每个输入生成64个样本
                
                for pos in range(self.model.decode_len):
                    tokens_at_pos = decoded_ids[:, :, pos].cpu().numpy().flatten()
                    for token in tokens_at_pos:
                        if token != self.model.decoder_vocab.bos_pad_id:
                            token_distribution[pos][token] = token_distribution[pos].get(token, 0) + 1
        
        # 打印分布信息
        print(f"Token distribution analysis during decode completed. Total samples: {total_samples}")
        for pos in range(self.model.decode_len):
            dist = token_distribution[pos]
            if dist:
                print(f"Position {pos}:")
                for token_id, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    token_str = self.model.decoder_vocab.itos[token_id]
                    percentage = (count / total_samples) * 100
                    print(f"  Token {token_id} ({token_str}): {count} times ({percentage:.2f}%)")
        
        # 可视化分布
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            os.makedirs('figs/token_dis', exist_ok=True)
            for pos in range(self.model.decode_len):
                dist = token_distribution[pos]
                if dist:
                    plt.figure(figsize=(10, 6))
                    token_ids = list(dist.keys())
                    token_strs = [self.model.decoder_vocab.itos[tid] for tid in token_ids]
                    # 按字典序排列token
                    sorted_pairs = sorted(zip(token_strs, dist.values()), key=lambda x: x[0])
                    token_strs, counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
                    percentages = [(count / total_samples) * 100 for count in counts]
                    sns.barplot(x=token_strs, y=percentages)
                    plt.title(f"Token Distribution at Position {pos} During Decode")
                    plt.xlabel("Token")
                    plt.ylabel("Percentage (%)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(f"figs/token_dis/decode_token_distribution_pos_{pos}.png")
                    plt.close()
                print(f"Token distribution visualization for position {pos} during decode saved as 'figs/token_dis/decode_token_distribution_pos_{pos}.png'")
        except Exception as e:
            print(f"Error in visualization: {e}")
            print("Skipping visualization due to error.")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize wandb (or other loggers)
    if cfg.use_wandb:
        swanlab.init(project=cfg.project_name, name=cfg.experiment_name, config=cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.dataset.name == "function":
        train_dataset = FunctionTrainingDataset(
            json_file_path=cfg.dataset.params.json_file_train_path,
            function_types=cfg.dataset.params.function_types,
            max_samples_per_type=None,
            random_seed=cfg.dataset.params.random_seed,
        )
        val_dataset = FunctionTrainingDataset(
            json_file_path=cfg.dataset.params.json_file_val_path,
            function_types=cfg.dataset.params.function_types,
            max_samples_per_type=None,
            random_seed=cfg.dataset.params.random_seed,
        )
        test_dataset = FunctionTrainingDataset(
            json_file_path=cfg.dataset.params.json_file_test_path,
            function_types=cfg.dataset.params.function_types,
            max_samples_per_type=None,
            random_seed=cfg.dataset.params.random_seed,
        )
    elif cfg.dataset.name == "bbob":
        train_dataset = BBOBDataset(
            hdf5_file_path=cfg.dataset.params.hdf5_file_train_path,
        )   
        val_dataset = BBOBDataset(
            hdf5_file_path=cfg.dataset.params.hdf5_file_val_path,
        )
        test_dataset = BBOBDataset(
            hdf5_file_path=cfg.dataset.params.hdf5_file_test_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    module = RegressionModule(cfg)

    custom_collate = lambda examples: collate_fn(examples, module.model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )
    raw_loader = (
        DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=custom_collate,
        )
        if test_dataset
        else None
    )

    # Calculate total training steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.num_epochs
    module.set_total_steps(total_steps)
    
    logger.info(f"Training setup: {steps_per_epoch} steps/epoch x {cfg.num_epochs} epochs = {total_steps} total steps")
    # Start training
    module.fit(
        train_loader=train_loader,
        raw_loader=raw_loader,
        num_epochs=cfg.num_epochs,
        val_loader=val_loader,
        checkpoint_dir=cfg.save_dir,
        save_every_n_epochs=cfg.save_every_n_epochs,
    )


if __name__ == "__main__":
    main()
