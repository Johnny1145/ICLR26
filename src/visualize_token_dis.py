import logging
from typing import Dict

import hydra
import numpy as np
import rootutils
import swanlab
import torch
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
from .model.regress_lm.vocabs import DecoderVocab, ExtendedSentencePieceVocab
from .train import RegressionModule

# Initialize vocabs
encoder_vocab = ExtendedSentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=P10Tokenizer())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def collate_fn(examples, model):
    tensor_examples = model.convert_examples(examples)
    return tensor_examples

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize wandb (or other loggers)
    # if cfg.use_wandb:
    #     swanlab.init(project=cfg.project_name, name=cfg.experiment_name, config=cfg)

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
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    # Create module and dataloaders
    module = RegressionModule(cfg)

    custom_collate = lambda examples: collate_fn(examples, module.model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
        if val_dataset
        else None
    )
    raw_loader = (
        DataLoader(
            test_dataset,
            batch_size=512,
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
    
    # 为训练集中的每个函数生成测试点
    
    # Start training
    module.model.eval()
    token_distribution = [{} for _ in range(module.model.decode_len)]
    total_samples = 0

    for batch in accelerate_tqdm(raw_loader, desc="Analyzing token distribution"):
        batch = {k: v for k, v in batch.items()}
        decoder_targets = batch["decoder_target"]
        batch_size = decoder_targets.size(0)
        total_samples += batch_size

        for pos in range(module.model.decode_len):
            tokens_at_pos = decoder_targets[:, pos].numpy()
            for token in tokens_at_pos:
                if token != module.model.decoder_vocab.bos_pad_id:
                    token_distribution[pos][token] = token_distribution[pos].get(token, 0) + 1

    # 打印分布信息
    print(f"Token distribution analysis completed. Total samples: {total_samples}")
    for pos in range(module.model.decode_len):
        dist = token_distribution[pos]
        if dist:
            print(f"Position {pos}:")
            for token_id, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                token_str = module.model.decoder_vocab.itos[token_id]
                percentage = (count / total_samples) * 100
                print(f"  Token {token_id} ({token_str}): {count} times ({percentage:.2f}%)")

    # 可视化分布
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        os.makedirs('figs/token_dis', exist_ok=True)
        for pos in range(module.model.decode_len):
            dist = token_distribution[pos]
            if dist:
                plt.figure(figsize=(10, 6))
                token_ids = list(dist.keys())
                token_strs = [module.model.decoder_vocab.itos[tid] for tid in token_ids]
                # 按字典序排列token
                sorted_pairs = sorted(zip(token_strs, dist.values()), key=lambda x: x[0])
                token_strs, counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
                counts = list(dist.values())
                percentages = [(count / total_samples) * 100 for count in counts]
                sns.barplot(x=token_strs, y=percentages)
                plt.title(f"Token Distribution at Position {pos}")
                plt.xlabel("Token")
                plt.ylabel("Percentage (%)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"figs/token_dis/test_token_distribution_pos_{pos}.png")
                plt.close()
            print(f"Token distribution visualization for position {pos} saved as 'figs/token_dis/test_token_distribution_pos_{pos}.png'")
    except Exception as e:
        print(f"Error in visualization: {e}")
        print("Skipping visualization due to error.")


if __name__ == "__main__":
    main()
