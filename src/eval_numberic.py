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
from .train_numberic import RegressionModule

# Initialize vocabs
encoder_vocab = ExtendedSentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=P10Tokenizer())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def collate_fn(examples, model):
    tensor_examples = model.convert_number_examples(examples)
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
            batch_size=20,
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
    module.eval(
        raw_loader=raw_loader,
        pretrain_ckpt_path='/data/xk/trx/ICLR26-numberic/outputs/numberic/EXP0/2025-08-12/01-31-07/checkpoints/checkpoint_epoch_30/model.pt',
        train_dataset=train_dataset,
    )


if __name__ == "__main__":
    main()
