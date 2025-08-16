import logging
import json
from typing import Dict
import os
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
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    # Create module and dataloaders
    module = RegressionModule(cfg)
    module.model.load_state_dict(torch.load("/data/xk/trx/ICLR26-numberic/outputs/numberic/EXP0/2025-08-11/regress_lm/checkpoints/checkpoint_epoch_10/model.pt"))
    module.model.to(module.accelerator.device)
    module.model.eval()

    # custom_collate = lambda examples: collate_fn(examples, module.model)

    func_id_trad_x = {}
    func_id_emb_x  = {}
    func_id_trad_y = {}

    for i in range(len(train_dataset)):
        item = train_dataset.get_item_with_metadata(i)
        exp_item = core.Example(
            x=item['x'],
            y=item['y']
        )
        func_id = item['function_id']
        func_type = item['function_type']
        if func_type not in func_id_trad_x:
            func_id_trad_x[func_type] = {}
            func_id_emb_x[func_type] = {}
            func_id_trad_y[func_type] = {}
        if func_id not in func_id_trad_x[func_type]:
            func_id_trad_x[func_type][func_id] = []
            func_id_emb_x[func_type][func_id] = []
            func_id_trad_y[func_type][func_id] = []
        model_input = module.model.convert_examples([exp_item])
        model_input = {k: v.to(module.accelerator.device) for k, v in model_input.items()}
        emb_x = module.model.get_mean_pooling_embedding(model_input)
        emb_x = emb_x.cpu().numpy()
        func_id_trad_x[func_type][func_id].append(item['x_num'])
        func_id_emb_x[func_type][func_id].append(emb_x.tolist())
        func_id_trad_y[func_type][func_id].append(item['y'])

    os.makedirs("regress_lm_lip_data", exist_ok=True)
    for func_type in func_id_trad_x:
        for func_id in func_id_trad_x[func_type]:
            json_data = {
                "func_id": func_id,
                "func_type": func_type,
                "trad_x": func_id_trad_x[func_type][func_id],
                "emb_x": func_id_emb_x[func_type][func_id],
                "trad_y": func_id_trad_y[func_type][func_id],
            }
            with open(f"regress_lm_lip_data/func_id_{func_id}.json", "w") as f:
                json.dump(json_data, f)

    


if __name__ == "__main__":
    main()
