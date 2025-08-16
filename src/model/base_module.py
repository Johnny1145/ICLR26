import abc
from functools import partial
from typing import Dict, Optional

import swanlab
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm
from torch.utils.data import DataLoader

from src.utils.checkpoint import CheckpointManager
from src.utils.number_token_loss import NumberTokenLoss

class BaseModule:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        gradient_accumulation_steps: int = 1,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        log_dir: Optional[str] = None,
    ) -> None:
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        self.model = model
        self.criterion = criterion
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.val_global_step = 0

        self.model_params = self._count_parameters()
        self.accelerator.print(f"Model parameters: {self.model_params:,}")

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        self.logger = None
        if self.use_wandb and self.project_name:
            self.logger = swanlab

    def _count_parameters(self) -> int:
        """计算模型参数量

        Returns:
            int: 模型参数总量
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params

    def get_model_parameters(self) -> Dict[str, int]:
        """获取模型参数量信息

        Returns:
            Dict[str, int]: 包含参数量信息的字典
        """
        return {
            "total_parameters": self.model_params,
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "non_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if not p.requires_grad
            ),
        }

    @abc.abstractmethod
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            torch.optim.Optimizer or dict: Optimizer or dict containing optimizer and scheduler
        """
        pass

    def prepare(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        if self.optimizer is None:
            optimizer_dict = self.configure_optimizers()
            if isinstance(optimizer_dict, dict):
                self.optimizer = optimizer_dict["optimizer"]
                self.scheduler = optimizer_dict.get("scheduler")

        if val_loader is not None:
            (
                self.model,
                self.optimizer,
                train_loader,
                val_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader, self.scheduler
            )
            self.val_loader = val_loader
        else:
            (
                self.model,
                self.optimizer,
                train_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, self.scheduler
            )

        self.train_loader = train_loader

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        progress_bar = accelerate_tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                    targets = batch.get("labels") or batch.get("targets")
                    if targets is None:
                        loss = outputs.loss
                    else:
                        if self.criterion:
                            loss = self.criterion(outputs, targets)
                        else:
                            loss, _ = self.model.compute_loss_and_metrics(batch, self.NTL)
                else:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if self.logger:
                    self.logger.log({"train_step_loss": loss.item()})
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
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                targets = batch.get("labels") or batch.get("targets")
                if targets is None:
                    loss = outputs.loss
                else:
                    if self.criterion:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss, _ = self.model.compute_loss_and_metrics(batch)
            else:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = total_loss / len(self.val_loader)

        return {"val_loss": val_loss}

    def save_checkpoint(self, save_path: str, epoch: int = None, metrics: Dict = None):
        self.accelerator.save_state(f"{save_path}/ckpt_{epoch}")

        import json

        param_info = self.get_model_parameters()
        param_file = f"{save_path}/model_parameters.json"
        with open(param_file, "w") as f:
            json.dump(param_info, f, indent=2)

    def load_checkpoint(self, load_path: str) -> Dict:
        self.accelerator.load_state(load_path)

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[list] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
    ):
        if not hasattr(self, "train_loader"):
            self.prepare(train_loader, val_loader)

        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        if self.logger:
            param_info = self.get_model_parameters()
            self.logger.log(param_info)
            self.accelerator.print(f"Model parameters logged: {param_info}")
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = 20
        self.epoch_counter = 0

        for epoch in range(num_epochs):
            self.epoch_counter = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            metrics = {**train_metrics, **val_metrics}

            if self.logger:
                self.logger.log(metrics)

            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(f"Epoch [{epoch+1}/{num_epochs}]: {metrics_str}")

            if self.accelerator.is_main_process and checkpoint_dir:
                if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                    self.checkpoint_manager.save_checkpoint(self, epoch + 1, metrics)

            if callbacks:
                for callback in callbacks:
                    callback(self, metrics, epoch)

            if self.early_stop_counter >= self.early_stop_patience:
                self.accelerator.print("Early stopping triggered")
                self.save_checkpoint(checkpoint_dir, epoch, metrics)
                break

        if self.accelerator.is_main_process and checkpoint_dir:
            self.checkpoint_manager.save_checkpoint(self, num_epochs, metrics)

        self.accelerator.end_training()
