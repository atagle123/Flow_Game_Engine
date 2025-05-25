import os
from omegaconf import OmegaConf
from tqdm import tqdm
from src.dataset.dataset import Maze_Dataset
import wandb
from typing import Dict
from src.models.flow import FlowLearner


class Trainer:
    def __init__(self, cfg, training_cfg):
        self.cfg = cfg
        self.training_cfg = training_cfg

        if training_cfg.wandb_log:
            wandb.init(
                project="Flow_Game_Engine",
                name=cfg.wandb.wandb_exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        os.makedirs(cfg.savepath, exist_ok=True)  # model path

    def train(self):
        dataset, dataset_val = self.load_dataset(self.training_cfg.dataset_path)

        model_cls = "FlowLearner"
        model = globals()[model_cls].create(self.cfg.seed, self.cfg.flow_model)

        model = self.train_loop(model, dataset, dataset_val)
        wandb.finish()

    def load_dataset(self, filepath):

        dataset = Maze_Dataset(filepath)

        dataset, dataset_val = (
            dataset.split(0.96) if self.training_cfg.val_dataset else (dataset, None)
        )

        return dataset, dataset_val

    def _log_info(self, info: Dict, step: int, prefix: str):

        info_str = " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in info.items()])
        print(f"{info_str} | (step {step})")

        if self.training_cfg.wandb_log:
            wandb.log({f"{prefix}/{k}": v for k, v in info.items()}, step=step)

    def train_loop(self, model, dataset, dataset_val):
        keys = None

        for step in tqdm(range(1, self.cfg.flow_model.train.steps + 1), smoothing=0.1):

            sample = dataset.sample_jax(self.cfg.flow_model.train.batch_size, keys=keys)
            model, info = model.update(sample)

            if step % self.training_cfg.log_freq == 0:
                self._log_info(info, step, prefix="train")

                if dataset_val is not None:
                    val_batch = dataset_val.sample_jax(
                        self.cfg.flow_model.train.batch_size, keys=keys
                    )
                    _, val_info = model.update(val_batch)
                    self._log_info(val_info, step, prefix="val")

            if step % self.training_cfg.save_freq == 0 or step == 1:
                model.save(self.cfg.savepath, step)

        return model
