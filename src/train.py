import logging
import os
from omegaconf import DictConfig, OmegaConf
import hydra
import random
import torch
import numpy as np
from src.dataloader.load import OpenArmDataset
from MambaVLA import train_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "basename",
    lambda path: os.path.basename(os.path.normpath(path)) if path else "",
)

def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    data_directory: str,
    batch_size: int = 256,
    num_epochs: int = 500,
    learning_rate: float = 1e-4,
    device: str = None,
    latent_dim: int = 256,
    embed_dim: int = 256,
    n_layer: int = 5,
    d_intermediate: int = 256,
    obs_tok_len: int = 2,
    action_seq_len: int = 5,
    save_dir: str = './checkpoints',
    save_freq: int = 10,
    max_len_data: int = 60,
    enable_ema: bool = True,
    ema_decay_rate: float = 0.995,
    enable_data_scaling: bool = True,
    data_scaler_type: str = "minmax",
    num_workers: int = 4,
    transformer_weight_decay: float = 0.05,
    obs_encoder_weight_decay: float = 0.05,
    betas: list = None,
    sampling_steps: int = 4,
    wandb_project: str = "OpenARM-VLA",
    wandb_entity: str = "sainavaneet",
    wandb_name: str = "new_model",
    model_type: str = "mamba",
    transformer_cfg: dict | None = None,
    demos_per_task: int = 50,
    obs_dim: int = 32,
    action_dim: int = 8,
    state_dim: int = 9,
    chunck_size: int = 5,
    start_idx: int = 0,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info("Loading OpenARM dataset...")
    data_directory = os.path.normpath(data_directory)
    dataset = OpenArmDataset(
        data_directory=data_directory,
        device="cpu",
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        max_len_data=max_len_data,
        chunck_size=chunck_size,
        start_idx=start_idx,
        demos_per_task=demos_per_task,
    )
    
    if betas is None:
        betas = [0.9, 0.9]

    train_policy(
        dataloader=dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        obs_tok_len=obs_tok_len,
        action_seq_len=action_seq_len,
        save_dir=save_dir,
        save_freq=save_freq,
        enable_ema=enable_ema,
        enable_data_scaling=enable_data_scaling,
        data_scaler_type=data_scaler_type,
        dataloader_workers=num_workers,
        eval_during_training=None,
        eval_callback=None,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        model_type=model_type,
        transformer_cfg=transformer_cfg,
        transformer_weight_decay=transformer_weight_decay,
        obs_encoder_weight_decay=obs_encoder_weight_decay,
        betas=betas,
        sampling_steps=sampling_steps,
        ema_decay_rate=ema_decay_rate,
    )


_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf")

@hydra.main(version_base=None, config_path=_config_path, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration.
    
    You can override any config value from command line, e.g.:
        python -m OpenARM-VLA.src.train batch_size=512 latent_dim=512
        python -m OpenARM-VLA.src.train data_directory=/path/to/openarm_dataset
        python -m OpenARM-VLA.src.train wandb.name=my_experiment
    """
    if not cfg.data_directory:
        raise ValueError("data_directory is required")

    set_seed_everywhere(cfg.seed)

    wandb_project = cfg.wandb.project if cfg.wandb.enabled else None
    wandb_entity = cfg.wandb.entity if cfg.wandb.enabled else None
    wandb_name = cfg.wandb.name if cfg.wandb.enabled else None

    betas = cfg.betas
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = list(betas)
    else:
        betas = [0.9, 0.9]
    
    train(
        data_directory=cfg.data_directory,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate,
        save_dir=cfg.save_dir,
        save_freq=cfg.save_freq,
        max_len_data=cfg.max_len_data,
        enable_ema=cfg.enable_ema,
        ema_decay_rate=cfg.ema_decay_rate,
        enable_data_scaling=cfg.enable_data_scaling,
        data_scaler_type=cfg.data_scaler_type,
        num_workers=cfg.num_workers,
        transformer_weight_decay=cfg.transformer_weight_decay,
        obs_encoder_weight_decay=cfg.obs_encoder_weight_decay,
        betas=betas,
        sampling_steps=cfg.sampling_steps,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        model_type=cfg.get("model_type", "mamba"),
        transformer_cfg=cfg.get("transformer", None),
        demos_per_task=cfg.demos_per_task,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        chunck_size=cfg.get('chunck_size', 1),
        start_idx=cfg.get('start_idx', 0),
    )


if __name__ == "__main__":
    main()
