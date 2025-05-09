import os
import logging
import hydra
from dataclasses import dataclass
from src.utils.training import Trainer


CONFIG_PATH = "../configs"
MEMORY_FRACTION = "0.75"
PREALLOCATE_MEMORY = "true"
WANDB_LOG = False
VAL_DATASET = True
LOG_FREQ = 100
SAVE_FREQ = 100000
DATASET_PATH = "logs/data/data_maze.npz"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    log_freq: int
    save_freq: int
    wandb_log : bool
    val_dataset : bool
    dataset_filepath: str

def configure_environment():
    """
    Configure environment variables for JAX.
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = MEMORY_FRACTION
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = PREALLOCATE_MEMORY
    logger.info("Environment variables configured for JAX.")

    TRAINING_CONFIG = TrainingConfig(
                                log_freq=LOG_FREQ,
                                save_freq=SAVE_FREQ,
                                wandb_log=WANDB_LOG,
                                val_dataset=VAL_DATASET,
                                dataset_filepath = DATASET_PATH
                                )
    return TRAINING_CONFIG

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg):
    """
    Main function to start the training process.
    """
    logger.info(f"Starting experiment with config: {cfg}")

    training_config = configure_environment()

    trainer = Trainer(cfg, training_config)
    trainer.train()
    
    logger.info("Training completed.")

if __name__ == "__main__":
    main()