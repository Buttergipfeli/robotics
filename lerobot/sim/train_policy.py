import sys
from pathlib import Path

import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

SIM_DIR = Path(__file__).parent

REPO_ID = "local/so101_ball_in_roll"
DATA_ROOT = SIM_DIR / "data" / "so101_ball_in_roll"
OUTPUT_DIR = SIM_DIR / "train" / "act_ball"
JOB_NAME = "act_ball"
TRAIN_STEPS = 50_000
BATCH_SIZE = 8
SAVE_FREQ = 5_000
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
SEED = 1000


def main(steps=TRAIN_STEPS, batch_size=BATCH_SIZE, output_dir=OUTPUT_DIR):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, root=str(DATA_ROOT)),
        policy=ACTConfig(device=DEVICE, push_to_hub=False),
        output_dir=Path(output_dir),
        job_name=JOB_NAME,
        steps=steps,
        batch_size=batch_size,
        save_freq=SAVE_FREQ,
        seed=SEED,
    )
    cfg.validate()
    train(cfg)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_STEPS
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else BATCH_SIZE
    main(steps=steps, batch_size=batch_size)
