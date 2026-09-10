import os
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/train/act_ball")


def main():
    dataset_repo = os.environ["DATASET_REPO"]
    model_repo = os.environ["MODEL_REPO"]
    steps = int(os.environ.get("TRAIN_STEPS", "50000"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset_repo, root=str(DATA_DIR)),
        policy=ACTConfig(device="cuda", push_to_hub=False),
        output_dir=OUTPUT_DIR,
        job_name="act_ball",
        steps=steps,
        batch_size=batch_size,
        seed=1000,
    )
    cfg.validate()
    train(cfg)

    api = HfApi()
    api.create_repo(model_repo, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=model_repo,
        folder_path=str(OUTPUT_DIR / "checkpoints" / "last" / "pretrained_model"),
    )
    print("Checkpoint uploaded to the hub.")


if __name__ == "__main__":
    main()
