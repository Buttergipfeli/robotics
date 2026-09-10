import argparse
import os
import sys
import time
from pathlib import Path

import runpod
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

RUNPOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNPOD_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "lerobot" / "sim" / "data" / "so101_ball_in_roll"
CHECKPOINT_TARGET = REPO_ROOT / "lerobot" / "sim" / "train" / "act_ball" / "checkpoints" / "last" / "pretrained_model"

DATASET_REPO_NAME = "so101_ball_in_roll"
MODEL_REPO_NAME = "so101_act_ball"
POD_NAME = "so101-act-training"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
CONTAINER_DISK_GB = 40
POLL_SECONDS = 60
MAX_HOURS = 6

BOOTSTRAP = (
    'bash -c "'
    "cd /workspace && "
    "pip install -q 'lerobot[dataset,training]' && "
    "huggingface-cli download $DATASET_REPO --repo-type dataset --local-dir /workspace/data && "
    "python /workspace/data/train_remote.py; "
    'runpodctl remove pod $RUNPOD_POD_ID"'
)


def upload_dataset(api, dataset_repo):
    if not DATA_ROOT.exists():
        sys.exit(f"Dataset not found at {DATA_ROOT}. Record episodes first.")
    print(f"Uploading dataset to {dataset_repo} ...")
    api.create_repo(dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder_path=str(DATA_ROOT))
    api.upload_file(
        repo_id=dataset_repo,
        repo_type="dataset",
        path_or_fileobj=str(RUNPOD_DIR / "train_remote.py"),
        path_in_repo="train_remote.py",
    )
    print("Upload done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    env_file = RUNPOD_DIR / ".env"
    if not env_file.exists():
        sys.exit(f"{env_file} not found. Copy .env.example to .env and fill it in.")
    load_dotenv(env_file)

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    hf_token = os.environ["HF_TOKEN"]
    hf_user = os.environ["HF_USERNAME"]
    api = HfApi(token=hf_token)

    dataset_repo = f"{hf_user}/{DATASET_REPO_NAME}"
    model_repo = f"{hf_user}/{MODEL_REPO_NAME}"

    if not args.skip_upload:
        upload_dataset(api, dataset_repo)

    baseline_sha = None
    if api.repo_exists(model_repo):
        baseline_sha = api.repo_info(model_repo).sha

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=args.gpu,
        container_disk_in_gb=CONTAINER_DISK_GB,
        docker_args=BOOTSTRAP,
        env={
            "HF_TOKEN": hf_token,
            "DATASET_REPO": dataset_repo,
            "MODEL_REPO": model_repo,
            "TRAIN_STEPS": str(args.steps),
            "BATCH_SIZE": str(args.batch_size),
        },
    )
    pod_id = pod["id"]
    print(f"Pod {pod_id} created ({args.gpu}), training {args.steps} steps.")

    deadline = time.time() + MAX_HOURS * 3600
    try:
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            if api.repo_exists(model_repo) and api.repo_info(model_repo).sha != baseline_sha:
                print("Checkpoint arrived on the hub.")
                break
            try:
                status = runpod.get_pod(pod_id)
                print(f"waiting... pod status: {status.get('desiredStatus', 'unknown')}")
            except Exception:
                print("waiting... pod no longer queryable")
        else:
            sys.exit(f"Timeout after {MAX_HOURS}h, pod terminated. Check the RunPod console logs.")
    finally:
        try:
            runpod.terminate_pod(pod_id)
            print(f"Pod {pod_id} terminated.")
        except Exception as e:
            print(f"Pod termination failed ({e}), stop it manually in the RunPod console!")

    CHECKPOINT_TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_repo, local_dir=str(CHECKPOINT_TARGET), token=hf_token)
    print(f"Checkpoint downloaded to {CHECKPOINT_TARGET}")
    print("Next: ./.venv/bin/python3 lerobot/sim/eval_policy.py 50")


if __name__ == "__main__":
    main()
