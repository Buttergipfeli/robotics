import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import runpod
from dotenv import load_dotenv

RUNPOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNPOD_DIR.parents[1]
SIM_DIR = REPO_ROOT / "lerobot" / "sim"
DATA_DIR = SIM_DIR / "data"
TRAIN_SCRIPT = SIM_DIR / "train_policy.py"
LOCAL_TRAIN_DIR = SIM_DIR / "train"

POD_NAME = "so101-act-training"
IMAGE = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_CLOUD_TYPE = "SECURE"
CONTAINER_DISK_GB = 40
REMOTE_DIR = "/workspace/so101"
REMOTE_PYTHON = "/workspace/venv/bin/python"
SSH_READY_TIMEOUT = 600
POLL_SECONDS = 60
MAX_HOURS = 24

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


def apply_ssh_key_from_env():
    key_path = os.environ.get("RUNPOD_SSH_KEY")
    if key_path:
        SSH_OPTS.extend(["-i", os.path.expanduser(key_path), "-o", "IdentitiesOnly=yes"])


def run_ssh(ip, port, command, capture=False):
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), f"root@{ip}", command],
        capture_output=capture,
        text=True,
    )


def run_rsync(port, sources, destination, delete=False):
    command = ["rsync", "-az", "-e", f"ssh -p {port} {' '.join(SSH_OPTS)}"]
    if delete:
        command.append("--delete")
    subprocess.run([*command, *sources, destination], check=True)


def wait_for_ssh(pod_id):
    deadline = time.time() + SSH_READY_TIMEOUT
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id) or {}
        runtime = pod.get("runtime") or {}
        for mapping in runtime.get("ports") or []:
            if mapping.get("privatePort") == 22 and mapping.get("isIpPublic"):
                ip, port = mapping["ip"], mapping["publicPort"]
                if run_ssh(ip, port, "true").returncode == 0:
                    return ip, port
        time.sleep(10)
    sys.exit("Pod did not become reachable via SSH.")


def wait_for_training(ip, port):
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        exit_code = run_ssh(ip, port, f"cat {REMOTE_DIR}/EXIT 2>/dev/null", capture=True).stdout.strip()
        if exit_code == "0":
            return
        if exit_code:
            log_tail = run_ssh(ip, port, f"tail -30 {REMOTE_DIR}/train.log", capture=True)
            sys.exit(f"Training failed with exit code {exit_code}. Last log lines:\n{log_tail.stdout}")
        progress = run_ssh(ip, port, f"tail -1 {REMOTE_DIR}/train.log 2>/dev/null", capture=True)
        print(f"waiting... {progress.stdout.strip()}")
    sys.exit(f"Timeout after {MAX_HOURS}h.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--cloud", default=DEFAULT_CLOUD_TYPE, choices=["COMMUNITY", "SECURE", "ALL"])
    args = parser.parse_args()

    env_file = RUNPOD_DIR / ".env"
    if not env_file.exists():
        sys.exit(f"{env_file} not found. Create it with RUNPOD_API_KEY=<key>.")
    load_dotenv(env_file)
    runpod.api_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod.api_key:
        sys.exit(f"RUNPOD_API_KEY missing in {env_file}.")
    apply_ssh_key_from_env()

    if not (DATA_DIR / "so101_ball_in_roll").exists():
        sys.exit(f"Dataset not found in {DATA_DIR}. Record episodes first.")

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=args.gpu,
        cloud_type=args.cloud,
        container_disk_in_gb=CONTAINER_DISK_GB,
        support_public_ip=True,
        start_ssh=True,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"Pod {pod_id} created ({args.gpu}), waiting for SSH...")

    try:
        ip, port = wait_for_ssh(pod_id)
        print(f"SSH ready at {ip}:{port}, uploading dataset and training script...")
        run_ssh(ip, port, f"mkdir -p {REMOTE_DIR}")
        run_rsync(port, [str(DATA_DIR), str(TRAIN_SCRIPT)], f"root@{ip}:{REMOTE_DIR}/")

        print("Installing lerobot on the pod...")
        install = run_ssh(
            ip,
            port,
            "python3 -m venv --system-site-packages /workspace/venv && "
            f"{REMOTE_PYTHON} -m pip install -q 'lerobot[dataset,training]'",
            capture=True,
        )
        if install.returncode != 0:
            sys.exit(f"Install failed:\n{install.stderr[-2000:]}")

        print(f"Starting training ({args.steps} steps, batch {args.batch_size})...")
        start = run_ssh(
            ip,
            port,
            f"cd {REMOTE_DIR} && rm -f EXIT && "
            f"nohup bash -c '{REMOTE_PYTHON} train_policy.py {args.steps} {args.batch_size} "
            f"> train.log 2>&1; echo $? > EXIT' >/dev/null 2>&1 &",
            capture=True,
        )
        if start.returncode != 0:
            sys.exit(f"Failed to start training:\n{start.stderr}")
        wait_for_training(ip, port)

        print("Training finished, downloading checkpoint...")
        LOCAL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        run_rsync(port, [f"root@{ip}:{REMOTE_DIR}/train/act_ball"], f"{LOCAL_TRAIN_DIR}/", delete=True)
        print(f"Checkpoint at {LOCAL_TRAIN_DIR}/act_ball/checkpoints/last/")
        print("Next: ./.venv/bin/python3 lerobot/sim/eval_policy.py 50")
    finally:
        try:
            runpod.terminate_pod(pod_id)
            print(f"Pod {pod_id} terminated.")
        except Exception as e:
            print(f"Pod termination failed ({e}), stop it manually in the RunPod console!")


if __name__ == "__main__":
    main()
