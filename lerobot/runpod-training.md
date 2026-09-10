# Training on RunPod

Train the ACT policy on a cloud GPU. The automated way is a single command:
`lerobot/runpod/launch_training.py` creates the pod via the RunPod API, drives
everything over SSH (upload, install, training, download) and always
terminates the pod at the end, on success, failure, timeout or Ctrl+C.

## One-time setup

### 1. SSH key

Generate a key pair if you do not have one yet (creates `~/.ssh/id_ed25519`
and `~/.ssh/id_ed25519.pub`, both stay in `~/.ssh/`):

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

Show the public key and copy it:

```bash
cat ~/.ssh/id_ed25519.pub
```

Paste it on [console.runpod.io/user/settings](https://www.console.runpod.io/user/settings)
under **SSH Public Keys**. RunPod injects the keys from your account settings
into pods automatically; `ssh` and `rsync` pick up the private key from its
default location, nothing else to configure.

### 2. API key

On the same settings page under **API Keys**: **Create API Key**, permission
**All** (the script manages pods via `api.runpod.io/graphql`, which the other
permission levels do not cover). The key is shown only once, copy it straight
into the env file:

```bash
cat > lerobot/runpod/.env <<'EOF'
RUNPOD_API_KEY=<your-key>
EOF
```

The file is gitignored. Treat the key like a password, it has full account
access.

### 3. Credits

Pods only start with a positive balance, top up a few dollars under
**Billing**. A full 50k-step run on an RTX 4090 costs roughly 1 to 2 USD.

## Run a training

Validate the whole chain first with a cheap test run (a few minutes, a few
cents):

```bash
./.venv/bin/python3 lerobot/runpod/launch_training.py --steps 200
```

Then the real run:

```bash
./.venv/bin/python3 lerobot/runpod/launch_training.py
```

Options: `--steps` (default 50000), `--batch-size` (default 32),
`--gpu "NVIDIA GeForce RTX 4090"` (any RunPod GPU type id).

What the script does:

1. Creates a pod (PyTorch image, public IP, SSH exposed).
2. Waits until SSH is reachable.
3. Uploads the dataset and `train_policy.py` via rsync.
4. Installs `lerobot[dataset,training]` on the pod, then repins torch and
   torchvision to cu128 wheels, which run on any CUDA 12.x or 13.x host
   driver (community hosts vary).
5. Verifies `torch.cuda.is_available()` and aborts if the GPU is unusable,
   so a bad host never trains silently on CPU.
6. Starts training under `nohup`, records the exit code.
7. Polls once a minute, printing the latest log line. Every checkpoint the
   training writes (every 5000 steps, `SAVE_FREQ` in `train_policy.py`) is
   downloaded to `lerobot/sim/train/act_ball/checkpoints/<step>/` as soon as
   it exists, so you can evaluate intermediate policies or stop early with
   Ctrl+C when the loss plateaus. On failure it prints the last 30 log lines
   and aborts; hard timeout after 24 h.
8. Downloads the final checkpoint; `checkpoints/last/` is exactly where
   `eval_policy.py` and `rollout_policy.py` look for it.
9. Terminates the pod (also on any failure path).

While a training runs, print the pod's log from a second terminal (`-f`
follows live, Ctrl+C detaches without touching the training):

```bash
./.venv/bin/python3 lerobot/runpod/show_logs.py -f
```

Afterwards, evaluate locally:

```bash
./.venv/bin/python3 lerobot/sim/eval_policy.py 50
```

## Manual fallback over plain SSH

If you ever need to do it by hand: deploy any pod with a PyTorch/CUDA template
that supports **SSH over exposed TCP** (the `ssh.runpod.io` proxy cannot do
scp/rsync), take IP and port from the pod's Connect dialog, then:

```bash
export POD_IP=<ip> POD_PORT=<port>
rsync -avz -e "ssh -p $POD_PORT" lerobot/sim/data lerobot/sim/train_policy.py root@$POD_IP:/workspace/so101/
ssh -p $POD_PORT root@$POD_IP
# on the pod:
pip install 'lerobot[dataset,training]'
cd /workspace/so101 && nohup python train_policy.py > train.log 2>&1 &
tail -f train.log
# back on the Mac, when the log prints "End of training":
rm -rf lerobot/sim/train
rsync -avz -e "ssh -p $POD_PORT" root@$POD_IP:/workspace/so101/train/act_ball lerobot/sim/train/
```

Stop the pod in the console afterwards, it bills while running.
