# Training on RunPod

Train the ACT policy on a cloud GPU instead of the local Mac. Only the dataset
and `train_policy.py` are needed on the pod, no MuJoCo, no scene files.

## 0. One-time setup: SSH key

Generate a key pair if you do not have one yet (creates `~/.ssh/id_ed25519`
and `~/.ssh/id_ed25519.pub`):

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

Show the public key and copy it:

```bash
cat ~/.ssh/id_ed25519.pub
```

Paste it on [console.runpod.io/user/settings](https://www.console.runpod.io/user/settings)
into the **SSH Public Keys** field. RunPod injects the keys from your account
settings into pods automatically, including already running ones.

## 1. Deploy a pod

- Any current NVIDIA GPU works; ACT training is light, so pick whatever is
  cheap and available (12+ GB VRAM is more than enough).
- Use a PyTorch/CUDA template and ~20 GB container disk.
- The pod must support **SSH over exposed TCP** (public IP): the proxy
  connection via `ssh.runpod.io` does not allow scp/rsync file transfer.
- Open the pod's **Connect** dialog and note IP and port from the
  "SSH over exposed TCP" command.

## 2. Set connection variables (local shell)

```bash
export POD_IP=<ip-from-connect-dialog>
export POD_PORT=<port-from-connect-dialog>
```

## 3. Upload dataset and training script (local shell)

Run from the repo root:

```bash
rsync -avz -e "ssh -p $POD_PORT" lerobot/sim/data lerobot/sim/train_policy.py root@$POD_IP:/workspace/so101/
```

This creates `/workspace/so101/train_policy.py` and
`/workspace/so101/data/so101_ball_in_roll/` on the pod, the relative layout the
script expects.

## 4. Connect and start training (on the pod)

```bash
ssh -p $POD_PORT root@$POD_IP
```

Then on the pod:

```bash
pip install 'lerobot[dataset,training]'
```

```bash
cd /workspace/so101 && nohup python train_policy.py > train.log 2>&1 &
```

The training survives SSH disconnects thanks to `nohup`. Watch progress with:

```bash
tail -f /workspace/so101/train.log
```

Done when the log prints `End of training` (checkpoints land in
`/workspace/so101/train/act_ball/checkpoints/`).

Optional: edit `BATCH_SIZE = 32` in the pod's copy of `train_policy.py` to use
the GPU better.

## 5. Download the checkpoint (local shell)

Remove the old local training output first, then pull:

```bash
rm -rf lerobot/sim/train
```

```bash
rsync -avz -e "ssh -p $POD_PORT" root@$POD_IP:/workspace/so101/train/act_ball lerobot/sim/train/
```

The checkpoint ends up at `lerobot/sim/train/act_ball/checkpoints/last/`,
exactly where `eval_policy.py` and `rollout_policy.py` look for it.

## 6. Stop the pod

Stop (or terminate) the pod in the RunPod console, it bills while running.

## 7. Evaluate locally

```bash
./.venv/bin/python3 lerobot/sim/eval_policy.py 50
```
