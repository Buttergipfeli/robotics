# Training progress

| | Run 1 | Run 2 |
| --- | --- | --- |
| Data | 50 expert episodes, fixed scene | 300 expert episodes, domain randomisation (mat offset, ball friction and mass, camera pose and fov, lighting, mat and paper shade) |
| Training | ACT, 20k steps, batch 8, Mac (MPS) | ACT, 20k of 50k steps, batch 32, RunPod RTX 5090, loss 5.36 → 0.20 |
| Sim eval, seed 100, 20 episodes | 11/20 (5 timeouts, 4 ball lost), no randomisation | 11/20 (7 timeouts, 2 ball lost), with randomisation |
| Real rollout | lifts, opens gripper, then oscillates ±1 cm | no coherent behaviour |

## Findings

- Same rate on a harder test. Run 2 holds the ball more reliably and fails by not
  finding it; all three balls near the far mat edge (x ≥ 0.34 m) timed out.
- Both real rollouts failed for reasons outside the policy:
  1. Joint units differed: sim normalises over MJCF ranges, the follower over its
     calibration ranges (wrist roll 78° off). Fixed by `real/joint_mapping.py`.
  2. The real ball is light blue with print and a barcode, the sim ball plain
     dark blue. Open.

## Next

- Match the ball colour to the real one, randomise colour and markings, add
  camera exposure and noise variation, re-record 300 episodes, retrain.
- Evaluate every checkpoint with 50 episodes at seed 100 so runs compare per
  episode.
