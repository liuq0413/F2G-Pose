# Usage

[← README](../README.md) · [Installation](INSTALL.md) · [Evaluation results](EVALUATION.md)

The CLI has six commands: `f2g-pose train`, `eval`, `predict`, `demo`, `benchmark`, and `sam3-mask`. Run any command with `--help` for its options. Complete [installation](INSTALL.md) first, including RADIO and the model weights.

## Interactive demo

```bash
f2g-pose demo --checkpoint weights/f2g-pose.pt --port 7860
```

Open **http://127.0.0.1:7860**. Upload RGB, raw depth, and a binary mask; enter the camera intrinsics and select the depth units. Click **Estimate pose** to view the axes, 3D box, and completed point cloud, then download JSON or PLY.

For SAM object selection, start the demo with:

```bash
f2g-pose demo --checkpoint weights/f2g-pose.pt \
  --sam-checkpoint weights/sam_vit_b_01ec64.pth --port 7860
```

Click foreground points or enter an `x_min y_min x_max y_max` box, choose **Preview SAM mask**, then **Confirm preview mask** before estimating pose. Each browser session has its own selection and downloads.

### Automatic SAM 3 masks

SAM 3 runs in a separate Python environment; see [installation](INSTALL.md#optional-sam-3-automatic-instance-masks). Start the demo with `--sam3-python` pointing to that environment:

```bash
f2g-pose demo --checkpoint weights/f2g-pose.pt --port 7860 \
  --sam3-python third_party_local/.venv-sam3/bin/python
```

Upload RGB, enter a short English phrase such as `water bottle`, then click **Find all with SAM 3**. Select each detected bottle in the dropdown, inspect the preview, and click **Confirm preview mask** before estimating its pose. The individual binary masks can also be downloaded. Upload aligned depth, set the actual camera intrinsics and depth units as usual. After moving bottles, upload the new RGB and aligned depth and generate masks again. Detection order is left to right; verify the preview because transparent bottles can be missed or merged.

Generate masks without starting the demo:

```bash
f2g-pose sam3-mask --rgb example/rgb.png --output outputs/bottle_masks \
  --sam3-python third_party_local/.venv-sam3/bin/python --prompt 'water bottle'
```

This creates `mask_01.png`, `mask_02.png`, etc. and `masks.json` with confidence scores. If you already have an authorized SAM 3 checkpoint on disk, pass `--sam3-checkpoint /path/to/sam3.pt`; otherwise SAM 3 downloads from Hugging Face using your own authenticated access. Run `f2g-pose predict` once per mask. A zero-instance result produces only `masks.json`; adjust the prompt or confidence threshold (`--confidence`) and inspect the results. SAM 3 produces RGB masks, which are also used to select pixels from aligned depth; a separate depth mask is unnecessary.

For a remote server, forward the port from your local terminal:

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

## Input formats

### Head camera HTTP capture

For the local `head_rgb_depth_http` bridge, start the demo with
`--head-camera-url http://6.6.7.100:8765/frame`. Click **Capture head RGB-D frame**
to fill the RGB, millimeter depth and RGB camera intrinsics, then generate SAM 3
masks, select the target, confirm its mask and click **Estimate pose**.
The bridge must publish depth aligned to color and include the RGB `camera_info`;
the demo rejects unaligned frames, inconsistent dimensions and RGB/depth timestamps
more than 100 ms apart. See the bridge's README for the aligned depth topic setup.
The output `T_cam_obj` is in the head camera optical frame. To express it in
`base_link`, use the validated head hand-eye calibration plus head joint angles
at the image timestamp; the fixed hand-eye transform alone is insufficient.

### ROS pose publisher for `head_object_grasp.py`

The F2G-Pose ROS bridge loads F2G-Pose/RADIO and the separate SAM 3 service once
at startup, then continuously captures RGB-D, estimates poses, and publishes
`geometry_msgs/PoseStamped` to `/object_pose`. If the target RGB and depth still
match the cached mask, it reuses that mask; otherwise SAM 3 segments the new
frame. Wait for `F2G-Pose 和 SAM 3 已加载` before starting a grasp. The default
mode does not wait for `/f2g_pose/request`.

In addition to `/object_pose`, it publishes a one-shot `std_msgs/Bool(data=True)`
event on `/object_pose_changed` when the estimated object position moves at least
20 mm from the last notified position. The threshold and topic can be changed
with `--position-change-threshold-mm` and `--change-topic`. The first valid pose
only establishes the reference and does not emit a change event. Other ROS nodes
can subscribe to this event to refresh a grasp target or stop an active action:

```bash
ros2 topic echo /object_pose_changed
```

For compatibility with the old request-driven behavior, add `--request-driven`.
In that mode the bridge subscribes to `/f2g_pose/request` (`std_msgs/Empty`) and
runs one inference window for each request.
It publishes only camera-space observations; the grasp program resolves the
calibrated head TF into `base_link` and performs its existing checks. The bridge
does not send IK or gripper commands. Keep the head camera and object still
through the capture; if the target appearance or depth changes substantially,
the bridge stops rather than reusing the old mask. Multiple detections require
an explicit `--instance-index` (left to right, starting from 1).

On the local machine, source the robot's ROS environment and start the head
calibration TF broadcaster with `calibration_data/head_extrinsic.yaml`. Ensure
`/tj/joint_states` is live and `base_link -> luna_head_calibrated_optical` is
available at the image time. In another terminal, start the read-only bridge:

```bash
source /home/liuqian/copy/Apex_Deploy/apex_ros_env.sh
source /home/liuqian/WBC/Apex_Deploy/install/setup.bash
export PATH=/usr/bin:$PATH PYTHONNOUSERSITE=1
ros2 run luna_handeye_calibration camera_tf_broadcaster --ros-args \
  -r __node:=luna_head_camera_tf \
  -p calibration_file:=/home/liuqian/WBC/Apex_Deploy/calibration_data/head_extrinsic.yaml \
  -p model_path:=/home/liuqian/copy/Apex_Deploy/robot_node/marvin_description/mjcf/Gento_Luna/Gento_Luna.xml \
  -p joint_topic:=/tj/joint_states
```

In a second terminal:

```bash
cd /home/liuqian/copy/F2G-Pose
source /home/liuqian/copy/Apex_Deploy/apex_ros_env.sh
source /home/liuqian/WBC/Apex_Deploy/install/setup.bash
/usr/bin/python3 scripts/publish_head_pose.py --prompt 'water bottle' \
  --instance-index 1 --max-frames 10
```

The command above continuously publishes poses. To use the old request-driven
mode instead:

```bash
/usr/bin/python3 scripts/publish_head_pose.py --prompt 'water bottle' \
  --instance-index 1 --max-frames 10 --request-driven
```

Start the grasp program in preview mode first, using the same ROS environment:

```bash
cd /home/liuqian/copy/test
/usr/bin/python3 head_object_grasp.py --side left \
  --object-pose-topic /object_pose --pose-request-topic /f2g_pose/request \
  --capture-timeout 45 \
  --model-xml /home/liuqian/copy/Apex_Deploy/robot_node/marvin_description/mjcf/Gento_Luna/Gento_Luna.xml
```

The bridge estimates the image time from the HTTP server's `rgb_age_s` at local
receipt, matching the calibration's `ros_host_arrival` time convention. Check
host clock synchronization and TF latency on the actual robot. The image mask
and object origin are not a grasp pose: inspect the previewed `base_link` goal,
choose a measured grasp offset, and validate reachability before enabling
`--execute`. The calibration's recorded MJCF checksum matches the
`/home/liuqian/copy/Apex_Deploy` model; the WBC model currently has a different
checksum. Confirm which model controls the physical robot and revalidate the
calibration before real motion. In continuous mode, the publisher ignores
duplicate timer ticks while inference is active and starts another inference
window after the current one finishes. Each window publishes at most 10 pose
frames by default, enough for the grasp program's five stable samples. On a
target change during inference, that window stops; the next window generates a
fresh mask. Stopping the publisher releases both models.

| Input | Format |
| --- | --- |
| RGB | PNG/JPEG file, or an `H×W×3` uint8 NumPy array in RGB order |
| Depth | Raw uint16 PNG or floating-point NPY; `H×W` |
| Mask | One foreground object, `H×W`, values 0/1 or 0/255 |
| Intrinsics | A 3×3 pinhole matrix or `[fx, fy, cx, cy]`, for the supplied image |
| Depth scale | Meters per raw unit: `0.001` for millimeters, `1.0` for meters |

RGB, depth, and mask must be aligned and have the same dimensions. The crop and outlier filtering need at least 50 valid foreground depth points. Focal lengths must be positive, with zero skew.

### Python

```python
import numpy as np
from PIL import Image
from f2g_pose import PoseEstimator

rgb = np.asarray(Image.open("example/rgb.png").convert("RGB"))
depth = np.load("example/depth.npy")  # this example stores meters
mask = np.asarray(Image.open("example/mask.png"))
K = np.loadtxt("example/K.txt")

estimator = PoseEstimator("weights/f2g-pose.pt")
result = estimator.predict(rgb, depth, K, mask, depth_scale=1.0)
```

| Output | Meaning |
| --- | --- |
| `T_cam_obj` | 4×4 object-to-camera transform: `p_cam = R @ p_obj + t` |
| `rotation` | 3×3 rotation matrix |
| `translation_m` | Translation in meters |
| `size_m` | Object dimensions in meters |
| `points_cam_m` | Completed point cloud, already in camera coordinates |

Camera axes are X right, Y down, and Z forward. Translation and shape restore the observed point-cloud center. Set `return_shape=False` to return pose and size only.

### Command line

```bash
f2g-pose predict --checkpoint weights/f2g-pose.pt \
  --rgb example/rgb.png --depth example/depth.npy --mask example/mask.png \
  --intrinsics example/K.txt --depth-scale 1 --output outputs/example
```

This writes `pose.json` and `shape_camera.ply`. Add `--pose-only` to skip shape completion. For millimeter PNG depth, use `--depth-scale 0.001`.

To create an example from your Omni6DPose data:

```bash
python scripts/make_example.py --data-root "$SOPE_ROOT" --split sope --output example
```

A ready-to-use real teapot sample, with CLI, Gradio, and single-object evaluation commands, is in [examples/rope](../examples/rope/README.md).

## Dataset paths

Use the official Omni6DPose layout:

```text
Omni6DPose/
├── SOPE/<part>/train/... and test/...
├── ROPE/<sequence>/...
├── Meta/obj_meta.json and real_obj_meta.json
└── original_sampling_objs/<object_id>/Aligned.npy
```

Set the paths for your machine:

```bash
export SOPE_ROOT=/path/to/Omni6DPose/SOPE
export ROPE_ROOT=/path/to/Omni6DPose/ROPE
export ALIGNED_SHAPES_ROOT=/path/to/Omni6DPose/original_sampling_objs
```

Metadata is read from the sibling `Meta` directory. Use `--object-meta` for a different location. All commands accept `--cache-dir` to choose where Torch, Hugging Face, and Gradio store caches.

## Training

The SOPE configuration uses **100 epochs** and a target global batch size of **512**. RADIO stays frozen while the feature-fusion layers, transformer, and output heads train jointly. The loader uses eight object draws per frame, DZI crop augmentation, and mask deformation.

On four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m f2g_pose.cli train \
  --config configs/sope.yaml --data-root "$SOPE_ROOT" \
  --shapes-root "$ALIGNED_SHAPES_ROOT" --batch-size 128 --output outputs/train
```

`--batch-size` is per GPU. Effective batch size is `batch-size × GPUs × accumulation`. For a smaller GPU, lower the per-GPU batch and increase `--accumulation`; training requires at least two samples per GPU.

A short single-GPU run:

```bash
f2g-pose train --config configs/sope.yaml --checkpoint weights/f2g-pose.pt \
  --data-root "$SOPE_ROOT" --shapes-root "$ALIGNED_SHAPES_ROOT" \
  --batch-size 2 --limit 16 --max-steps 2 --epochs 1 --output outputs/smoke
```

Resume that run for one more update:

```bash
f2g-pose train --config configs/sope.yaml --data-root "$SOPE_ROOT" \
  --shapes-root "$ALIGNED_SHAPES_ROOT" --batch-size 2 --limit 16 --epochs 1 \
  --resume outputs/smoke/checkpoint-last.pt --max-steps 3 --output outputs/smoke
```

Training writes losses and gradient checks to `training.jsonl`, the sample list to `train_manifest.jsonl`, and resumable state to `checkpoint-last.pt`. Use `--checkpoint` to initialize model weights; use `--resume` to restore optimizer and training state from your own training run.

## Evaluation

The following commands use the precision settings of the reported full-test results:

```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

f2g-pose eval --checkpoint weights/f2g-pose.pt --data-root "$SOPE_ROOT" \
  --split sope --protocol full --precision bf16 --batch-size 128 \
  --workers 4 --metric-workers 4 --seed 0 --output outputs/sope_full

f2g-pose eval --checkpoint weights/f2g-pose.pt --data-root "$ROPE_ROOT" \
  --split rope --protocol full --precision bf16 --batch-size 128 \
  --workers 4 --metric-workers 4 --seed 0 --output outputs/rope_full
```

Use a new output directory for each run. `--protocol historical` selects the earlier subset protocol: the first 5,000 SOPE test frames with one seeded object each, or every 50th ROPE frame within each sequence. See [Evaluation](EVALUATION.md) for coverage, metrics, and precision details.

Each run saves configuration and environment information, sample manifests, excluded samples with reasons, raw predictions, `per_category.csv`, and `metrics.json`. AUC/VUS are fractions in JSON and percentages in CSV; errors are degrees and centimeters.

### Evaluate an existing manifest across GPUs

Reuse a full run's sample manifest for subsequent checkpoints or distributed evaluation. For example, two shards on two GPUs:

```bash
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 f2g-pose eval \
    --checkpoint weights/f2g-pose.pt --data-root "$SOPE_ROOT" \
    --split sope --protocol full --precision bf16 --batch-size 128 \
    --manifest outputs/sope_full/candidate_manifest.jsonl \
    --metadata-exclusions outputs/sope_full/metadata_excluded.jsonl \
    --shard-count 2 --shard-index "$shard" --output "outputs/shard$shard" &
done
wait

python scripts/merge_evaluations.py --shards outputs/shard0 outputs/shard1 \
  --expected-manifest outputs/sope_full/candidate_manifest.jsonl --output outputs/merged
```

Keep the original manifest beside its `run.json`: the merger uses both to verify full-test coverage. It also checks sample counts, exclusions, and model/code hashes. Keep the shard directories, which hold the raw predictions referenced by the merged result.

## Benchmarking

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 f2g-pose benchmark \
  --checkpoint weights/f2g-pose.pt --rgb example/rgb.png \
  --depth example/depth.npy --mask example/mask.png --intrinsics example/K.txt \
  --depth-scale 1 --precision fp32 --warmup 30 --iterations 200 \
  --output outputs/benchmark
```

The benchmark reports full and pose-only model times, preprocessing time, FPS, and peak GPU memory. SAM is timed separately in the demo. The command above uses the reference batch-1 FP32 settings.
