# F2G-Pose: Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation

**TL;DR:** F2G-Pose estimates an object’s 6D pose, metric size, and completed 3D shape from a single segmented RGB-D observation in one forward pass.

<p align="center">
  <strong><a href="https://jrryzh.github.io/F2G-Pose/">Project page</a></strong> ·
  <a href="https://arxiv.org/abs/2610.31623">arXiv</a> ·
  <a href="https://arxiv.org/pdf/2610.31623">Paper PDF</a> ·
  <a href="https://huggingface.co/J3rr1/F2G-Pose/resolve/main/f2g-pose.pt?download=true">Weights</a>
</p>

<p align="center">
  <a href="docs/media/film.mp4"><img src="docs/images/film.jpg" width="720" alt="Preview of the latest F2G-Pose 90-second film"></a><br>
  <a href="docs/media/film.mp4">Watch the 90-second film (MP4)</a>
</p>

F2G-Pose lifts visual foundation features onto observed 3D points. Inputs are an RGB image, aligned depth, camera intrinsics, and a foreground mask; no category label or object-specific CAD model is needed.

- 🛠️ [Installation](docs/INSTALL.md)
- ▶️ [Demo & inference](docs/USAGE.md#interactive-demo)
- 🤖 [Head-camera RGB-D pose and grasp](#head-camera-rgb-d-pose-and-grasp)
- 🧪 [Training](docs/USAGE.md#training)
- 📊 [Evaluation](docs/EVALUATION.md)
- 🚀 [Deployment](docs/DEPLOYMENT.md)

![F2G-Pose pipeline: RGB and partial point cloud features are fused by a geometry-aware MoE Transformer, then decoded into pose, size, and completed shape.](assets/pipeline.png)

## How it works

1. **Lift appearance into 3D.** A frozen RADIO backbone supplies dense image features, which are projected onto the observed point cloud.
2. **Combine appearance and geometry.** DGCNN builds local geometric tokens; a Mixture-of-Experts Transformer captures object-level structure.
3. **Predict pose, size, and shape.** Parallel heads estimate the object-to-camera transform, metric dimensions, and a completed point cloud. Use pose-only mode when you only need pose and size.

The model is trained on synthetic SOPE data and evaluated on both SOPE and real-world ROPE data.

## Results in pictures

![ROPE examples comparing RGB input, ground truth, GenPose++, F2G-Pose pose estimates, and partial-to-complete point clouds.](assets/rope_results.jpg)

**Pose and shape on ROPE.** Columns show RGB, ground truth, GenPose++, F2G-Pose, and the observed-to-completed point cloud. These examples are from the paper.

<details>
<summary>More ROPE examples</summary>

![Additional pose and shape completion examples from the paper, spanning varied objects and cluttered scenes.](assets/rope_more.png)

</details>

### Robot manipulation

<p align="center">
  <img src="assets/robot_manipulation.png" width="720" alt="ARX robot pick-and-place trials using F2G-Pose estimates.">
</p>

The paper also demonstrates pose-guided pick-and-place with an ARX robot, a parallel-jaw gripper, and an RGB-D camera. Completed shapes provide geometry for collision checking.

## Try it

Follow [Installation](docs/INSTALL.md) to set up the environment, RADIO, and model weights. Then launch the interactive demo:

```bash
f2g-pose demo --checkpoint weights/f2g-pose.pt --port 7860
```

Open **http://127.0.0.1:7860**, upload RGB, depth, and a mask, then enter the camera intrinsics and depth units. The demo displays pose axes, a 3D box, and the completed shape, with JSON and PLY downloads. Optional [SAM prompting](docs/USAGE.md#interactive-demo) lets you select the object with clicks or a box.

Optional [SAM 3 automatic masks](docs/USAGE.md#automatic-sam-3-masks) find all instances matching a text prompt such as `water bottle` and save a separate mask for each one. SAM 3 runs in its own environment.

## 头部相机 RGB-D 位姿与抓取

完整流程由以下程序组成：机器人端 RGB-D HTTP 服务、头部相机手眼 TF 发布器、
F2G-Pose Web 界面和 ROS 位姿发布器。位姿发布器在头部相机坐标系中估计物体
位姿，并发布到 `/object_pose`；抓取程序再通过相机 TF 转换到 `base_link`。
F2G-Pose 和位姿发布器本身不会发送机器人运动指令。

下面的命令需要在不同终端中运行。实际部署路径不同时，请替换对应路径。

### 1. 机器人端：RGB-D 图像服务

在机器人电脑上启动对齐到彩色图像的 RealSense 深度流：

```bash
source /opt/kernelmind/apex/install/setup.bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/apex/cyclonedds.xml

ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

在机器人电脑的另一个终端启动 HTTP 图像服务：

```bash
source /opt/kernelmind/apex/install/setup.bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/apex/cyclonedds.xml

cd /home/nvidia/head_rgb_depth_http
/usr/bin/python3 robot_server.py --host 6.6.7.100 --port 8765 \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw
```

服务提供类似 `http://6.6.7.100:8765/frame` 的接口，必须同时返回 RGB、与彩色
图对齐的深度图以及 RGB 相机内参。启动推理前可检查接口：

```bash
curl -sS http://6.6.7.100:8765/frame -o /tmp/head_frame.json
```

### 2. 机器人端：头部相机手眼 TF

在机器人上启动经过标定的 TF 发布器。模型文件和标定文件必须属于同一套机器人：

```bash
source /opt/ros/humble/setup.bash
source /opt/kernelmind/apex/install/setup.bash
source /home/nvidia/luna_handeye_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///etc/apex/cyclonedds.xml

ros2 run luna_handeye_calibration camera_tf_broadcaster --ros-args \
  -r __node:=luna_head_camera_tf \
  -p calibration_file:=/home/nvidia/luna_handeye_ws/calibration_data/head_extrinsic.yaml \
  -p model_path:=/opt/kernelmind/apex/install/marvin_description/share/marvin_description/mjcf/Gento_Luna/Gento_Luna.xml \
  -p joint_topic:=/tj/joint_states
```

检查 TF 是否存在：

```bash
ros2 run tf2_ros tf2_echo base_link luna_head_calibrated_optical
```

### 3. 主机端：F2G-Pose Web 界面

在主机上启动 Web 界面，并指定远程 RGB-D 接口和独立的 SAM 3 Python 环境：

```bash
cd /home/liuqian/copy/F2G-Pose
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  .venv/bin/f2g-pose demo \
  --checkpoint weights/f2g-pose.pt \
  --radio-repository "$PWD/third_party_local/RADIO" \
  --radio-checkpoint "$PWD/weights/radio-v2.5-l_half.pth.tar" \
  --sam3-python "$PWD/third_party_local/.venv-sam3/bin/python" \
  --sam3-checkpoint "$PWD/weights/sam3/sam3.pt" \
  --head-camera-url http://6.6.7.100:8765/frame \
  --port 7861
```

打开 [http://127.0.0.1:7861](http://127.0.0.1:7861)，点击 **Capture head RGB-D
frame**，输入 `water bottle` 等提示词，生成 SAM 3 掩码，选择并确认目标掩码，
最后估计位姿。这个界面用于检查 RGB-D 对齐、相机内参、深度单位和掩码质量，
不会控制机器人。

### 4. 主机端：连续发布 ROS 位姿

默认模式会持续启动推理窗口。当 RGB-D 图像仍与缓存掩码匹配时复用掩码，并发布
稳定的相机坐标系物体位姿：

```bash
cd /home/liuqian/copy/F2G-Pose
source /home/liuqian/copy/Apex_Deploy/apex_ros_env.sh
source /home/liuqian/WBC/Apex_Deploy/install/setup.bash
export PATH=/usr/bin:$PATH
export PYTHONNOUSERSITE=1

/usr/bin/python3 scripts/publish_head_pose.py \
  --prompt 'water bottle' \
  --instance-index 1 \
  --max-frames 10
```

程序在头部相机光学坐标系下向 `/object_pose` 发布 `geometry_msgs/PoseStamped`。
当估计位置变化达到 20 mm 或以上时，会在 `/object_pose_changed` 发布一次
`std_msgs/Bool(data=True)`。可以这样查看：

```bash
ros2 topic echo /object_pose
ros2 topic echo /object_pose_changed
```

也可以一次识别多个物体类别。重复使用 `--prompt`，或在一个提示词中用逗号分隔；
第一个提示词继续作为抓取程序使用的活动目标发布到 `/object_pose`，所有目标的名称和位姿
会以 JSON `std_msgs/String` 逐条发布到 `/object_poses`：

```bash
/usr/bin/python3 scripts/publish_head_pose.py \
  --prompt 'water bottle' --prompt 'cup' \
  --active-object 'water bottle'
ros2 topic echo /object_poses
```

消息中的 `object_name` 是提示词名称，`pose.position_m` 和
`pose.quaternion_xyzw` 是对应的头部相机坐标系位姿。`--active-object` 用于选择继续
发送到 `/object_pose` 的目标，默认是第一个提示词。
如果某个提示词没有检测到目标，只报告该提示词的错误，其他已检测到的物体仍会继续发送。

使用 `--instance-index 2`、`3` 等参数可以选择其他 SAM 3 检测结果。若要恢复旧的
按请求推理模式，增加 `--request-driven`；此时每收到一条 `/f2g_pose/request`
消息才启动一轮推理。第一帧有效位姿只建立位置变化基准，不会触发变化通知。

### 5. 可选：抓取程序

先以预览模式启动抓取程序。它监听 `/object_pose`，使用
`base_link -> luna_head_calibrated_optical`，没有 `--execute` 时不会移动机器人：

```bash
cd /home/liuqian/copy/test/move_agent/robot_server
source /home/liuqian/copy/Apex_Deploy/apex_ros_env.sh
export PYTHONNOUSERSITE=1

/usr/bin/python3 -B -m agent_tools.scheduler \
  agent_tools/head_object_grasp_task.json
```

检查生成的命令和目标位姿后，再显式加入 `--execute` 执行：

```bash
/usr/bin/python3 -B -m agent_tools.scheduler \
  agent_tools/head_object_grasp_task.json --execute
```

任务通常使用 `side: left` 和 `grasp_offset_mm: [0, 0, -40]`，测试时可以使用
`position-only` 或 `reach-only`。执行前确认 TF、控制源、工作空间和急停均正常。
完整任务及 FastAPI 说明见机器人端
`/home/liuqian/copy/test/move_agent/robot_server/README.md`，位姿桥接说明见
[`docs/USAGE.md`](docs/USAGE.md)。

For a ready-to-use real input, try the [bundled ROPE teapot sample](examples/rope/README.md). It includes aligned RGB, metric depth, a target mask, camera intrinsics, and a one-object evaluation manifest.

For Python applications:

```python
from f2g_pose import PoseEstimator

estimator = PoseEstimator("weights/f2g-pose.pt")
result = estimator.predict(rgb, depth, K, mask, depth_scale=0.001)

T_cam_obj = result["T_cam_obj"]       # 4×4 object-to-camera transform
size_m = result["size_m"]             # object dimensions in meters
points_cam = result["points_cam_m"]   # completed shape in camera coordinates
```

Here, `rgb`, `depth`, and `mask` are aligned NumPy arrays; `K` is a 3×3 matrix or `[fx, fy, cx, cy]`. Use `depth_scale=0.001` for millimeter depth or `1.0` for meters. See [input formats and coordinates](docs/USAGE.md#input-formats) for a complete example.

**Weights:** [download `f2g-pose.pt`](https://huggingface.co/J3rr1/F2G-Pose/resolve/main/f2g-pose.pt?download=true) (SHA-256 `bff06bd58a42fd805d8232f9ab7b4612b8eb833edf6b6cccd23e5040a8c9ca18`). RADIO is downloaded separately.

## Benchmark results

Full-test results for the provided model:

| Dataset | AUC25 ↑ | AUC50 ↑ | AUC75 ↑ | Mean rotation ↓ | Mean translation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOPE | 60.24 | 44.73 | 17.69 | 14.73° | 0.824 cm |
| ROPE | 46.75 | 28.19 | 6.17 | 26.99° | 1.181 cm |

AUC values are percentages. See [Evaluation](docs/EVALUATION.md) for VUS, evaluation settings, and reproduction commands.

On RTX 4090, batch-1 model inference runs at **36.6 FPS** with shape completion and **37.2 FPS** in pose-only mode. These FP32 measurements use 30 warmup and 200 synchronized iterations; preprocessing and SAM are excluded.

## Documentation

| Task | Guide |
| --- | --- |
| Set up CUDA extensions and weights | [Installation](docs/INSTALL.md) |
| Run the demo or predict on your own RGB-D data | [Usage](docs/USAGE.md) |
| Train, resume, or evaluate on SOPE / ROPE | [Training and evaluation commands](docs/USAGE.md#training) |
| Understand metrics and runtime | [Evaluation](docs/EVALUATION.md) |
| Run with Docker or prepare a Hugging Face Space | [Deployment](docs/DEPLOYMENT.md) |
| Check model inputs, outputs, and limitations | [Model card](docs/MODEL_CARD.md) |
| Inspect implementation checks | [Validation](docs/VALIDATION.md) |

## Citation

If you use F2G-Pose in your research, please cite:

```bibtex
@article{zhang2026f2gpose,
  title={F2G-Pose: Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation},
  author={Zhang, Jinyu and Lin, Haitao and Hou, Jiashu and Xue, Xiangyang and Fu, Yanwei},
  journal={arXiv preprint arXiv:2610.31623},
  year={2026},
  eprint={2610.31623},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.31623}
}
```

## Acknowledgments and license

We thank the authors of RADIO, PoinTr / AdaPoinTr, PointNet++, Omni6DPose, GDR-Net, CenterNet, NOCS, SGPA, FS-Net, and Segment Anything for their code, datasets, and ideas.

F2G-Pose integration code uses [Apache-2.0](LICENSE). Third-party components retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). RADIO is installed and downloaded separately under NVIDIA's terms.
