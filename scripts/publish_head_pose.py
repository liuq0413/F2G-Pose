#!/usr/bin/python3
# 用途：连续调用 F2G-Pose，发布头部相机系物体 PoseStamped；本脚本不发布运动指令。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && /usr/bin/python3 scripts/publish_head_pose.py --prompt 'water bottle'
# 多目标：在命令中重复 --prompt 或使用逗号分隔，并从 /object_poses 读取名称和位姿 JSON。
"""ROS 环境与 F2G-Pose 虚拟环境之间的连续位姿桥接。"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty, String


def pose_message(value, now, max_age):
    stamp = float(value['stamp'])
    position = value['position_m']
    quaternion = value['quaternion_xyzw']
    if (not math.isfinite(stamp) or not -.05 <= now - stamp <= max_age or
            len(position) != 3 or len(quaternion) != 4 or
            not all(math.isfinite(float(x)) for x in position + quaternion) or
            abs(math.sqrt(sum(float(x)**2 for x in quaternion)) - 1) > .01 or
            value['frame_id'] != 'luna_head_calibrated_optical'):
        raise ValueError('位姿时间戳、坐标系、位置或四元数无效。')
    message = PoseStamped()
    seconds = math.floor(stamp)
    nanoseconds = int(round((stamp - seconds) * 1e9))
    if nanoseconds == 1_000_000_000:
        seconds += 1
        nanoseconds = 0
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = nanoseconds
    message.header.frame_id = value['frame_id']
    message.pose.position.x, message.pose.position.y, message.pose.position.z = position
    (message.pose.orientation.x, message.pose.orientation.y,
     message.pose.orientation.z, message.pose.orientation.w) = quaternion
    return message


def position_change_mm(previous, current):
    """返回两个三维位置之间的欧氏距离，单位为毫米。"""
    return math.dist(previous, current) * 1000.0


class HeadPosePublisher(Node):
    def __init__(self, args):
        super().__init__('f2g_head_pose_publisher')
        self.args = args
        self.started = False
        self.process = None
        self.lock = threading.Lock()
        self.stopping = False
        self.publisher = self.create_publisher(PoseStamped, args.pose_topic, qos_profile_sensor_data)
        self.objects_publisher = self.create_publisher(
            String, args.objects_topic, 10)
        self.change_publisher = self.create_publisher(
            Bool, args.change_topic, 10)
        self.last_change_position = None
        if not args.continuous:
            self.create_subscription(Empty, args.request_topic, self.on_request,
                                     qos_profile_sensor_data)
        self.start_worker()
        if args.continuous:
            # 每轮推理结束后由定时器触发下一轮，避免在 worker 读取线程中重入 stdin。
            self.create_timer(.2, self.continuous_tick)
            self.get_logger().info(
                f'连续推理并发布到 {args.pose_topic}；不等待 {args.request_topic}，不执行机器人运动')
        else:
            self.get_logger().info(
                f'等待 {args.request_topic} 请求，发布到 {args.pose_topic}；'
                f'位置变化通知发布到 {args.change_topic}；不执行机器人运动')

    def start_worker(self):
        a = self.args
        command = [str(a.worker_python), '-u', '-m', 'f2g_pose.head_pose_worker',
                   '--camera-url', a.camera_url, '--checkpoint', str(a.checkpoint),
                   '--radio-repository', str(a.radio_repository),
                   '--radio-checkpoint', str(a.radio_checkpoint),
                   '--sam3-python', str(a.sam3_python),
                   '--sam3-checkpoint', str(a.sam3_checkpoint),
                   '--duration', str(a.duration),
                   '--max-frames', str(a.max_frames), '--serve']
        for prompt in a.prompts:
            command += ['--prompt', prompt]
        if a.instance_index is not None:
            command += ['--instance-index', str(a.instance_index)]
        self.process = subprocess.Popen(command, cwd=a.project_root, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        threading.Thread(target=self.read_worker, args=(self.process,), daemon=True).start()

    def on_request(self, _message):
        with self.lock:
            if self.started or self.stopping:
                return
            if self.process is None or self.process.poll() is not None:
                self.start_worker()
            self.started = True
            try:
                self.process.stdin.write('{"request":true}\n')
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.started = False
                self.get_logger().error(f'无法请求位姿推理：{exc}')

    def continuous_tick(self):
        """连续模式下启动下一轮，也负责在 worker 异常退出后重启它。"""
        if self.stopping or not self.args.continuous:
            return
        self.on_request(None)

    def read_worker(self, process):
        try:
            for line in process.stdout:
                try:
                    payload = json.loads(line)
                    if payload.get('ready'):
                        self.get_logger().info('F2G-Pose 和 SAM 3 已加载，可开始发布位姿')
                        continue
                    if payload.get('done'):
                        with self.lock:
                            self.started = False
                        if self.args.continuous:
                            self.get_logger().info('本轮推理结束；模型保持加载，继续发布下一轮位姿')
                        else:
                            self.get_logger().info('本次请求结束；模型保持加载，可继续处理下一次请求')
                        continue
                    if 'error' in payload:
                        object_name = payload.get('object_name')
                        prefix = f'{object_name}：' if object_name else ''
                        self.get_logger().error(f'位姿推理失败：{prefix}{payload["error"]}')
                        continue
                    pose_value = payload['pose']
                    pose = pose_message(pose_value, time.time(), self.args.max_pose_age)
                    object_name = str(payload.get('object_name', self.args.prompts[0]))
                    if object_name not in self.args.prompts:
                        raise ValueError(f'未知物体名称：{object_name}')
                except (ValueError, KeyError, TypeError) as exc:
                    if not line.strip().startswith('{'):
                        self.get_logger().debug(f'模型输出：{line.strip()[-300:]}')
                    else:
                        self.get_logger().warning(f'忽略无效或过期的推理结果：{exc}')
                    continue
                object_record = {'object_name': object_name, 'pose': pose_value}
                object_message = String()
                object_message.data = json.dumps(object_record, ensure_ascii=False,
                                                 allow_nan=False)
                self.objects_publisher.publish(object_message)
                # 保留原有 PoseStamped 接口，只把活动物体发送给抓取程序。
                if object_name != self.args.active_object:
                    self.get_logger().info(
                        f'已发布 {object_name} 位姿到 {self.args.objects_topic}：'
                        f'{pose_value["position_m"]}')
                    continue
                self.publisher.publish(pose)
                current_position = (
                    float(pose.pose.position.x), float(pose.pose.position.y),
                    float(pose.pose.position.z))
                if (self.last_change_position is not None and
                        position_change_mm(self.last_change_position, current_position) >=
                        self.args.position_change_threshold_mm):
                    changed = Bool()
                    changed.data = True
                    self.change_publisher.publish(changed)
                    self.get_logger().warning(
                        f'检测到物体位置变化，已通知 {self.args.change_topic}：'
                        f'{position_change_mm(self.last_change_position, current_position):.1f} mm')
                    # 以最近一次通知的位置为基准，避免每一帧重复通知同一次变化。
                    self.last_change_position = current_position
                elif self.last_change_position is None:
                    # 第一帧只建立基准，不把初次检测误报为“位置变化”。
                    self.last_change_position = current_position
                self.get_logger().info(
                    f'已发布 {object_name} 头部相机系物体位置：{pose_value["position_m"]}')
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.started = False
            if not self.stopping:
                self.get_logger().error(f'位姿服务退出，退出码={process.poll()}；下次请求将重启')

    def stop(self):
        self.stopping = True
        process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root', type=Path, default=root)
    parser.add_argument('--worker-python', type=Path, default=root / '.venv/bin/python')
    parser.add_argument('--camera-url', default='http://6.6.7.100:8765/frame')
    parser.add_argument('--checkpoint', type=Path, default=root / 'weights/f2g-pose.pt')
    parser.add_argument('--radio-repository', type=Path, default=root / 'third_party_local/RADIO')
    parser.add_argument('--radio-checkpoint', type=Path, default=root / 'weights/radio-v2.5-l_half.pth.tar')
    parser.add_argument('--sam3-python', type=Path, default=root / 'third_party_local/.venv-sam3/bin/python')
    parser.add_argument('--sam3-checkpoint', type=Path, default=root / 'weights/sam3/sam3.pt')
    parser.add_argument('--request-topic', default='/f2g_pose/request')
    parser.add_argument('--pose-topic', default='/object_pose')
    parser.add_argument('--objects-topic', default='/object_poses',
                        help='所有目标名称和位姿的 JSON 话题，消息类型为 std_msgs/String')
    parser.add_argument('--change-topic', default='/object_pose_changed',
                        help='物体位置变化通知话题，消息类型为 std_msgs/Bool')
    parser.add_argument('--position-change-threshold-mm', type=float, default=20.,
                        help='触发位置变化通知的距离阈值，默认 20 mm')
    parser.add_argument('--prompt', action='append', default=None,
                        help='目标提示词；可重复传入，也可用逗号或分号分隔多个物体')
    parser.add_argument('--active-object',
                        help='继续发布到 /object_pose 并供抓取的物体名称；默认使用第一个提示词')
    parser.add_argument('--request-driven', action='store_false', dest='continuous',
                        help='兼容旧模式：仅收到 /f2g_pose/request 后运行一轮')
    parser.set_defaults(continuous=True)
    parser.add_argument('--instance-index', type=int)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--max-frames', type=int, default=10,
                        help='每次请求最多发布的位姿帧数，默认 10；抓取程序需要多帧稳定观测')
    parser.add_argument('--max-pose-age', type=float, default=.5)
    args = parser.parse_args(argv)
    args.prompts = []
    for value in args.prompt or ['water bottle']:
        for item in value.replace(';', ',').split(','):
            item = item.strip()
            if item and item not in args.prompts:
                args.prompts.append(item)
    if not args.prompts:
        parser.error('至少需要一个目标提示词。')
    args.active_object = args.active_object or args.prompts[0]
    if (args.active_object not in args.prompts or
            args.instance_index is not None and args.instance_index < 1 or
            args.max_frames < 1 or not math.isfinite(args.position_change_threshold_mm) or
            args.position_change_threshold_mm <= 0 or not all(
            math.isfinite(v) and v > 0 for v in (args.duration, args.max_pose_age))):
        parser.error('请求参数无效')
    for path in (args.worker_python, args.checkpoint, args.radio_repository,
                 args.radio_checkpoint, args.sam3_python, args.sam3_checkpoint):
        if not path.exists():
            parser.error(f'缺少文件或目录：{path}')
    return args


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = HeadPosePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
